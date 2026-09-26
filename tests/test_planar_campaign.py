"""Sensor/reward/curriculum contract tests followed by real PPO/export/replay."""
from dataclasses import replace
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import mujoco
import numpy as np
import pytest
import torch

from k1_motion.export import export_checkpoint
from k1_motion.learning import ActorCritic, Policy, TrainConfig, train
from k1_motion.model_transfer import initialize_model, input_mapping
from k1_motion.observations import ObservationBuilder, observation_contract
from k1_motion.tracking_env import TrackerEnv
from k1_motion.training_curriculum import TrainingCurriculum
from test_training import make_library


def test_planar_feedback_world_coordinates_heading_and_scalar_batch_parity(tmp_path):
    _, library = make_library(tmp_path)
    env = TrackerEnv(library, 1, 'cpu', history=3, observation_profile='planar')
    try:
        zero = torch.zeros(1, dtype=torch.long)
        env.reset(clips=zero, frames=zero)
        robot = env.physics.robots[0]
        robot.data.qpos[:2] = [4., -2.]
        robot.data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        robot.data.qvel[:3] = [.3, -.4, 0]
        robot.data.qvel[3:6] = [0, 0, .7]
        mujoco.mj_forward(robot.model, robot.data)
        env.history.reset()
        batch, _ = env.observe()
        reference = robot.neutral_reference()
        builder = ObservationBuilder(robot.neutral, history=3, profile='planar')
        scalar = builder.build(robot.state(0), reference, np.zeros(22), 0)
        np.testing.assert_allclose(batch[0], scalar, atol=2e-6, rtol=0)
        # Appended features: actual/ref XY, body displacement error, world
        # actual/ref XY velocity, body velocity error, yaw sin/cos pairs, rates.
        extra = scalar.reshape(3, -1)[-1, 135:-1]
        np.testing.assert_allclose(extra[:4], [4, -2, 0, 0], atol=1e-6)
        np.testing.assert_allclose(extra[4:6], [2, 4], atol=1e-6)
        np.testing.assert_allclose(extra[6:8], [.3, -.4], atol=1e-6)
        np.testing.assert_allclose(extra[12:16], [0, 1, 1, 0], atol=1e-6)
        np.testing.assert_allclose(extra[18:20], [.7, 0], atol=1e-6)
        missing = replace(robot.state(0), root_position=None, linear_velocity=None)
        with pytest.raises(ValueError, match='odometry'):
            builder.build(missing, reference, np.zeros(22), 0)
        with pytest.raises(ValueError, match='finite'):
            replace(robot.state(0), linear_velocity=np.array([np.nan, 0, 0]))
    finally:
        env.close()


def test_planar_actor_reads_present_feedback_and_never_future_reference(tmp_path):
    _, library = make_library(tmp_path)
    env = TrackerEnv(library, 2, 'cpu', history=3, reference_storage='packed', observation_profile='planar')
    try:
        zero = torch.zeros(2, dtype=torch.long)
        env.reset(clips=zero, frames=zero)
        env.history.reset()
        before, privileged = env.observe()
        env.library.values['root_position'][2:, :2] += 10
        env.library.values['root_velocity'][2:, :3] += 4
        env.history.reset()
        after, privileged_after = env.observe()
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        assert not torch.equal(privileged, privileged_after)
        robot = env.physics.robots[0]
        robot.data.qvel[0] = .4
        mujoco.mj_forward(robot.model, robot.data)
        env.history.reset()
        changed, _ = env.observe()
        assert not torch.equal(after[0], changed[0])
        torch.testing.assert_close(after[1], changed[1], atol=0, rtol=0)
    finally:
        env.close()


def test_planar_initializer_preserves_actor_critic_and_original_normalization():
    torch.set_num_threads(1)
    old_contract, new_contract = observation_contract(3), observation_contract(3, 'planar')
    old = ActorCritic(old_contract['size'], old_contract['size']+17, (32, 16))
    old.actor.normalizer.mean.normal_()
    old.actor.normalizer.variance.uniform_(.5, 2)
    saved = {'actor_size': old_contract['size'], 'critic_size': old_contract['size']+17,
             'observation': old_contract, 'model': old.state_dict(), 'hidden_sizes': [32, 16]}
    new = ActorCritic(new_contract['size'], new_contract['size']+17, (32, 16))
    receipt = initialize_model(new, saved, new_contract)
    assert receipt['max_output_difference']['actor_full_precision'] <= 2e-5
    for size, call_old, call_new in [(old_contract['size'], old.actor, new.actor),
                                   (old_contract['size']+17, old.value, new.value)]:
        new_size = size+new_contract['size']-old_contract['size']
        mapping = input_mapping(size, new_size, old_contract, new_contract, torch.device('cpu'))
        sample = torch.randn(16, new_size)*3
        torch.testing.assert_close(call_new(sample), call_old(sample[:, mapping]), atol=2e-5, rtol=1e-6)


def test_tracking_costs_huber_window_collision_once_and_partial_reset():
    from k1_motion.motion_costs import TrackingCosts
    costs = TrackingCosts(2, 'cpu', .02, velocity_weight=2, displacement_weight=1,
                          first_collision_penalty=.3)
    zeros = torch.zeros(2, 3)
    costs.reset(torch.arange(2), zeros, zeros)
    velocity = zeros.clone()
    velocity[:, 0] = .3
    for tick in range(1, 26):
        position = velocity * (.02*tick)
        penalty, _ = costs.step(position, velocity, zeros, zeros, torch.ones(2))
        expected = .02 + (.3 if tick == 1 else 0) + (.01 if tick == 25 else 0)
        torch.testing.assert_close(penalty, torch.full((2,), expected), atol=1e-6, rtol=0)
    costs.reset(torch.tensor([0]), position[:1], zeros[:1])
    penalty, _ = costs.step(velocity*.52, velocity, zeros, zeros, torch.ones(2))
    torch.testing.assert_close(penalty, torch.tensor([.32, .03]), atol=1e-6, rtol=0)
    no_cost = TrackingCosts(2, 'cpu', .02)
    no_cost.reset(torch.arange(2), zeros, zeros)
    penalty, _ = no_cost.step(position, velocity, zeros, zeros, torch.ones(2))
    torch.testing.assert_close(penalty, torch.zeros(2), atol=0, rtol=0)


def test_walking_schedule_reaches_empirical_time_distribution_and_resumes(tmp_path):
    from types import SimpleNamespace
    library = SimpleNamespace(device=torch.device('cpu'), rows=[{'id':'w','split':'train'}, {'id':'g','split':'train'}],
                              lengths=torch.tensor([201, 401]), take_weights=torch.ones(2),
                              episode_duration_ema=torch.tensor([20., 80.]))
    manifest = tmp_path/'curriculum.json'
    manifest.write_text(json.dumps({'version':'train-walking-phase-v2','train_ids':['w','g'],
        'locomotion_ids':['w'],'sampling':{'initial_walking_share':.6,
        'hold_until_transitions':32768000,'anneal_until_transitions':98304000}}))
    sampler = TrainingCurriculum(library, manifest)
    exposure = library.weights * library.episode_duration_ema
    torch.testing.assert_close(exposure/exposure.sum(), torch.tensor([.6, .4]))
    sampler.group_steps[:] = 16384000
    sampler.update()
    sampler.group_steps[:] = 32768000
    metrics = sampler.update()
    assert metrics['schedule_transitions'] == 98304000
    assert metrics['target_walking_transition_share'] == pytest.approx(1/3)
    exposure = library.weights * library.episode_duration_ema
    torch.testing.assert_close(exposure/exposure.sum(), torch.tensor([1/3, 2/3]))
    restored = TrainingCurriculum(library, manifest)
    restored.load_state_dict(sampler.state_dict())
    assert restored.transitions == sampler.transitions
    torch.testing.assert_close(restored.library.weights, sampler.library.weights)


def test_planar_ppo_export_replay_and_resume_reject_changed_contracts(tmp_path):
    from k1_motion.contracts import MotionClip
    from k1_motion.control_validation import replay_clip
    robot, library = make_library(tmp_path)
    torch.set_num_threads(1)
    config = TrainConfig(
        stage='student', iterations=2, horizon=4, epochs=1, minibatch=8,
        hidden_sizes=(32, 16), evaluation_interval=0, checkpoint_interval=1, bc_weight=0)
    original = TrackerEnv(library, 2, 'cpu', history=3)
    try:
        train(original, tmp_path/'original', config)
    finally:
        original.close()
    settings = dict(history=3, observation_profile='planar', tracking_huber=True,
                    self_collision_weight=4, first_collision_penalty=.3)
    env = TrackerEnv(library, 2, 'cpu', 'mujoco_cpp', physics_options={'workers':2}, **settings)
    try:
        result = train(env, tmp_path/'new', config, initialize_checkpoint=tmp_path/'original/checkpoint.pt')
        assert result['finite_updates'] and result['checkpoint_reload_max_error'] == 0
        assert result['observation']['version'] == 'k1-causal-planar-v2'
        export_checkpoint(tmp_path/'new/checkpoint.pt', tmp_path/'export/actor.pt')
        policy = Policy(tmp_path/'export/actor.pt', robot.signature)
        replay = replay_clip(robot, policy, MotionClip.load(library/'standing.npz'))
        assert replay['resets_during_trial'] == 0
        resumed = train(env, tmp_path/'resumed', replace(config, iterations=1),
                        resume_checkpoint=tmp_path/'new/checkpoint.pt')
        assert resumed['optimizer_steps'] > result['optimizer_steps']
    finally:
        env.close()
    wrong = TrackerEnv(library, 2, 'cpu', 'mujoco_cpp', physics_options={'workers':2},
                       **{**settings, 'tracking_huber':False})
    try:
        with pytest.raises(ValueError, match='reward settings'):
            train(wrong, tmp_path/'wrong', config, resume_checkpoint=tmp_path/'new/checkpoint.pt')
    finally:
        wrong.close()


def test_signal_stop_writes_terminal_checkpoint_and_report(tmp_path):
    _, library = make_library(tmp_path)
    output = tmp_path/'signal-run'
    script = '''
import sys,torch
from k1_motion.tracking_env import TrackerEnv
from k1_motion.learning import TrainConfig,train
torch.set_num_threads(1)
env=TrackerEnv(sys.argv[1],2,'cpu','mujoco_cpp',physics_options={'workers':2})
try:
 train(env,sys.argv[2],TrainConfig(stage='student',iterations=100000,horizon=8,epochs=1,minibatch=16,
       evaluation_interval=0,checkpoint_interval=1,hidden_sizes=(32,16),bc_weight=0))
finally: env.close()
'''
    with (tmp_path/'signal.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-c', script, str(library), str(output)],
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+30
            while not (output/'checkpoint.pt').exists() and process.poll() is None and time.monotonic()<deadline:
                time.sleep(.05)
            assert (output/'checkpoint.pt').exists(), (tmp_path/'signal.log').read_text()
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=30) == 0, (tmp_path/'signal.log').read_text()
            report = json.loads((output/'report.json').read_text())
            assert report['stop_reason'] == 'signal_SIGTERM'
            assert report['checkpoint_reload_max_error'] == 0
            assert list(output.glob('checkpoint-*.pt'))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_cache_proof_excludes_only_runtime_policy_and_robot_feedback():
    import k1_motion.reference_cache as cache
    package = Path(cache.__file__).parent
    learning = (package/'learning.py').read_text()
    contracts = (package/'contracts.py').read_text()
    assert cache._preprocessing_ast(learning.replace('class Policy:', 'class Policy:\n    runtime = 1')) == cache._preprocessing_ast(learning)
    assert cache._preprocessing_ast(contracts.replace('class RobotState:', 'class RobotState:\n    runtime = 1')) == cache._preprocessing_ast(contracts)
    for name in ('Reference', 'MotionClip'):
        changed = contracts.replace(f'class {name}:', f'class {name}:\n    preprocessing = 1')
        assert cache._preprocessing_ast(changed) != cache._preprocessing_ast(contracts)
    changed = learning.replace('class MotionLibrary:', 'class MotionLibrary:\n    preprocessing = 1')
    assert cache._preprocessing_ast(changed) != cache._preprocessing_ast(learning)


def test_campaign_commands_isolate_treatments_and_match_exposure():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_planar_campaign import learner_command, NAMES
    root, output, cache = Path('/bundle'), Path('/runs'), Path('/cache.pt')
    for i, name in enumerate(NAMES):
        cmd = learner_command(root, output/name, cache, name, 2048, False)
        def value(flag):
            return cmd[cmd.index(flag)+1]
        assert value('--num-envs') == '2048'
        assert value('--max-seconds') == '36000'
        assert '--curriculum-manifest' in cmd
        assert (value('--observation-profile') == 'planar') == (i in (0, 3))
        assert (value('--self-collision-weight') == '4') == (i in (1, 3))
        assert ('--tracking-huber' in cmd) == (i in (2, 3))
        assert value('--backend') == ('warp' if i == 3 else 'mujoco_cpp')
        assert int(value('--milestone-interval'))*2048*32 == 32768000
    cmd = learner_command(root, output/'combined', cache, NAMES[3], 1024, True)
    assert '--max-seconds' not in cmd
    assert int(cmd[cmd.index('--milestone-interval')+1])*1024*32 == 32768000
