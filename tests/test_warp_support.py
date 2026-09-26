"""GPU support must match solved-force/Jacobian measurements, including resets.

Failures covered: airborne support, velocity from a stale cvel, contact ordering
mixing worlds/feet, no tangential slip, stale support after a partial reset,
lost overflow flags, and the actual causal reward refusing GPU training.
"""
import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import torch

from k1_motion.actuators import effective_model
from k1_motion.support import foot_support
from k1_motion.warp_physics import WarpPhysics
from test_actuator_contract import reset_reference, settings
from test_training import make_library

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires NVIDIA CUDA')


def test_warp_support_matches_scalar_oracle_of_same_solved_gpu_state(tmp_path):
    torch.set_num_threads(1)
    robot, _ = make_library(tmp_path)
    robot.substeps = 1
    robot.control_dt = .002
    config = settings(robot)
    physics = WarpPhysics(robot, 4, 'cuda:0', action_settings=config, epa_horizon=96)
    reference = {k:v.cuda() for k,v in reset_reference(robot, 4).items()}
    reference['root_position'][:3, 2] -= 2.0002
    reference['root_velocity'][1, :3] = torch.tensor([.6, -.2, 0.], device='cuda:0')
    reference['root_velocity'][1, 5] = .8
    reference['root_orientation'][2] = torch.tensor([np.cos(.05), np.sin(.05), 0., 0.], device='cuda:0')
    model = effective_model(robot, config)
    data = mujoco.MjData(model)
    one_foot_seen = False
    peak_slip = 0.
    try:
        physics.reset(torch.arange(4, device='cuda:0'), reference)
        for _ in range(3):
            physics.step(reference['joint_position'])
            measured = physics.foot_support()
            assert all(v.device.type == 'cuda' for v in measured.values())
            for world in range(4):
                # Copy solved GPU state only; no native dynamics or forward solve.
                mjw.get_data_into(data, model, physics.data, world)
                # Warp 3.11's copier populates the new geom pair, but MuJoCo
                # 3.10 also needs its separate legacy geom1/geom2 fields.
                data.contact.geom1[:] = data.contact.geom[:, 0]
                data.contact.geom2[:] = data.contact.geom[:, 1]
                expected = foot_support(model, data)
                for key in expected:
                    np.testing.assert_allclose(measured[key][world].cpu(), expected[key], rtol=2e-4, atol=2e-5)
            assert measured['contact'][0].all()
            one_foot_seen |= int(measured['contact'][2].sum()) == 1
            assert not measured['contact'][3].any()
            assert measured['normal_force'][3].sum() == 0
            peak_slip = max(peak_slip, float(measured['slip_speed'][1].max()))
        assert one_foot_seen  # Tilted contact may unload after its initial impulse.
        assert peak_slip > .05  # A moving foot can also unload after the impact.
        physics.reset(torch.tensor([1], device='cuda:0'), {k:v[:1] for k,v in reference.items()})
        physics.step(reference['joint_position'])
        measured = physics.foot_support()
        assert measured['normal_force'][1].sum() > 1
        before = {k:v.clone() for k,v in measured.items()}
        physics.reset(torch.tensor([1], device='cuda:0'), {k:v[3:] for k,v in reference.items()})
        for key, value in physics.foot_support().items():
            assert not value[1].any()
            torch.testing.assert_close(value[[0,2,3]], before[key][[0,2,3]])
        assert physics.contract['support']['version'] == 'ground-foot-contact-patch-v1'
        physics.overflow[0] = 256
        physics.reset(torch.tensor([0], device='cuda:0'), {k:v[3:] for k,v in reference.items()})
        with pytest.raises(RuntimeError, match='overflow'):
            physics.validate()
        physics.overflow.zero_()
        physics.overflow_latch.zero_()
    finally:
        physics.close()


def test_causal_balanced_real_gpu_ppo_reload_and_preview(tmp_path):
    from k1_motion.learning import TrainConfig, train
    from k1_motion.tracking_env import TrackerEnv
    torch.set_num_threads(1)
    robot, library = make_library(tmp_path)
    env = TrackerEnv(library, 4, 'cuda:0', 'warp', reference_storage='packed',
        physics_options={'epa_horizon':96}, reward_profile='causal-balanced-v1',
        observation_profile='preview', safety_profile='casual-safe-v1', action_settings=settings(robot))
    try:
        report = train(env, tmp_path/'training', TrainConfig(stage='student', iterations=2,
            horizon=4, epochs=1, minibatch=16, evaluation_interval=0, hidden_sizes=(32,16), bc_weight=0))
        assert report['backend'] == 'warp'
        assert report['finite_updates'] and report['checkpoint_reload_max_error'] == 0
        assert report['observation']['preview_horizon_s'] == .3
        assert 'stance_still' in report['last_metrics']['reward_components']
        assert report['physics_contract']['support']['device'] == 'cuda'
    finally:
        env.close()
