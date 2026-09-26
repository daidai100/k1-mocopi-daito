"""Queue failure cases: wrong backend, short runs called complete, size confounds,
preflight weights reused, and promoting a regressed policy to finish a budget.
"""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))


def option(command, key):
    return command[command.index(key)+1]


def plan(tmp_path):
    from run_warp_capacity_queue import build_commands
    return build_commands(tmp_path, tmp_path/'scripts', tmp_path/'library', tmp_path/'cache.pt',
                          tmp_path/'initial.pt', updates=8000, num_envs=2048, milestone=1000, seed=45)


def test_three_sizes_share_exact_budget_data_backend_and_fresh_initialization(tmp_path):
    runs = plan(tmp_path)
    assert [r['hidden_sizes'] for r in runs] == [[512,256],[2048,1024],[4096,2048]]
    assert [r['actor_parameters'] for r in runs] == [997654,5563414,15321110]
    assert [r['total_parameters'] for r in runs] == [2061613,11392045,31172653]
    for run in runs:
        command = run['command']
        for key,value in {'--backend':'warp','--device':'cuda:0','--iterations':'8000',
                          '--num-envs':'2048','--horizon':'32','--minibatch':'4096',
                          '--reward-profile':'causal-balanced-v1','--preview-horizon-s':'.3',
                          '--epa-horizon':'96','--seed':'45'}.items():
            assert option(command, key) == value
        assert option(command,'--initialize') == str(tmp_path/'initial.pt')
        assert '--resume' not in command and '--max-seconds' not in command
        assert option(run['preflight_command'],'--initialize') == option(command,'--initialize')
        assert option(run['preflight_command'],'--output') != option(command,'--output')


def report(run):
    return dict(backend='warp',device='cuda:0',finite_updates=True,checkpoint_reload_max_error=0,
        config=dict(hidden_sizes=run['hidden_sizes']),num_envs=2048,
        transitions=524288000,optimizer_steps=512000,last_metrics=dict(iteration=8000,
            reference_exposure=dict(seen_originals=2751,total_originals=2751)),
        observation=dict(preview_horizon_s=.3),
        physics_contract=dict(implementation='mujoco-warp-k1-pv-v2',epa_horizon=96,
                              support=dict(device='cuda')))


def test_completion_requires_real_gpu_exact_updates_and_full_corpus(tmp_path):
    from run_warp_capacity_queue import validate_report
    run = plan(tmp_path)[0]
    good = report(run)
    validate_report(good, run, 8000, 2048, originals=2751, require_coverage=True)
    for mutate in [lambda r:r.update(backend='mujoco_cpp'),
                   lambda r:r.update(checkpoint_reload_max_error=.1),
                   lambda r:r['last_metrics'].update(iteration=7999),
                   lambda r:r.update(transitions=65536),
                   lambda r:r['last_metrics']['reference_exposure'].update(seen_originals=3),
                   lambda r:r['physics_contract']['support'].update(device='cpu')]:
        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(ValueError):
            validate_report(bad, run, 8000, 2048, originals=2751, require_coverage=True)


def test_failed_child_persists_failure_and_does_not_start_the_next_size(tmp_path):
    from run_warp_capacity_queue import run_queue
    runs = plan(tmp_path)
    for run in runs:
        run['preflight_command'] = [sys.executable,'-c','raise SystemExit(23)']
    queue = dict(output=str(tmp_path),runs=runs,source_revision='test',num_envs=2048,
                 preflight_updates=3,updates=8000,originals=2751)
    with pytest.raises(RuntimeError,match='preflight'):
        run_queue(queue, {})
    status = json.loads((tmp_path/'status.json').read_text())
    assert status['phase'] == 'failed'
    assert status['runs'][runs[-1]['name']]['phase'] == 'failed'
    assert status['runs'][runs[0]['name']]['phase'] == 'queued'
    assert not list(tmp_path.glob('*/training/checkpoint.pt'))


def test_translation_report_rejects_wrong_objective_discount_or_scene(tmp_path):
    from run_warp_capacity_queue import validate_report
    run = plan(tmp_path)[0]
    expected = dict(reward_settings={'profile': 'survival-position-v2', 'weights': {'root_velocity': 6}},
                    gamma=.9996667222160499, gae_lambda=.99,
                    scene_transitions={'version': 'shuffled-scenes-v1', 'episode_seconds': 120.})
    run['expected_training_contract'] = expected
    good = report(run)
    good.update(reward_settings=expected['reward_settings'], scene_transitions=expected['scene_transitions'])
    good['config'].update(gamma=expected['gamma'], gae_lambda=expected['gae_lambda'])
    validate_report(good, run, 8000, 2048, originals=2751, require_coverage=False)
    for mutate in [lambda r:r.update(reward_settings={}),
                   lambda r:r.update(scene_transitions=None),
                   lambda r:r['config'].update(gamma=.99)]:
        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(ValueError, match='training contract'):
            validate_report(bad, run, 8000, 2048, originals=2751, require_coverage=False)


def test_native_queue_requires_native_backend_and_substep_safety(tmp_path):
    from run_warp_capacity_queue import validate_report
    run = plan(tmp_path)[0]
    run['backend'] = 'mujoco_cpp'
    good = report(run)
    good.update(backend='mujoco_cpp', physics_contract=dict(backend='native-mujoco-cpp-openmp',
        substeps=10, physics_dt=.002, control_dt=.02, actuator={'safety_dt':.002}))
    validate_report(good, run, 8000, 2048, originals=2751, require_coverage=False)
    good['physics_contract']['substeps'] = 1
    with pytest.raises(ValueError):
        validate_report(good, run, 8000, 2048, originals=2751, require_coverage=False)


def test_post_experiment_failure_is_not_reported_as_complete(tmp_path):
    from run_warp_capacity_queue import run_post_experiment
    job = {'name': 'retarget', 'command': [sys.executable, '-c', 'raise SystemExit(23)'],
           'log': str(tmp_path/'retarget.log'), 'receipt': str(tmp_path/'summary.json')}
    with pytest.raises(RuntimeError, match='retarget'):
        run_post_experiment(job, {}, tmp_path)
    job['command'] = [sys.executable, '-c', 'pass']
    with pytest.raises(FileNotFoundError):
        run_post_experiment(job, {}, tmp_path)
    (tmp_path/'summary.json').write_text(json.dumps({'phase':'complete','processed_originals':24}))
    result = run_post_experiment(job, {}, tmp_path)
    assert result['processed_originals'] == 24
