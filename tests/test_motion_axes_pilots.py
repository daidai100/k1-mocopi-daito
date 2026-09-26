"""Failure cases: one bad limb erases all pose signal; translation is counted
as pose error; root weight is multiplied by joint count; a good pose masks a
fall; a poor pose masks safe horizontal progress; partial replays look complete.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from k1_motion.world_objective import WorldBodyTracking
from test_reward_series import sample


def test_pointwise_shape_keeps_partial_credit_and_root_cost_is_separate():
    state, ref, _ = sample()
    state['omega'] = torch.zeros(2, 3)
    reward = WorldBodyTracking('world-pointwise-root-v1', settings={
        'weights': {'root_xy_cost': 2.0}})
    state['landmarks'][:, 0, 0] += .12
    _, parts = reward.step(state, ref, ref['landmark_velocity'])
    expected = (16 + torch.exp(torch.tensor(-1.))) / 17
    torch.testing.assert_close(parts['body_relative'], expected.expand(2))
    torch.testing.assert_close(parts['root_xy_cost'], torch.zeros(2))
    state['position'][:, 0] += 1.
    state['landmarks'][:, :, 0] += 1.
    _, shifted = reward.step(state, ref, ref['landmark_velocity'])
    torch.testing.assert_close(shifted['body_relative'], parts['body_relative'])
    torch.testing.assert_close(shifted['root_xy_cost'], torch.full((2,), -1.5))
    torch.testing.assert_close(shifted['weighted/root_xy_cost'], torch.full((2,), -3.))
    assert reward.contract['point_aggregation'] == 'mean_of_per_point_exponentials'
    with pytest.raises(ValueError):
        WorldBodyTracking('world-pointwise-root-v1', settings={'weights': {'root_xy_cost': -1}})


def test_motion_axes_benchmark_separates_horizontal_joint_and_safety(tmp_path):
    panel = [
        {'id': 'walk', 'family': 'walk', 'capture_group': 'group_walk', 'semantic_group': 'ordinary_walk'},
        {'id': 'fall', 'family': 'walk', 'capture_group': 'group_fall', 'semantic_group': 'ordinary_walk'},
    ]
    panel_path = tmp_path / 'panel.json'
    panel_path.write_text(json.dumps(panel))
    replay = tmp_path / 'replay'
    replay.mkdir()
    common = dict(family='walk', resets_during_trial=0,
        root_velocity_rmse_m_s=.4, relative_body_rmse_m=.08,
        mean_root_orientation_error_rad=.1,
        reference_horizontal_displacement_m=1., actual_horizontal_displacement_m=1.,
        actuator_safety={'operating_speed_fraction': 0., 'joint_limit_fraction': 0.},
        trajectory_v3={'version': 'k1-trajectory-fidelity-v3-screening',
                       'survived_fraction': 1., 'root_xy_rmse_m': .2,
                       'root_xy_p95_m': .3, 'full_reference_duration_xy_score': .8},
        absolute_motion_v1={'clean': False})
    (replay/'walk.json').write_text(json.dumps({**common, 'id': 'walk', 'capture_group': 'group_walk',
        'completed': True, 'fell': False, 'self_collision_ticks': 0,
        'clean_success': False, 'root_progress_ratio': 1., 'joint_rmse_rad': .7}))
    (replay/'fall.json').write_text(json.dumps({**common, 'id': 'fall', 'capture_group': 'group_fall',
        'completed': False, 'fell': True, 'self_collision_ticks': 0,
        'clean_success': False, 'root_progress_ratio': .2, 'joint_rmse_rad': .01,
        'trajectory_v3': {**common['trajectory_v3'], 'survived_fraction': .4,
                          'full_reference_duration_xy_score': .2}}))
    (replay/'summary.json').write_text(json.dumps({
        'contract': {'panel_sha256': hashlib.sha256(panel_path.read_bytes()).hexdigest(),
                     'source_revision': 'source'},
        'execution_errors': 0, 'resets_during_trials': 0,
        'all': {'trials': 2, 'completed': 1, 'clean': 0, 'collision_trials': 0, 'fell': 1}}))
    output = tmp_path/'axes.json'
    command = [sys.executable, str(Path(__file__).resolve().parents[1]/'scripts/summarize_motion_axes.py'),
               '--panel', str(panel_path), '--replay', str(replay), '--output', str(output)]
    subprocess.run(command, check=True)
    result = json.loads(output.read_text())
    walk = result['by_semantic_group']['ordinary_walk']
    assert result['version'] == 'k1-motion-axes-v1'
    assert walk['safe_progress'] == 1
    assert walk['completed'] == 1 and walk['fell'] == 1
    assert walk['horizontal']['full_duration_xy_score_mean'] == pytest.approx(.5)
    assert walk['joints']['joint_rmse_rad_completed_mean'] == pytest.approx(.7)
    assert walk['horizontal']['root_xy_rmse_m_completed_mean'] == pytest.approx(.2)


def test_three_pilot_plans_hold_everything_except_root_weight(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_motion_axes_pilots import build_runs
    bundle, inputs, output = (tmp_path/name for name in ('bundle', 'inputs', 'output'))
    runs = build_runs(bundle, inputs, output, tmp_path/'panel.json', Path(sys.executable))
    assert [run['root_xy_cost_weight'] for run in runs] == [.25, 1., 2.]
    assert len({run['seed'] for run in runs}) == 1
    assert len({run['gpu_index'] for run in runs}) == 1
    assert len({run['source_library'] for run in runs}) == 1
    assert len({run['curriculum'] for run in runs}) == 1
    for run in runs:
        assert run['updates'] == 125
        command = run['training_command']
        assert command[command.index('--reward-profile')+1] == 'world-pointwise-root-v1'
        assert command[command.index('--iterations')+1] == '125'
        assert command[command.index('--world-reward-settings')+1].endswith(
            'root_'+str(run['root_xy_cost_weight']).replace('.', '_')+'.json')
