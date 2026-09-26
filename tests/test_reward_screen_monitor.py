"""Screen monitor fails closed on incomplete evidence and preserves checkpoints.

Risks: overlapping host names, unsafe remote paths, copying mutable live weights,
partial/mismatched replay treated as success, semantic macro means weighted by
family abundance, or terminal queues leaving an endless monitor.
"""
import importlib.util
import json
from pathlib import Path

import pytest


def monitor():
    path = Path(__file__).resolve().parents[1]/'scripts/watch_reward_screen.py'
    spec = importlib.util.spec_from_file_location('reward_screen_monitor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_rows():
    panel = [{'id': str(i), 'capture_group': f'g{i}', 'family': 'walk',
              'semantic_group': 'walk' if i < 2 else 'gesture'} for i in range(3)]
    trials = []
    for i, row in enumerate(panel):
        trials.append({**row, 'completed': True, 'clean_success': i < 2, 'self_collision_ticks': int(i == 2),
                       'fell': False, 'resets_during_trial': 0,
                       'trajectory_v3': {'version': 'k1-trajectory-fidelity-v3-screening', 'screening_only': True,
                           'clean_v3': i < 2, 'full_reference_duration_xy_score': [1., 1., .1][i],
                           'root_xy_rmse_m': [.0, .0, 2.][i], 'survived_fraction': 1.}})
    summary = {'contract': {'panel_sha256': 'panel', 'source_revision': 'source'},
               'execution_errors': 0, 'resets_during_trials': 0,
               'all': {'trials': 3, 'completed': 3, 'clean': 2, 'collision_trials': 1, 'fell': 0}}
    return panel, trials, summary


def test_arbitrary_panel_semantic_macro_recount_and_missing_or_wrong_evidence():
    module = monitor()
    panel, trials, summary = fixture_rows()
    result = module.aggregate_replay(panel, trials, summary, 'panel', 'source')
    assert result['all']['trials'] == 3
    assert result['all']['clean_v3'] == 2
    assert result['trajectory_v3_macro']['full_reference_duration_xy_score'] == pytest.approx(.55)
    assert result['by_semantic_group']['walk']['clean_v3'] == 2
    assert result['automatically_promoted'] is False
    with pytest.raises(ValueError, match='trial'):
        module.aggregate_replay(panel, trials[:-1], summary, 'panel', 'source')
    with pytest.raises(ValueError, match='contract'):
        module.aggregate_replay(panel, trials, summary, 'changed', 'source')
    trials[0]['trajectory_v3']['full_reference_duration_xy_score'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        module.aggregate_replay(panel, trials, summary, 'panel', 'source')


def test_training_root_guard_rejects_escape_and_renamed_run():
    module = monitor()
    assert str(module.checked_training_directory('/remote/campaign', 'simple_main', '/remote/campaign/simple_main/training')) == '/remote/campaign/simple_main/training'
    for name, path in [('simple_main', '/other/simple_main/training'), ('../escape', '/remote/escape/training'),
                       ('simple_main', '/remote/campaign/simple_main/../elsewhere/training')]:
        with pytest.raises(ValueError):
            module.checked_training_directory('/remote/campaign', name, path)


def test_terminal_capture_is_immutable_and_live_current_is_not_a_candidate(tmp_path):
    module = monitor()
    training = tmp_path/'training'
    training.mkdir()
    (training/'checkpoint.pt').write_bytes(b'live')
    (training/'checkpoint-000125.pt').write_bytes(b'milestone')
    assert [p.name for p in module.checkpoint_candidates(training, False)] == ['checkpoint-000125.pt']
    target = training/'checkpoint-terminal.pt'
    module.capture_terminal_checkpoint(training/'checkpoint.pt', target)
    (training/'checkpoint.pt').write_bytes(b'later')
    module.capture_terminal_checkpoint(training/'checkpoint.pt', target)
    assert target.read_bytes() == b'live'
    assert [p.name for p in module.checkpoint_candidates(training, True)] == ['checkpoint-000125.pt', 'checkpoint-terminal.pt']


def test_one_cycle_namespaces_hosts_and_finishes_only_terminal_campaigns(tmp_path, monkeypatch):
    module = monitor()
    panel, trials, summary = fixture_rows()
    panel_path = tmp_path/'panel.json'
    panel_path.write_text(json.dumps(panel))
    bundle = tmp_path/'bundle'
    bundle.mkdir()
    (bundle/'bundle.json').write_text(json.dumps({'source_revision': 'source'}))
    campaign = {'phase': 'completed', 'runs': {}}
    for host in ['server', 'desktop']:
        training = tmp_path/host/'simple_main/training'
        training.mkdir(parents=True)
        (training/'checkpoint-000125.pt').write_bytes(b'weights')
        campaign['runs'][f'{host}/simple_main'] = {'phase': 'completed', 'training_directory': str(training)}
    monkeypatch.setattr(module, 'refresh_campaigns', lambda *args, **kwargs: campaign)
    args = type('Args', (), {'root': tmp_path, 'bundle': bundle, 'panel': panel_path,
                            'server_root': '/remote/campaign', 'desktop_root': tmp_path/'desktop'})()
    calls = []

    def evaluate(checkpoint, key, args, panel, panel_hash, revision):
        calls.append(key)
        return {'all': {'trials': 3}, 'automatically_promoted': False}

    state, done = module.cycle(args, {}, evaluator=evaluate)
    assert done and len(calls) == 2
    assert set(state['comparisons']) == {'server/simple_main/checkpoint-000125', 'desktop/simple_main/checkpoint-000125'}
    state, done = module.cycle(args, state, evaluator=evaluate)
    assert done and len(calls) == 2
    campaign['phase'] = 'running'
    _, done = module.cycle(args, state, evaluator=evaluate)
    assert not done


def test_failed_supervisor_does_not_hide_a_still_running_child(tmp_path, monkeypatch):
    module = monitor()
    panel, _, _ = fixture_rows()
    panel_path = tmp_path/'panel.json'
    panel_path.write_text(json.dumps(panel))
    bundle = tmp_path/'bundle'
    bundle.mkdir()
    (bundle/'bundle.json').write_text(json.dumps({'source_revision': 'source'}))
    campaign = {'phase': 'completed_with_failures', 'runs': {
        'server/simple_main': {'phase': 'learner_updates', 'training_directory': str(tmp_path/'training')}}}
    monkeypatch.setattr(module, 'refresh_campaigns', lambda *args, **kwargs: campaign)
    args = type('Args', (), {'root': tmp_path, 'bundle': bundle, 'panel': panel_path,
                            'server_root': '/remote/campaign', 'desktop_root': tmp_path/'desktop'})()
    state, done = module.cycle(args, {}, evaluator=lambda *args: pytest.fail('No immutable candidate exists'))
    assert not done
    assert state['phase'] == 'awaiting_terminal_run_evidence'


def world_fixture_rows():
    panel, trials, summary = fixture_rows()
    for i, row in enumerate(trials):
        row.update(world_body_rmse_m=[.05,.2,.4][i],world_body_p95_m=[.1,.3,.6][i],
                   full_reference_duration_world_score=[.95,.7,.1][i],
                   actuator_safety=dict(operating_speed_fraction=.1 if i==1 else 0.,
                       operating_speed_max_ratio=1.1 if i==1 else .8,
                       nominal_speed_fraction=0.,nominal_speed_max_ratio=.9,
                       joint_limit_fraction=.01 if i==2 else 0.,
                       joint_limit_max_error=.001 if i==2 else 0.,torque_saturation=.2,
                       operating_speed_excess_squared=.001 if i==1 else 0.,
                       sample_period_s=.002,sample_count=500,contract={'version':'fixture'}),
                   absolute_motion_v1=dict(version='world-position-safety-v1',clean=i==0,
                       world_rmse_limit_m=.15,world_p95_limit_m=.30,full_duration_completed=True))
    return panel,trials,summary


def test_world_safety_aggregation_is_additive_with_semantic_macro_and_legacy_compatibility():
    m=monitor()
    panel,trials,summary=world_fixture_rows()
    result=m.aggregate_replay(panel,trials,summary,'panel','source')
    assert result['absolute_motion_available'] is True
    assert result['all']['absolute_clean']==1
    assert result['all']['clean']==2  # Historical acceptance remains independent.
    assert result['all']['operating_overspeed_trials']==1
    assert result['all']['nominal_overspeed_trials']==0
    assert result['all']['joint_limit_violation_trials']==1
    assert result['all']['world_body_rmse_m']==pytest.approx(.65/3)
    assert result['all']['full_reference_duration_world_score']==pytest.approx(1.75/3)
    assert result['world_safety_macro']['absolute_clean_rate']==pytest.approx(.25)
    assert result['world_safety_macro']['full_reference_duration_world_score']==pytest.approx(.4625)
    oldpanel,oldtrials,oldsummary=fixture_rows()
    old=m.aggregate_replay(oldpanel,oldtrials,oldsummary,'panel','source')
    assert old['absolute_motion_available'] is False
    assert 'absolute_clean' not in old['all'] and 'world_safety_macro' not in old


@pytest.mark.parametrize('fault',['missing','nonfinite','false_clean'])
def test_world_safety_rejects_partial_nonfinite_or_inconsistent_new_evidence(fault):
    m=monitor()
    panel,trials,summary=world_fixture_rows()
    if fault=='missing':
        del trials[1]['absolute_motion_v1']
    elif fault=='nonfinite':
        trials[1]['world_body_rmse_m']=float('nan')
    else:
        trials[1]['absolute_motion_v1']['clean']=True
    with pytest.raises(ValueError,match='[Ww]orld|[Aa]bsolute'):
        m.aggregate_replay(panel,trials,summary,'panel','source')
