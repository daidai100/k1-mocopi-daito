"""Failure contracts: duration is not quality, windows are not independent takes,
held-out relatives cannot enter training, and survival cannot hide regressions.
"""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def example_rows():
    rows = []
    for family, name in [('walk', 'walk_loop'), ('gesture', 'wave')]:
        for i, seconds in enumerate((3., 12., 25.)):
            rows.append(dict(id=f'{family}{i}', family=family, frames=int(seconds*50)+1,
                split='train', is_mirror=False, dataset='bones_seed', training_eligible=True,
                capture_group=f'bones_seed/{name}_{i}_001', take_name=f'{name}_{i}_001',
                kinematics_accepted=True, rejected_ticks=0, reference_path=f'{family}{i}.npz',
                recovery_audit=dict(accepted=True, geometry_audited=True, audit_hz=500,
                    rejection_reasons=[], stance_slip_p95_m_s=.05, max_ground_penetration_m=.001)))
    return rows


def test_sustained_contract_retains_short_families_and_stratifies_long_quality():
    from k1_motion.sustained_training import build_curriculum
    rows = example_rows()
    locomotion = [r['id'] for r in rows if r['family']=='walk']
    before = copy.deepcopy(rows)
    contract = build_curriculum(rows, locomotion)
    weights = contract['target_transition_weights']
    assert set(weights) == {r['id'] for r in rows} and all(w > 0 for w in weights.values())
    assert sum(weights.values()) == pytest.approx(1.)
    assert sum(weights[k] for k in locomotion) == pytest.approx(.5)
    assert sum(weights[r['id']] for r in rows if r['frames']>=501) == pytest.approx(.5)
    assert rows == before
    assert contract == build_curriculum(list(reversed(rows)), locomotion)
    for bad in ([*rows, rows[0]], [{**rows[0], 'is_mirror':True}, *rows[1:]],
                [{**rows[0], 'split':'test'}, *rows[1:]]):
        with pytest.raises(ValueError):
            build_curriculum(bad, locomotion)
    # Long references with failed admission must not gain long-duration weight.
    bad = copy.deepcopy(rows)
    bad[2]['rejected_ticks'] = 1
    with pytest.raises(ValueError):
        build_curriculum(bad, locomotion)


def test_sustained_reset_has_ten_seconds_remaining_and_durable_failure_state(tmp_path):
    from k1_motion.sustained_training import build_curriculum
    from k1_motion.training_curriculum import TrainingCurriculum
    rows = example_rows()
    path = tmp_path/'curriculum.json'
    path.write_text(json.dumps(build_curriculum(rows, [r['id'] for r in rows[:3]])))
    library = SimpleNamespace(rows=rows, device='cpu', lengths=torch.tensor([r['frames'] for r in rows]),
        take_weights=torch.ones(len(rows)), episode_duration_ema=torch.full((len(rows),), 100.))
    sampler = TrainingCurriculum(library, path)
    assert sampler.fidelity_phases
    torch.manual_seed(24)
    clips = torch.arange(24000) % len(rows)
    # Concentrate failures at the tail: the minimum remaining horizon must still hold.
    sampler.failure_counts[:, -1] = 1e6
    frames = sampler.sample_frames(clips)
    available = library.lengths[clips]-1
    assert torch.all(available-frames >= available.clamp(max=500))
    mix = sampler.reset_counts/sampler.reset_counts.sum()
    torch.testing.assert_close(mix, torch.tensor([.8,.1,.1]), rtol=0, atol=.012)
    assert int(frames[available<500].max()) == 0
    restored = TrainingCurriculum(library, path)
    restored.load_state_dict(sampler.state_dict())
    torch.testing.assert_close(restored.failure_counts, sampler.failure_counts)
    broken = json.loads(path.read_text())
    broken['minimum_remaining_s'] = -1
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError):
        TrainingCurriculum(library, path)


def test_long_source_selection_excludes_heldout_related_takes_and_existing_sources():
    from prepare_long_references import select_sources
    def row(key, group, seconds=15., dataset='kit_motion_language'):
        return dict(source_motion_id=key, capture_group=group, duration_seconds=seconds,
                    dataset=dataset, annotations=['walking forwards'], numerical_validation='passed')
    inventory = [row('new','kit/new'), row('short','kit/short',8),
                 row('old','kit/old'), row('leak','kit/test'), row('long','lafan1/walk1',180,'lafan1')]
    base = [dict(source_motion_id='old',capture_group='kit/old',split='train'),
            dict(source_motion_id='heldout',capture_group='kit/test',split='test')]
    splits = {r['capture_group']:'train' for r in inventory}
    selected, report = select_sources(inventory, base, splits, per_family_dataset=4)
    assert {r['source_motion_id'] for r in selected} == {'new','long'}
    assert all(r['split']=='train' for r in selected)
    assert report['excluded']['heldout_related_take'] == 1
    assert report['excluded']['shorter_than_10s'] == 1
    assert next(r for r in selected if r['source_motion_id']=='long')['duration_seconds'] == 180


def test_milestone_gate_blocks_lost_joint_pass_and_target_mismatch():
    from k1_motion.sustained_training import assess_milestone
    def result(key, clean=True, collision=False, score=.8):
        return dict(id=key, execution_passed=True, completed=True, clean_success=clean,
            absolute_motion_v1={'clean':clean}, self_collision_ticks=int(collision), fell=False,
            full_reference_duration_world_score=score, resets_during_trial=0,
            actuator_safety={'joint_limit_fraction':0., 'operating_speed_fraction':0.})
    baseline=[result('a'),result('b',False)]
    candidate=[result('a',False),result('b',True)]
    contract={'panel_sha256':'panel', 'source_revision':'source', 'reference_scale':None}
    gate=assess_milestone(candidate, baseline, contract, contract)
    assert not gate['continue_training'] and gate['lost_joint_ids']==['a']
    assert assess_milestone(baseline,baseline,contract,contract)['continue_training']
    with pytest.raises(ValueError,match='target'):
        assess_milestone(baseline,baseline,{**contract,'reference_scale':{'scale':.7}},contract)
    with pytest.raises(ValueError):
        assess_milestone([baseline[0],baseline[0]],baseline,contract,contract)


def test_real_sustained_ppo_export_replay_resume(tmp_path):
    from dataclasses import replace
    from test_training import make_library
    from k1_motion.contracts import MotionClip
    from k1_motion.export import export_checkpoint
    from k1_motion.learning import Policy, TrainConfig, train
    from k1_motion.control_validation import replay_clip
    from k1_motion.tracking_env import TrackerEnv
    from k1_motion.sustained_training import build_curriculum
    robot, directory = make_library(tmp_path)
    original = json.loads((directory/'index.jsonl').read_text())
    rows=[]
    for key,family in [('w','walk'),('g','gesture')]:
        row={**original,'id':key,'family':family,'is_mirror':False,'training_eligible':True,
            'frames':601,'capture_group':'synthetic/'+key,'reference_path':key+'.npz',
            'rejected_ticks':0,'dataset':'synthetic'}
        row['recovery_audit']=dict(accepted=True,geometry_audited=True,audit_hz=500,
            rejection_reasons=[],stance_slip_p95_m_s=0.,max_ground_penetration_m=0.)
        MotionClip.from_references([robot.neutral_reference(i*.02) for i in range(601)],row).save(directory/row['reference_path'])
        rows.append(row)
    (directory/'index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest=tmp_path/'sustained.json'
    manifest.write_text(json.dumps(build_curriculum(rows,['w'])))
    torch.set_num_threads(1)
    env=TrackerEnv(directory,2,'cpu','mujoco_cpp',history=3,reference_storage='packed',
        reward_profile='world-body-v1',safety_profile='casual-safe-v1',physics_options={'workers':2})
    cfg=TrainConfig(stage='student',iterations=2,horizon=8,epochs=1,minibatch=16,
        hidden_sizes=(32,16),evaluation_interval=0,checkpoint_interval=1,bc_weight=0,
        curriculum_manifest=str(manifest))
    try:
        report=train(env,tmp_path/'training',cfg)
        assert report['finite_updates'] and report['checkpoint_reload_max_error']==0
        assert 'fidelity_events' in report['last_metrics']['curriculum']
        telemetry=report['last_metrics']['curriculum']['sustained_tracking']
        first=json.loads((tmp_path/'training/metrics.jsonl').read_text().splitlines()[0])
        assert first['curriculum']['sustained_tracking']['reset_available_mean_s'] >= 10.
        assert sum(telemetry['duration_transition_shares'].values()) == pytest.approx(1.)
        assert telemetry['ended_episodes_10s'] == 0
        export_checkpoint(tmp_path/'training/checkpoint.pt',tmp_path/'export/actor.pt')
        result=replay_clip(robot,Policy(tmp_path/'export/actor.pt',robot.signature),MotionClip.load(directory/'g.npz'))
        assert result['resets_during_trial']==0
        resumed=train(env,tmp_path/'resumed',replace(cfg,iterations=1),resume_checkpoint=tmp_path/'training/checkpoint.pt')
        assert resumed['optimizer_steps']>report['optimizer_steps']
    finally:
        env.close()
