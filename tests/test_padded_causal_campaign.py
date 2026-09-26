"""Campaign must include the whole selected pool and distinguish training from promotion.

Failures: a three-clip manifest can masquerade as the complete long pool;
optimizer updates can stop without evidence; old or mismatched evaluator
contracts can pass a review; a mildly regressed exploratory actor can replace
the protected champion; gross behavioral degradation can consume eight hours.
"""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def trial(key, *, clean=True, collision=False, score=.8):
    return dict(id=key,execution_passed=True,completed=True,clean_success=clean,
        absolute_motion_v1={'clean':clean},self_collision_ticks=int(collision),fell=False,
        full_reference_duration_world_score=score,resets_during_trial=0,
        actuator_safety={'joint_limit_fraction':0.,'operating_speed_fraction':0.})


def test_bounded_exploration_never_promotes_regressions_and_rejects_gross_loss():
    from run_padded_causal_campaign import review_candidate
    rows=[trial(str(i)) for i in range(30)]
    contract=dict(panel_sha256='panel',source_revision='source',reference_scale=None)
    baseline=dict(trials={'training':rows,'development':rows},
        contracts={'training':contract,'development':contract},execution_errors=0)
    candidate=copy.deepcopy(baseline)
    candidate['trials']['development'][0]=trial('0',clean=False,score=.79)
    decision=review_candidate(candidate,baseline,baseline)
    assert decision['continue_training'] and not decision['replace_champion']
    for row in candidate['trials']['development']:
        row['completed']=False
        row['clean_success']=False
        row['absolute_motion_v1']['clean']=False
        row['fell']=True
    assert not review_candidate(candidate,baseline,baseline)['continue_training']
    candidate=copy.deepcopy(baseline)
    candidate['contracts']['development']['source_revision']='other'
    with pytest.raises(ValueError,match='contract'):
        review_candidate(candidate,baseline,baseline)


def test_training_membership_cannot_be_a_small_subset():
    from run_padded_causal_campaign import verify_membership
    rows=[dict(id=str(i),split='train',is_mirror=False,training_eligible=True,
        standing_padding={'original_duration_s':12.}) for i in range(8)]
    all_rows=[*rows,dict(id='short',standing_padding={'original_duration_s':9.8})]
    contract={'train_ids':[r['id'] for r in rows],
              'target_transition_weights':{r['id']:1/8 for r in rows}}
    verify_membership(rows,all_rows,contract)
    with pytest.raises(ValueError,match='all'):
        verify_membership(rows[:3],all_rows,contract)
    contract['target_transition_weights']['0']=0
    with pytest.raises(ValueError,match='positive'):
        verify_membership(rows,all_rows,contract)


def test_initial_coverage_visits_every_original_and_resumes_remaining_order(tmp_path):
    import json
    from types import SimpleNamespace
    import torch
    from k1_motion.sustained_training import build_curriculum
    from k1_motion.training_curriculum import TrainingCurriculum
    from test_sustained_training import example_rows
    rows=example_rows()
    manifest=build_curriculum(rows,[r['id'] for r in rows[:3]])
    manifest['initial_coverage_sweep']=True
    path=tmp_path/'curriculum.json'
    path.write_text(json.dumps(manifest))
    def library():
        return SimpleNamespace(rows=rows,device='cpu',lengths=torch.tensor([r['frames'] for r in rows]),
            take_weights=torch.ones(len(rows)),episode_duration_ema=torch.ones(len(rows)))
    sampler=TrainingCurriculum(library(),path)
    first=sampler.sample_clips(2)
    saved=sampler.state_dict()
    restored=TrainingCurriculum(library(),path)
    restored.load_state_dict(saved)
    remainder=sampler.sample_clips(len(rows)-2)
    torch.testing.assert_close(restored.sample_clips(len(rows)-2),remainder)
    assert sorted(torch.cat((first,remainder)).tolist())==list(range(len(rows)))
    assert sampler.update()['initial_coverage_remaining']==0
