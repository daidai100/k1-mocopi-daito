"""Versioned duration/quality sampling and fail-closed milestone decisions.

Reference quality is kinematic evidence; it never asserts dynamic solvability.
"""
from collections import Counter, defaultdict
import math

from .reference_admission import reference_rejections, take_family
from .training_curriculum import FAILURE_OBSERVATION

DURATION_SHARES = {'under_10s': .5, '10_to_20s': .35, '20s_plus': .15}
QUALITY_SHARES = {'strict_geometry': .9, 'bounded_ground': .1}


def duration_bucket(row):
    seconds = (row['frames']-1)*.02
    return 'under_10s' if seconds < 10 else '10_to_20s' if seconds < 20 else '20s_plus'


def quality_tier(row):
    audit = row.get('recovery_audit', {})
    if (audit.get('accepted') and audit.get('geometry_audited')
            and audit.get('audit_hz', 0) >= 500 and not audit.get('rejection_reasons')):
        return 'strict_geometry'
    return 'bounded_ground'


def _balanced_weights(rows):
    """60% related-take balance, 20% family floor, 20% source diversity floor."""
    takes = Counter(take_family(r['capture_group']) for r in rows)
    family_takes = defaultdict(set)
    source_takes = defaultdict(set)
    family_counts, source_counts = Counter(), Counter()
    for r in rows:
        take = take_family(r['capture_group'])
        family_takes[r['family']].add(take)
        source_takes[r['dataset']].add(take)
        family_counts[r['family'], take] += 1
        source_counts[r['dataset'], take] += 1
    result = {}
    for r in rows:
        take, family, source = take_family(r['capture_group']), r['family'], r['dataset']
        result[r['id']] = (.6/len(takes)/takes[take]
            + .2/len(family_takes)/len(family_takes[family])/family_counts[family,take]
            + .2/len(source_takes)/len(source_takes[source])/source_counts[source,take])
    return result


def build_curriculum(rows, locomotion_ids, *, sustained=True):
    rows = sorted(rows,key=lambda r:r['id'])
    ids = {r['id'] for r in rows}
    selected = set(locomotion_ids)
    if not rows or len(ids)!=len(rows) or not selected or not selected<ids:
        raise ValueError('Unique references and both locomotion/remainder groups are required')
    for row in rows:
        if (row.get('split')!='train' or row.get('is_mirror') is not False
                or row.get('training_eligible') is not True or reference_rejections(row)):
            raise ValueError('Ineligible training reference: '+row['id'])
        if quality_tier(row)=='bounded_ground' and row.get('kinematics_accepted'):
            raise ValueError('Strict reference lacks independent 500 Hz evidence: '+row['id'])
    weights, strata = {}, {}
    for group in ('clear_locomotion','broad_remainder'):
        group_rows=[r for r in rows if (r['id'] in selected)==(group=='clear_locomotion')]
        available={duration_bucket(r) for r in group_rows}
        duration_total=sum(DURATION_SHARES[b] for b in available)
        for bucket in sorted(available):
            bucket_rows=[r for r in group_rows if duration_bucket(r)==bucket]
            tiers={quality_tier(r) for r in bucket_rows}
            quality_total=sum(QUALITY_SHARES[t] for t in tiers)
            for tier in sorted(tiers):
                subset=[r for r in bucket_rows if quality_tier(r)==tier]
                share=.5*DURATION_SHARES[bucket]/duration_total*QUALITY_SHARES[tier]/quality_total
                weights.update({key:share*w for key,w in _balanced_weights(subset).items()})
                strata[group+'/'+bucket+'/'+tier]=dict(originals=len(subset),
                    related_take_families=len({take_family(r['capture_group']) for r in subset}),
                    reference_hours=sum((r['frames']-1)*.02 for r in subset)/3600,
                    target_transition_share=share)
    family_shares, duration_shares, quality_shares, source_shares = (defaultdict(float) for _ in range(4))
    for row in rows:
        w=weights[row['id']]
        family_shares[row['family']]+=w
        duration_shares[duration_bucket(row)]+=w
        quality_shares[quality_tier(row)]+=w
        source_shares[row['dataset']]+=w
    return dict(version='sustained-quality-curriculum-v1' if sustained else 'sustained-quality-control-v1',
        train_ids=sorted(ids),locomotion_ids=sorted(selected),
        target_transition_weights=dict(sorted(weights.items())),
        target_family_transition_shares=dict(sorted(family_shares.items())),
        target_duration_transition_shares=dict(sorted(duration_shares.items())),
        target_quality_transition_shares=dict(sorted(quality_shares.items())),
        target_source_transition_shares=dict(sorted(source_shares.items())),strata=strata,
        sampling=dict(locomotion_transition_share=.5,annealing=False,
            duration_shares=DURATION_SHARES,quality_shares=QUALITY_SHARES,
            missing_strata='renormalize available strata within their parent; retain every admitted original',
            within_stratum='60% equal related take; 20% family then take; 20% source then take; equal clips per take',
            episode_duration_correction='target_transition_weight / episode_duration_ema.clamp(min=1)'),
        reset_mix={'start':.8,'failure_biased':.1,'uniform':.1} if sustained else
                  {'start':.5,'failure_biased':.25,'uniform':.25},
        minimum_remaining_s=10. if sustained else 0.,
        short_motion_reset='start of the complete recording when shorter than minimum remaining duration',
        failure_observation=dict(FAILURE_OBSERVATION),
        failure_ledger='Measured sustained world/speed error and safety events, plus terminal failures',
        reference_scale=None,reference_arrays_changed=False,mirrors=0,
        physics_qualified=False,confirmation_panel_used=False)


def summarize_trials(trials):
    if not trials or len({r['id'] for r in trials})!=len(trials):
        raise ValueError('Unique nonempty trial records required')
    if any(not r.get('execution_passed') or r.get('resets_during_trial')!=0 for r in trials):
        raise ValueError('Milestone requires error-free uninterrupted trials')
    for r in trials:
        score=r['full_reference_duration_world_score']
        if not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('Invalid full-duration world score')
        if (r['clean_success'] or r['absolute_motion_v1']['clean']) and (
                not r['completed'] or r['self_collision_ticks']):
            raise ValueError('Clean trial contradicts completion/collision evidence')
        for field in ('joint_limit_fraction','operating_speed_fraction'):
            value=r['actuator_safety'][field]
            if not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Invalid measured actuator safety')
    return dict(trials=len(trials),raw=sum(bool(r['completed']) for r in trials),
        completed_without_collision=sum(bool(r['completed'] and not r['self_collision_ticks']) for r in trials),
        historical_clean=sum(bool(r['clean_success']) for r in trials),
        world_safety_clean=sum(bool(r['absolute_motion_v1']['clean']) for r in trials),
        jointly_clean=sum(bool(r['clean_success'] and r['absolute_motion_v1']['clean']) for r in trials),
        collisions=sum(r['self_collision_ticks']>0 for r in trials),falls=sum(bool(r['fell']) for r in trials),
        joint_limit_trials=sum(r['actuator_safety']['joint_limit_fraction']>0 for r in trials),
        operating_speed_trials=sum(r['actuator_safety']['operating_speed_fraction']>0 for r in trials),
        full_duration_world_score=sum(r['full_reference_duration_world_score'] for r in trials)/len(trials),
        execution_errors=0)


def assess_milestone(candidate, protected, candidate_contract, protected_contract):
    """Zero lost-pass/safety regression budget, declared before seeing candidates."""
    if {r['id'] for r in candidate}!={r['id'] for r in protected}:
        raise ValueError('Milestone panel identities differ')
    for key in ('panel_sha256','source_revision','reference_scale'):
        if key not in candidate_contract or key not in protected_contract or (
                candidate_contract[key]!=protected_contract[key]):
            raise ValueError('Milestone evaluator/panel/target contract differs: '+key)
    for key in ('action_settings','actuator_contract','observation'):
        if key in candidate_contract or key in protected_contract:
            if candidate_contract.get(key)!=protected_contract.get(key):
                raise ValueError('Milestone controller contract differs: '+key)
    before,after=summarize_trials(protected),summarize_trials(candidate)
    def ids(rows,kind):
        return {r['id'] for r in rows if (r['clean_success'] if kind=='historical' else
            r['absolute_motion_v1']['clean'] if kind=='world' else
            r['clean_success'] and r['absolute_motion_v1']['clean'])}
    lost={kind:sorted(ids(protected,kind)-ids(candidate,kind)) for kind in ('historical','world','joint')}
    checks={field:after[field]<=before[field] for field in
        ('collisions','falls','joint_limit_trials','operating_speed_trials')}
    checks.update({field:after[field]+1e-12>=before[field] for field in
        ('raw','completed_without_collision','full_duration_world_score')})
    checks.update({kind+'_retention':not values for kind,values in lost.items()})
    improved=after['jointly_clean']>before['jointly_clean'] or (
        after['jointly_clean']==before['jointly_clean'] and
        after['full_duration_world_score']>before['full_duration_world_score']+1e-12)
    by_id={r['id']:r for r in protected}
    return dict(version='sustained-milestone-gate-v1',regression_budget=0,
        continue_training=all(checks.values()),replace_champion=all(checks.values()) and improved,
        checks=checks,failed_checks=sorted(k for k,v in checks.items() if not v),
        lost_joint_ids=lost['joint'],lost_historical_ids=lost['historical'],lost_world_ids=lost['world'],
        before=before,after=after,per_motion_world_score_delta={r['id']:
            r['full_reference_duration_world_score']-by_id[r['id']]['full_reference_duration_world_score']
            for r in candidate},behaviorally_accepted=False,hardware_verified=False)
