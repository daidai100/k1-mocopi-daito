"""Explicit development qualification before ranking motion-policy candidates.

This preserves the strongest recorded result from each protected reference. It
never promotes a policy or substitutes development evidence for confirmation.
"""
import math


LOWER_IS_BETTER = ('fell', 'collision_trials', 'operating_overspeed_trials',
                   'nominal_overspeed_trials', 'joint_limit_violation_trials')
HIGHER_IS_BETTER = ('completed', 'full_reference_duration_world_score')


def _validate_summary(summary, panel_ids):
    values = summary['all']
    count_fields = ('trials', 'completed', 'clean', 'absolute_clean', *LOWER_IS_BETTER)
    for field in count_fields:
        value = values.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= len(panel_ids):
            raise ValueError(f'Invalid policy selection count: {field}')
    if values['trials'] != len(panel_ids) or summary['contract'].get('originals') != len(panel_ids):
        raise ValueError('Policy selection requires the entire fixed panel')
    score = values.get('full_reference_duration_world_score')
    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid full-reference-duration world score')
    if summary.get('execution_errors') != 0 or summary.get('resets_during_trials', 0) != 0:
        raise ValueError('Policy selection requires error-free, uninterrupted evaluation')
    for field, count_field in (('clean_ids', 'clean'), ('absolute_clean_ids', 'absolute_clean')):
        ids = summary.get(field, [])
        if (len(set(ids)) != len(ids) or not set(ids) <= panel_ids or len(ids) != values[count_field]
                or len(ids) > values['completed']):
            raise ValueError(f'Invalid policy selection pass identities: {field}')


def assess_policy_candidate(candidate, protected, panel, *, minimum_new_ordinary_walk_passes=2):
    """Return qualification and a separate ranking key against protected results.

    ``candidate`` and named ``protected`` values use ``aggregate_replay`` output;
    ``panel`` is the same original-motion panel used for those replays. All
    records must share the frozen evaluator revision and panel digest. Preserve
    the union of historical and world+safety pass IDs, the best raw completion,
    world score, and minimum counts of falls, collisions, and each safety
    violation. Require two new ordinary walks passing BOTH historical and
    world+safety gates. There is no tolerance or trade between these gates.

    A non-null ranking key is returned only after qualification. Ranking uses
    locomotion and clean coverage before score; it grants no acceptance status.
    """
    if not protected:
        raise ValueError('At least one protected policy is required')
    if (not isinstance(minimum_new_ordinary_walk_passes, int)
            or isinstance(minimum_new_ordinary_walk_passes, bool) or minimum_new_ordinary_walk_passes < 2):
        raise ValueError('Require at least two new ordinary-walk joint passes')
    panel_ids = {row['id'] for row in panel}
    if not panel_ids or len(panel_ids) != len(panel):
        raise ValueError('Policy selection requires a nonempty unique original panel')
    if any('semantic_group' not in row for row in panel):
        raise ValueError('Audited panel semantics are required')
    summaries = [candidate, *protected.values()]
    for summary in summaries:
        _validate_summary(summary, panel_ids)
    for field in ('source_revision', 'panel_sha256'):
        identities = {summary['contract'].get(field) for summary in summaries}
        if len(identities) != 1 or not next(iter(identities)):
            raise ValueError(f'Policy selection requires matching evaluator {field}')

    historical = set().union(*(set(summary['clean_ids']) for summary in protected.values()))
    absolute = set().union(*(set(summary['absolute_clean_ids']) for summary in protected.values()))
    candidate_historical, candidate_absolute = set(candidate['clean_ids']), set(candidate['absolute_clean_ids'])
    envelope = {field: min(summary['all'][field] for summary in protected.values())
                for field in LOWER_IS_BETTER}
    envelope.update({field: max(summary['all'][field] for summary in protected.values())
                     for field in HIGHER_IS_BETTER})
    envelope.update(historical_clean_count=len(historical), absolute_clean_count=len(absolute),
                    historical_clean_ids=sorted(historical), absolute_clean_ids=sorted(absolute))
    checks = {field: candidate['all'][field] <= envelope[field] for field in LOWER_IS_BETTER}
    checks.update({field: candidate['all'][field] >= envelope[field] for field in HIGHER_IS_BETTER})
    checks.update(historical_clean_ids=historical <= candidate_historical,
                  absolute_clean_ids=absolute <= candidate_absolute,
                  historical_clean_count=candidate['all']['clean'] >= len(historical),
                  absolute_clean_count=candidate['all']['absolute_clean'] >= len(absolute))
    walks = {row['id'] for row in panel if row['semantic_group']=='ordinary_walk'}
    runs = {row['id'] for row in panel if row['semantic_group']=='ordinary_run'}
    new_historical = (candidate_historical-historical)&walks
    new_absolute = (candidate_absolute-absolute)&walks
    new_joint = new_historical&new_absolute
    progress = dict(new_ordinary_walk_historical_passes=len(new_historical),
                    new_ordinary_walk_absolute_passes=len(new_absolute),
                    new_ordinary_walk_joint_passes=len(new_joint), new_ordinary_walk_joint_ids=sorted(new_joint))
    checks['new_ordinary_walk_joint_passes'] = len(new_joint) >= minimum_new_ordinary_walk_passes
    qualified = all(checks.values())
    joint_passes = candidate_historical&candidate_absolute
    ranking = [len(joint_passes&walks), len(joint_passes&runs), len(candidate_absolute),
               len(candidate_historical), -candidate['all']['collision_trials'],
               candidate['all']['full_reference_duration_world_score']]
    return dict(version='protected-motion-development-selection-v1', qualification_passed=qualified,
        checks=checks, failed_checks=sorted(name for name, passed in checks.items() if not passed),
        protected_policies=list(protected), protected_envelope=envelope, progress=progress,
        lost_historical_ids=sorted(historical-candidate_historical), lost_absolute_ids=sorted(absolute-candidate_absolute),
        ranking_key=ranking if qualified else None,
        ranking_fields=['ordinary_walk_joint_passes', 'ordinary_run_joint_passes', 'absolute_clean',
                        'historical_clean', 'negative_collision_trials', 'full_reference_duration_world_score'],
        automatically_promoted=False, behaviorally_accepted=False, hardware_verified=False,
        scope='Development qualification and ranking only; untouched confirmation remains required')
