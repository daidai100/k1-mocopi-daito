"""Test-first selection failures: traded passes, hidden safety regressions,
missing locomotion, malformed identities, different evaluators, and auto-promotion.
"""
import copy
import json
from pathlib import Path

import pytest


@pytest.fixture
def evidence():
    return json.loads((Path(__file__).parent/'fixtures/policy_selection_development_20260922.json').read_text())


def protected(evidence):
    return {key: evidence['results'][key] for key in ('guard_only', 'guard_world')}


def improving_candidate(evidence):
    references = protected(evidence)
    candidate = copy.deepcopy(references['guard_only'])
    for field in ('clean_ids', 'absolute_clean_ids'):
        candidate[field] = sorted(set().union(*(row[field] for row in references.values())))
    walks = [row['id'] for row in evidence['panel'] if row['semantic_group']=='ordinary_walk'][:2]
    candidate['clean_ids'] = sorted(set(candidate['clean_ids'])|set(walks))
    candidate['absolute_clean_ids'] = sorted(set(candidate['absolute_clean_ids'])|set(walks))
    candidate['all'].update(clean=len(candidate['clean_ids']), absolute_clean=len(candidate['absolute_clean_ids']),
                            collision_trials=14, full_reference_duration_world_score=.52)
    return candidate


def test_known_pilots_fail_protected_envelope(evidence):
    from k1_motion.policy_selection import assess_policy_candidate
    for name in ('guard_world', 'guard_velocity'):
        result = assess_policy_candidate(evidence['results'][name], protected(evidence), evidence['panel'])
        assert not result['qualification_passed']
        assert result['ranking_key'] is None
        assert not result['automatically_promoted']
        assert 'operating_overspeed_trials' in result['failed_checks']
        assert 'joint_limit_violation_trials' in result['failed_checks']
        assert 'new_ordinary_walk_joint_passes' in result['failed_checks']
    assert 'nominal_overspeed_trials' in result['failed_checks']


def test_qualification_preserves_union_and_requires_real_locomotion(evidence):
    from k1_motion.policy_selection import assess_policy_candidate
    candidate = improving_candidate(evidence)
    result = assess_policy_candidate(candidate, protected(evidence), evidence['panel'])
    assert result['qualification_passed'] and result['ranking_key'] is not None
    assert result['protected_envelope']['historical_clean_count'] == 19
    assert result['protected_envelope']['absolute_clean_count'] == 14
    assert result['progress']['new_ordinary_walk_joint_passes'] == 2
    assert candidate['all']['clean'] == 21 and candidate['all']['absolute_clean'] == 16
    assert not result['automatically_promoted'] and not result['behaviorally_accepted']


@pytest.mark.parametrize('field,direction', [('completed', -1), ('fell', 1), ('collision_trials', 1),
    ('operating_overspeed_trials', 1), ('nominal_overspeed_trials', 1), ('joint_limit_violation_trials', 1),
    ('full_reference_duration_world_score', -1)])
def test_each_regression_blocks_qualification_before_ranking(evidence, field, direction):
    from k1_motion.policy_selection import assess_policy_candidate
    candidate = improving_candidate(evidence)
    references = protected(evidence)
    values = [row['all'][field] for row in references.values()]
    edge = max(values) if direction < 0 else min(values)
    candidate['all'][field] = edge + direction*(1e-8 if field.endswith('score') else 1)
    result = assess_policy_candidate(candidate, references, evidence['panel'])
    assert not result['qualification_passed'] and field in result['failed_checks']


@pytest.mark.parametrize('field', ['clean_ids', 'absolute_clean_ids'])
def test_same_counts_cannot_trade_away_protected_passes(evidence, field):
    from k1_motion.policy_selection import assess_policy_candidate
    candidate = improving_candidate(evidence)
    old = candidate[field].pop(0)
    replacement = next(row['id'] for row in evidence['panel'] if row['id'] not in candidate[field] and row['id']!=old)
    candidate[field].append(replacement)
    result = assess_policy_candidate(candidate, protected(evidence), evidence['panel'])
    assert not result['qualification_passed']
    assert ('historical_clean_ids' if field=='clean_ids' else 'absolute_clean_ids') in result['failed_checks']


def test_world_only_walk_progress_does_not_hide_historical_failure(evidence):
    from k1_motion.policy_selection import assess_policy_candidate
    candidate = improving_candidate(evidence)
    walk_ids = {row['id'] for row in evidence['panel'] if row['semantic_group']=='ordinary_walk'}
    candidate['clean_ids'] = [key for key in candidate['clean_ids'] if key not in walk_ids]
    candidate['all']['clean'] = len(candidate['clean_ids'])
    result = assess_policy_candidate(candidate, protected(evidence), evidence['panel'])
    assert result['progress']['new_ordinary_walk_absolute_passes'] == 2
    assert result['progress']['new_ordinary_walk_historical_passes'] == 0
    assert not result['qualification_passed']


@pytest.mark.parametrize('mutation', ['different_source', 'different_panel', 'duplicate_pass', 'unknown_pass',
                                     'count_mismatch', 'nan_score', 'negative_count', 'execution_error'])
def test_untrustworthy_comparisons_are_rejected(evidence, mutation):
    from k1_motion.policy_selection import assess_policy_candidate
    candidate = improving_candidate(evidence)
    if mutation=='different_source':
        candidate['contract']['source_revision'] = 'other'
    elif mutation=='different_panel':
        candidate['contract']['panel_sha256'] = 'other'
    elif mutation=='duplicate_pass':
        candidate['clean_ids'].append(candidate['clean_ids'][0])
    elif mutation=='unknown_pass':
        candidate['clean_ids'][0] = 'unknown'
    elif mutation=='count_mismatch':
        candidate['all']['clean'] += 1
    elif mutation=='nan_score':
        candidate['all']['full_reference_duration_world_score'] = float('nan')
    elif mutation=='negative_count':
        candidate['all']['fell'] = -1
    elif mutation=='execution_error':
        candidate['execution_errors'] = 1
    with pytest.raises(ValueError):
        assess_policy_candidate(candidate, protected(evidence), evidence['panel'])
