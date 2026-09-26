#!/usr/bin/env python3
"""Freeze an all-reference casual mix using conservative names plus measured motion."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

VERSION = 'minimal-casual-curriculum-v1'
TRANSLATION_NAME = re.compile(r'^(?:walk_|walking_|relaxed_walk_|jog_|run_|turn_walk_|turn_jog_)')
TURN_NAME = re.compile(r'^(?:idle_turn_|crossed_arms_idle_turn_|neutral_dancecard_idle_turn_)')
TRANSLATION_EVENTS = ('forward_travel', 'backward_travel', 'leftward_travel', 'rightward_travel')


def semantic_assignment(row, features):
    """Underclassification is deliberate: unconfirmed targets remain in broad coverage."""
    name = row.get('take_name', row.get('filename', '')).lower()
    longest = features.get('event_longest_s', {})
    sustained_translation = max((float(longest.get(k, 0)) for k in TRANSLATION_EVENTS), default=0.)
    sustained_turn = float(longest.get('turning', 0.))
    translation = bool(TRANSLATION_NAME.search(name)) and sustained_translation >= .3
    turning = bool(TURN_NAME.search(name)) and sustained_turn >= .2
    return {'group': 'clear_locomotion' if translation or turning else 'broad_remainder',
            'intent': 'translation' if translation else 'body_turn' if turning else 'unconfirmed_or_broad',
            'name': name, 'sustained_translation_s': sustained_translation,
            'sustained_yaw_rate_s': sustained_turn}


def build_contract(rows, features):
    from k1_motion.reference_admission import take_family
    rows = sorted(rows, key=lambda r: r['id'])
    ids = [r['id'] for r in rows]
    if (not rows or len(set(ids)) != len(ids) or any(r['split'] != 'train' or r.get('is_mirror') for r in rows)):
        raise ValueError('Casual curriculum requires unique original training references')
    if any(key not in features for key in ids):
        raise ValueError('Missing saved motion feature for training reference')
    assignments = {r['id']: semantic_assignment(r, features[r['id']]) for r in rows}
    selected = [r['id'] for r in rows if assignments[r['id']]['group'] == 'clear_locomotion']
    if not selected or len(selected) == len(rows):
        raise ValueError('Casual curriculum requires both motion groups')
    weights = {}
    group_summary = {}
    for label in ['clear_locomotion', 'broad_remainder']:
        subset = [r for r in rows if assignments[r['id']]['group'] == label]
        group_keys = [(r['family'], take_family(r['capture_group'])) for r in subset]
        clips_per_take = Counter(group for _, group in group_keys)
        takes_per_family = Counter(f for f, _ in set(group_keys))
        for r, (_, group) in zip(subset, group_keys):
            weights[r['id']] = .5 / len(clips_per_take) / clips_per_take[group]
        group_summary[label] = {'clips': len(subset), 'take_families': len({g for _, g in group_keys}),
                                'family_clips': dict(sorted(Counter(r['family'] for r in subset).items())),
                                'family_take_counts': dict(sorted(takes_per_family.items())),
                                'target_transition_share': .5}
    family_targets = defaultdict(float)
    for r in rows:
        family_targets[r['family']] += weights[r['id']]
    return {'version': VERSION, 'train_ids': ids, 'locomotion_ids': selected,
            'sampling': {'locomotion_transition_share': .5, 'annealing': False,
                         'within_group': 'equal_related_take_then_equal_clip; native_family_is_telemetry_only',
                         'episode_duration_correction': 'target_transition_weight / episode_duration_ema.clamp(min=1)'},
            'reset_mix': {'start': .5, 'failure_biased': .25, 'uniform': .25},
            'failure_ledger': 'unchanged fall or coarse pose-tracking failure only; speed, progress and collision diagnostics deferred',
            'target_transition_weights': dict(sorted(weights.items())),
            'target_family_transition_shares': dict(sorted(family_targets.items())),
            'groups': group_summary, 'semantic_assignments': assignments,
            'semantic_rule': {'translation_name_regex': TRANSLATION_NAME.pattern,
                              'body_turn_name_regex': TURN_NAME.pattern,
                              'translation': 'At least 0.3 continuous seconds with a body-frame horizontal component above 0.15 m/s',
                              'body_turn': 'At least 0.2 continuous seconds with absolute root angular-Z velocity above 0.5 rad/s',
                              'limits': 'Measured root angular-Z rate is a turning proxy; this is conservative semantic sampling, not dynamic qualification'},
            'preservation': 'All admitted originals retain positive mass, original family labels, reference arrays, clocks and admission gates unchanged'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite a frozen curriculum')
    index = args.library / 'index.jsonl'
    payload = index.read_bytes()
    rows = [r for line in payload.splitlines() if (r := json.loads(line))['split'] == 'train']
    by_id = {r['id']: r for r in rows}
    features = {}
    for line in args.features.open():
        record = json.loads(line)
        key = record['key']
        if key['id'] not in by_id:
            continue
        row = by_id[key['id']]
        reference = (args.library / row['reference_path']).resolve()
        stat = reference.stat()
        if (Path(key['path']).resolve() != reference or stat.st_size != key['size']
                or stat.st_mtime_ns != key['mtime_ns'] or row['model_signature'] != key['model_signature']
                or record['features']['frames'] != row['frames']):
            raise ValueError(f'Stale motion features: {row["id"]}')
        if key['id'] in features:
            raise ValueError('Duplicate cached motion features')
        features[key['id']] = record['features']
    contract = build_contract(rows, features)
    contract['source'] = {'library_manifest': str(index.resolve()), 'manifest_sha256': hashlib.sha256(payload).hexdigest(),
                          'features': str(args.features.resolve()),
                          'features_identity_check': 'Each train payload path, size, mtime, frame count and model signature checked against existing measured-feature record',
                          'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'train_originals': len(rows), 'groups': contract['groups'],
                      'target_family_transition_shares': contract['target_family_transition_shares']}, indent=2))


if __name__ == '__main__':
    main()
