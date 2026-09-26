#!/usr/bin/env python3
"""Fixed train-only full-duration official80 retarget preview; retain all rejects."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAKES = [
    ('walk_normal_loop', 'walk_ff_loop_180_R_normal_pace'),
    ('walk_fast_loop', 'walk_ff_loop_360_R_fast'),
    ('walk_start', 'walk_ff_start_180_R_normal_pace'),
    ('walk_stop', 'walk_ff_stop_180_R_normal_pace'),
    ('jog_slow_loop', 'jog_ff_loop_360_R_slow'),
    ('jog_fast_loop', 'jog_ff_loop_360_R_fast'),
    ('jog_start', 'jog_ff_start_360_R_fast'),
    ('jog_stop', 'jog_ff_stop_360_R_normal_pace'),
    ('run_start', 'run_start_180_R'),
    ('body_turn', 'idle_turn_135_L'),
    ('walk_turn', 'turn_walk_270_R'),
    ('jog_turn', 'turn_jog_315_R'),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def initialize(snapshot):
    os.environ['K1_MOTION_ROOT'] = str(ROOT)
    sys.path.insert(0, str(snapshot))
    for name in list(sys.modules):
        if name == 'k1_motion' or name.startswith('k1_motion.'):
            del sys.modules[name]


def measure(clip, target_landmarks, desired_root, limits):
    speed = np.abs(np.diff(clip.values['joint_position'], axis=0)/np.diff(clip.times)[:, None])
    error = clip.values['landmarks']-target_landmarks
    return dict(joint_speed_max_rad_s=float(speed.max()),
        ticks_any_joint_at_legacy_cap=int((speed>=5.99).any(axis=1).sum()),
        ticks_any_joint_above_legacy_cap=int((speed>6+1e-5).any(axis=1).sum()),
        ticks_any_joint_at_own_cap=int((speed>=np.asarray(limits)*.999).any(axis=1).sum()),
        ticks_any_joint_above_own_cap=int((speed>np.asarray(limits)+1e-6).any(axis=1).sum()),
        compared_ticks=len(speed),
        independent_directional_landmark_rms_m=float(np.sqrt(np.mean(error**2))),
        independent_foot_landmark_rms_m=float(np.sqrt(np.mean(error[:,[11,12,15,16]]**2))),
        independent_root_horizontal_rmse_m=float(np.sqrt(np.mean(np.sum((clip.values['root_position'][:,:2]-desired_root[:,:2])**2,axis=-1)))),
        valid_ticks=int(clip.values['valid'].sum()),invalid_ticks=int((~clip.values['valid'].astype(bool)).sum()))


def run_one(job):
    from k1_motion.adapters import bvh_frames
    from k1_motion.calibration import Calibration
    from k1_motion.contracts import MotionClip
    from k1_motion.ground_reference import correct_walking_ground
    from k1_motion.low_pose import directional_targets
    from k1_motion.recovery_geometry import recovery_model
    from k1_motion.recovery_validation import audit_recovery, retarget_recovery
    from k1_motion.retarget_speed import retarget_speed_contract
    from k1_motion.streaming import control_schedule
    row, folder = job
    folder = Path(folder)
    started = time.monotonic()
    robot = recovery_model('retarget')
    frames = list(bvh_frames(folder/'source.bvh', 'bones_seed', row['source_motion_id'], target_hz=50.0))
    human = dict(times=np.array([f.source_time for f in frames]),
                 positions=np.stack([f.positions for f in frames]),
                 orientations=np.stack([f.orientations for f in frames]))
    np.savez_compressed(folder/'canonical-human.npz', **human)
    original = MotionClip.load(folder/'production-original.npz')
    ticks, indices = control_schedule(human['times'], robot.control_dt)
    assert np.array_equal(ticks, original.times), f"Original control clocks changed: {row['id']}"
    assert np.array_equal(human['times'][indices], original.source_times), f"Source clocks changed: {row['id']}"
    calibration = Calibration.from_neutral(frames[0], robot)
    calibrated = [calibration.apply(frames[i]) for i in indices]
    targets = np.stack([directional_targets(robot, f) for f in calibrated])
    desired_root = np.stack([f.positions[0] for f in calibrated])
    metadata = {k: row[k] for k in ['id','capture_group','source_motion_id','family','split','is_mirror']}
    metadata.update(experiment='retarget-speed80-preview-v1',training_eligible=False,physics_qualified=False)
    variants = {}
    generated = {}
    for label, profile in [('legacy6','legacy-command-v1'),('official80','official-80-v1')]:
        clip, reports = retarget_recovery(robot, human, metadata, speed_profile=profile)
        raw_path = folder/f'{label}-raw.npz'
        clip.save(raw_path)
        clip = MotionClip.load(raw_path)
        # Both treatments receive the same pre-existing bounded ground correction.
        corrected, correction = correct_walking_ground(clip)
        path = folder/f'{label}-corrected.npz'
        corrected.save(path)
        corrected = MotionClip.load(path)
        error = float(np.mean([r['rms_landmark_error_m'] for r in reports]))
        error_bound = error+correction['mean_landmark_error_increase_bound_m']
        generated[label] = corrected
        baseline = generated['legacy6']
        baseline_error = error_bound if label=='legacy6' else variants['legacy6']['reported_error_bound_m']
        audit = audit_recovery(corrected, baseline, [{'rms_landmark_error_m':error_bound}],
                               {'rms_landmark_error_m':baseline_error},ground_profile=True,speed_profile=profile)
        contract = retarget_speed_contract(robot, profile)
        variants[label] = dict(retarget_speed_contract=contract,strict_audit=audit,
            raw_ik_error_m=error,reported_error_bound_m=error_bound,ground_correction=correction,
            blocked_steps=reports[-1].get('blocked_steps_total',0),
            metrics=measure(corrected,targets,desired_root,contract['joint_velocity_limits_rad_s']),
            raw_sha256=digest(raw_path),corrected_sha256=digest(path))
        write_json(folder/f'{label}-reports.json',reports)
    # Record the new clip's status under the unchanged historical six-rad/s gate.
    candidate = generated['official80']
    old_gate = audit_recovery(candidate,generated['legacy6'],
        [{'rms_landmark_error_m':variants['official80']['reported_error_bound_m']}],
        {'rms_landmark_error_m':variants['legacy6']['reported_error_bound_m']},ground_profile=True)
    nominal = retarget_speed_contract(robot)['joint_velocity_limits_rad_s']
    original_audit = audit_recovery(original,original,[{'rms_landmark_error_m':row['rms_landmark_error_m']}],row,ground_profile=True)
    original_delta = {k:float(np.max(np.abs(original.values[k]-generated['legacy6'].values[k])))
                      for k in ['root_position','joint_position','landmarks']}
    result = dict(id=row['id'],label=row['preview_label'],capture_group=row['capture_group'],
        frames=len(ticks),duration_s=float(ticks[-1]-ticks[0]),source_sha256=digest(folder/'source.bvh'),
        canonical_human_sha256=digest(folder/'canonical-human.npz'),production_original_sha256=digest(folder/'production-original.npz'),
        original_metrics=measure(original,targets,desired_root,nominal),original_strict_audit=original_audit,
        fresh_legacy_vs_production_max_abs=original_delta,variants=variants,
        official80_under_historical_legacy_speed_gate=old_gate,
        original_clock_exact_match=True,model_signature=robot.signature,
        elapsed_seconds=time.monotonic()-started,physics_qualified=False,training_eligible=False)
    write_json(folder/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,default=Path('/mnt/storage/k1-motion/datasets/bones-seed'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=4:
        raise ValueError('Bounded preview allows one to four workers')
    from freeze_source import freeze_source
    snapshot,revision=freeze_source(ROOT)
    initialize(snapshot)
    from k1_motion.ground_reference import GROUND_REFERENCE_SETTINGS
    from k1_motion.recovery_validation import RECOVERY_GATES
    from k1_motion.reference_admission import take_family
    from k1_motion.retarget_speed import retarget_speed_contract
    from k1_motion.robot import K1Model
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    index=ROOT/'artifacts/broad-motion-coverage-20260920/curated-v1/pool/index.jsonl'
    manifest=json.loads((ROOT/'manifests/minimal-casual-curriculum-v1.json').read_text())
    rows=[json.loads(x) for x in index.open()]
    selected=[]
    jobs=[]
    for label,name in TAKES:
        candidates=[r for r in rows if r['split']=='train' and not r.get('is_mirror')
                    and take_family(r['capture_group'])=='bones_seed/'+name
                    and r['retarget_version']=='causal-ik-v8-command-path-recovery'
                    and r['id'] in manifest['locomotion_ids']]
        if not candidates:
            raise ValueError(f'No current v8 train-only clear-locomotion candidate: {name}')
        row=min(candidates,key=lambda r:hashlib.sha256(('speed80-preview-v1/'+r['id']).encode()).hexdigest())
        row={**row,'preview_label':label}
        folder=output/row['id']
        folder.mkdir()
        shutil.copy2(args.dataset_root/row['source_path'],folder/'source.bvh')
        shutil.copy2(row['reference_path'],folder/'production-original.npz')
        selected.append(row)
        jobs.append((row,str(folder)))
    assert len({take_family(r['capture_group']) for r in selected})==12
    assert not {take_family(r['capture_group']) for r in selected} & {take_family(r['capture_group']) for r in rows if r['split']!='train'}
    robot=K1Model()
    config=dict(version='retarget-speed80-preview-v1',source_snapshot=str(snapshot),source_revision=revision,
        script_sha256=digest(__file__),physical_config_path=str(ROOT/'configs/k1.json'),
        physical_config_sha256=digest(ROOT/'configs/k1.json'),model_signature=robot.signature,
        profiles={name:retarget_speed_contract(robot,name) for name in ['legacy-command-v1','official-80-v1']},
        panel_selection='Fixed12 train-only related takes; currentv8; audited clear locomotion; deterministic ID hash, no policy outcomes',
        gates=RECOVERY_GATES,ground_correction=GROUND_REFERENCE_SETTINGS,workers=args.workers,
        preserved_production_library=True,full_conversion_authorized=False,
        fidelity_measure='Independent source-calibrated directional targets and horizontal root trajectory; same targets both treatments',
        acceptance_scope='Strict500Hz geometry/reference audit; no physics/controller/hardware qualification')
    write_json(output/'retarget-config.json',config)
    write_json(output/'panel.json',selected)
    shutil.copy2(__file__,output/'preview_retarget_speed.py')
    started=time.monotonic()
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(snapshot,)) as pool:
        for future in as_completed([pool.submit(run_one,job) for job in jobs]):
            result=future.result()
            results.append(result)
            print(json.dumps(dict(id=result['id'],label=result['label'],complete=len(results),total=len(jobs),
                legacy_pass=result['variants']['legacy6']['strict_audit']['accepted'],
                official80_pass=result['variants']['official80']['strict_audit']['accepted'])),flush=True)
    assert digest(ROOT/'configs/k1.json')==config['physical_config_sha256']
    results.sort(key=lambda r:r['label'])
    report=dict(contract=config,rows=results,completed=12,elapsed_seconds=time.monotonic()-started,
        strict_acceptance={name:sum(r['variants'][name]['strict_audit']['accepted'] for r in results) for name in ['legacy6','official80']},
        rejected_attempts_retained=True,production_mutations=False,actor_evaluations=0)
    write_json(output/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['contract','rows']},indent=2))


if __name__=='__main__':
    main()
