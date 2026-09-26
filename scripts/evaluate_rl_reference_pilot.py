#!/usr/bin/env python3
"""Replay a frozen validation panel without resets or admission feedback loops."""
# ruff: noqa: E402
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from freeze_source import freeze_source

ROOT = Path(__file__).resolve().parents[1]
FROZEN, REVISION = freeze_source(ROOT, os.environ.get("K1_FROZEN_SOURCE"))
os.environ["K1_MOTION_ROOT"] = str(ROOT)
sys.path.insert(0, str(FROZEN))
from k1_motion.contracts import MotionClip
from k1_motion.control_validation import replay_clip
from k1_motion.learning import Policy
from k1_motion.reference_admission import take_family
from k1_motion.robot import K1Model

ROBOT = POLICY = None


def initialize(path):
    global ROBOT, POLICY
    import torch
    torch.set_num_threads(1)
    ROBOT = K1Model()
    POLICY = Policy(path, ROBOT.signature)


def failed_trial(row, output, error):
    result = {**row, 'execution_passed': False, 'error': f'{type(error).__name__}: {error}',
              'completed': False, 'clean_success': False}
    (Path(output) / (row['id'] + '.json')).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def trial(job):
    row, output = job
    try:
        clip = MotionClip.load(row["reference_path"])
        for key in ('capture_group', 'family'):
            if clip.metadata.get(key) != row[key]:
                raise ValueError(f"Panel/reference identity mismatch: {key}")
        parents = {take_family(p) for p in POLICY.metadata["train_parents"]}
        if take_family(row["capture_group"]) in parents:
            raise ValueError("Evaluation take family overlaps policy training provenance")
        digest = hashlib.sha256(Path(row['reference_path']).read_bytes()).hexdigest()
        if row.get('reference_sha256', digest) != digest:
            raise ValueError('Panel reference payload changed')
        result = {**row, **replay_clip(ROBOT, POLICY, clip, Path(output) / (row["id"] + ".npz")),
                  'reference_sha256': digest, 'execution_passed': True}
    except Exception as error:
        return failed_trial(row, output, error)
    (Path(output) / (row["id"] + ".json")).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    rows = json.loads(args.panel.read_text())
    if args.workers < 1 or not rows:
        raise ValueError("Positive workers and a nonempty panel are required")
    for row in rows:
        identifier = row.get("id") if isinstance(row, dict) else None
        if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier)
                or identifier in {"summary", "contract", "motion-axes-summary"}):
            raise ValueError("Panel id must be a safe, non-reserved filename stem")
        if any(not isinstance(row.get(key), str) or not row[key]
               for key in ("cohort", "family", "capture_group", "reference_path")):
            raise ValueError("Panel row requires cohort, family, capture_group and reference_path")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate evaluation motion")
    args.output.mkdir(parents=True, exist_ok=False)
    contract = {"source_revision": REVISION, "policy": str(args.policy.resolve()),
                "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                "policy_metadata_sha256": hashlib.sha256(args.policy.with_suffix('.json').read_bytes()).hexdigest(),
                "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(), "originals": len(rows),
                "reference_scale": json.loads(args.policy.with_suffix(".json").read_text()).get("reference_scale")}
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    results = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                 initargs=(str(args.policy),)) as pool:
            futures = {pool.submit(trial, (r, str(args.output))): r for r in rows}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as error:
                    result = failed_trial(futures[future], args.output, error)
                results.append(result)
                print(json.dumps({k: result[k] for k in ("id", "cohort", "family", "completed", "clean_success")}), flush=True)
    except Exception as error:
        # Pool startup/submission can fail before a future exists for each row.
        finished = {r['id'] for r in results}
        results.extend(failed_trial(r, args.output, error) for r in rows if r['id'] not in finished)
    errors = sum(not r["execution_passed"] for r in results)
    summary = {"contract": contract, "execution_errors": errors, "execution_passed": errors == 0, "resets_during_trials": 0,
               "behaviorally_accepted": False, "hardware_verified": False}
    for cohort in ["all", *sorted({r["cohort"] for r in rows})]:
        group = [r for r in results if cohort == "all" or r["cohort"] == cohort]
        valid = [r for r in group if r["execution_passed"]]
        summary[cohort] = {"trials": len(group), "completed": sum(r["completed"] for r in group),
                           "clean": sum(r["clean_success"] for r in group),
                           "collision_trials": sum(r["self_collision_ticks"] > 0 for r in valid),
                           "fell": sum(r["fell"] for r in valid),
                           "mean_root_velocity_rmse_m_s": (sum(r["root_velocity_rmse_m_s"] for r in valid) / len(valid) if valid else None)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if errors:
        raise SystemExit(f"{errors} replay execution errors; see summary.json and trial records")
    from summarize_motion_axes import summarize
    summarize(args.panel, args.output, args.output / 'motion-axes-summary.json')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
