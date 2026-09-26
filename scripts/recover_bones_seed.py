#!/usr/bin/env python3
"""Follow a pinned conversion ledger, independently audit it, and retry repairs."""
# Checkout entrypoint: bootstrap imports before importing project modules.
# ruff: noqa: E402

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from convert_bones_seed import records, shard_for_capture_group, summarize
from k1_motion.adapters import bvh_frames
from k1_motion.contracts import MotionClip
from k1_motion.recovery_validation import RECOVERY_GATES, audit_recovery, retarget_recovery
from k1_motion.retarget_recovery import RETARGET_VERSION
from k1_motion.recovery_geometry import recovery_model
from k1_motion.robot import K1Model

EXECUTION_ID = None
SOURCE_FILES = [
    "scripts/recover_bones_seed.py", "src/k1_motion/retarget_recovery.py",
    "src/k1_motion/recovery_validation.py", "src/k1_motion/recovery_geometry.py",
    "src/k1_motion/adapters.py", "src/k1_motion/robot.py", "src/k1_motion/streaming.py",
    "src/k1_motion/contracts.py", "src/k1_motion/calibration.py", "src/k1_motion/math3d.py",
    "scripts/convert_bones_seed.py",
]


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def initialize_worker(execution_id):
    global EXECUTION_ID
    EXECUTION_ID = execution_id


def bind_execution(output, candidate, parity_path=None):
    """Keep the original campaign immutable and bind a verified performance epoch.

    Earlier rows retain the original source contract. New rows name an execution
    receipt with the exact new sources and worker count. Model, sampling, solver,
    gates, metadata and source assignment cannot change through this mechanism.
    """
    contract_path = output / "campaign.json"
    if not contract_path.exists():
        atomic_json(contract_path, candidate)
        return candidate, None
    original = json.loads(contract_path.read_text())
    if original == candidate:
        return original, None
    if parity_path is None:
        raise ValueError("Recovery contract changed; choose a new output version or supply exact performance parity evidence")
    mutable = {"source_hashes", "workers", "busy_workers"}
    if ({k: v for k, v in original.items() if k not in mutable}
            != {k: v for k, v in candidate.items() if k not in mutable}):
        raise ValueError("Performance continuation changed the semantic campaign contract")
    proof = json.loads(parity_path.read_text())
    if (proof.get("baseline_source_hashes") != original["source_hashes"]
            or proof.get("candidate_source_hashes") != candidate["source_hashes"]
            or not proof.get("complete") or not proof.get("rows")
            or not all(r.get("payload_and_audit_exact_match") is True for r in proof["rows"])):
        raise ValueError("Performance parity evidence is incomplete or bound to different sources")
    execution = {
        "version": 1, "candidate_contract": candidate,
        "original_campaign_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "parity_report": str(parity_path.resolve()),
        "parity_report_sha256": hashlib.sha256(parity_path.read_bytes()).hexdigest(),
        "scope": "Performance-only continuation; original ledger and admission gates preserved",
    }
    execution_id = hashlib.sha256(json.dumps(execution, sort_keys=True).encode()).hexdigest()
    directory = output / "executions"
    directory.mkdir(exist_ok=True)
    atomic_json(directory / f"{execution_id}.json", execution)
    return original, execution_id


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def recover(job):
    original, dataset, baseline_root, output = job
    dataset, baseline_root, output = map(Path, (dataset, baseline_root, output))
    started = time.monotonic()
    row = {**original, "baseline_kinematics_accepted": original["kinematics_accepted"],
           "baseline_retarget_version": original.get("retarget_version"),
           "physics_qualified": False, "training_eligible": False,
           "recovery_pipeline": "bones-seed-recovery-v1"}
    if EXECUTION_ID is not None:
        row["recovery_execution"] = EXECUTION_ID
    row.pop("reference_path", None)
    if not original.get("reference_path"):
        return {**row, "kinematics_accepted": False, "recovery_status": "source_error_retained"}
    baseline_path = baseline_root / original["reference_path"]
    baseline = MotionClip.load(baseline_path)
    if (baseline.metadata["source_motion_id"] != original["source_motion_id"]
            or baseline.metadata["capture_group"] != original["capture_group"]):
        raise ValueError("Baseline clip and ledger lineage mismatch")
    row["baseline_reference_path"] = str(baseline_path)
    # A known-invalid frame or control-clock speed violation already requires a
    # retry. Avoid a redundant 500 Hz geometry replay of that rejected baseline.
    robot = recovery_model("retarget")
    if baseline.metadata.get("model_signature") != robot.signature:
        raise ValueError("Baseline clip model contract changed")
    baseline_speed = np.abs(np.diff(baseline.values["joint_position"], axis=0)
                            / np.diff(baseline.times)[:, None])
    precheck_reasons = []
    if not np.all(baseline.values["valid"]):
        precheck_reasons.append("retarget_invalid_ticks")
    if np.any(baseline_speed > np.minimum(robot.velocity_limit, robot.config["command_velocity_limit"]) + 1e-6):
        precheck_reasons.append("control_clock_velocity_limit")
    if precheck_reasons:
        baseline_audit = {"accepted": False, "rejection_reasons": precheck_reasons,
                          "geometry_audited": False, "physics_qualified": False,
                          "training_eligible": False}
    else:
        baseline_audit = audit_recovery(
            baseline, baseline, [{"rms_landmark_error_m": original["rms_landmark_error_m"]}], original)
    row["baseline_audit"] = baseline_audit
    if baseline_audit["accepted"]:
        selected, audit = baseline, baseline_audit
        row.update(kinematics_accepted=True, recovery_status="baseline_passed_stricter_audit")
    else:
        frames = list(bvh_frames(dataset / original["source_path"], "bones_seed",
                                 original["source_motion_id"], target_hz=50.0))
        human = {"times": np.array([frame.source_time for frame in frames]),
                 "positions": np.stack([frame.positions for frame in frames]),
                 "orientations": np.stack([frame.orientations for frame in frames])}
        selected, reports = retarget_recovery(robot, human, original)
        # The audit reads the actual persisted/reloaded payload, not solver flags.
        attempt_path = output / "attempts" / (row["id"] + ".npz")
        selected.metadata.update(kinematics_accepted=False, recovery_status="unverified_attempt")
        selected.save(attempt_path)
        selected = MotionClip.load(attempt_path)
        audit = audit_recovery(selected, baseline, reports, original)
        invalid = ~selected.values["valid"].astype(bool)
        row.update(
            retarget_version=RETARGET_VERSION,
            kinematics_accepted=audit["accepted"],
            recovery_status="repaired_and_audited" if audit["accepted"] else "repair_rejected",
            attempt_reference_path=f"attempts/{row['id']}.npz",
            frames=len(selected.times), valid_ticks=int((~invalid).sum()),
            rejected_ticks=int(invalid.sum()), rejected_tick_fraction=float(invalid.mean()),
            rejection_counts=dict(Counter(reason for report in reports for reason in report["rejection_reasons"])),
            rms_landmark_error_m=float(np.mean([r["rms_landmark_error_m"] for r in reports])),
            retarget_p95_ms=float(np.percentile([r["seconds"] for r in reports], 95) * 1000),
            calibration=selected.metadata["calibration"], sampling=selected.metadata["sampling"],
        )
    row["recovery_audit"] = audit
    row["elapsed_seconds"] = time.monotonic() - started
    if row["kinematics_accepted"]:
        row["reference_path"] = f"clips/{row['id']}.npz"
        selected.metadata.update(row)
        selected.save(output / row["reference_path"])
        reloaded = MotionClip.load(output / row["reference_path"])
        if (reloaded.metadata["id"] != row["id"]
                or not np.array_equal(reloaded.values["joint_position"], selected.values["joint_position"])):
            raise ValueError("Final recovery artifact failed reload verification")
    return row


def is_active(unit):
    if not unit:
        return False
    result = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True)
    return result.stdout.strip() in {"active", "activating"}


def read_complete_lines(stream):
    while True:
        position = stream.tell()
        line = stream.readline()
        if not line:
            return
        if not line.endswith("\n"):
            stream.seek(position)
            return
        yield json.loads(line)


def report(rows, expected, contract, output, running, execution_id=None):
    result = summarize(list(rows.values()), expected)
    result.update(campaign_contract=contract, running=running, pipeline="bones-seed-recovery-v1",
                  recovery_status_counts=dict(Counter(r["recovery_status"] for r in rows.values())),
                  recovered_originals=sum(not r["is_mirror"] and not r["baseline_kinematics_accepted"]
                                          and r["kinematics_accepted"] for r in rows.values()),
                  baseline_original_acceptances_withheld=sum(not r["is_mirror"]
                                                             and r["baseline_kinematics_accepted"]
                                                             and not r["kinematics_accepted"] for r in rows.values()),
                  physics_qualified_hours=0.0, training_eligible_motions=0,
                  scope="Causal retargeting and independent 500 Hz geometric/reference audit; dynamic qualification pending.")
    if execution_id is not None:
        result["active_execution"] = execution_id
    atomic_json(output / "summary.json", result)
    print(json.dumps({k: result[k] for k in ["processed_rows", "expected_metadata_rows", "accepted_original_motions",
                                            "recovered_originals", "baseline_original_acceptances_withheld", "running"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--busy-workers", type=int)
    parser.add_argument("--follow-service")
    parser.add_argument("--ids", type=Path, help="Frozen canary IDs, never production truncation")
    parser.add_argument("--performance-parity", type=Path,
                        help="Exact payload/audit parity report for a performance-only continuation")
    args = parser.parse_args()
    if args.workers < 1 or args.busy_workers is not None and not 1 <= args.busy_workers <= args.workers:
        raise ValueError("Invalid worker counts")
    baseline_contract = json.loads((args.baseline / "campaign.json").read_text())
    shard_count, shard_index = baseline_contract["shard_count"], baseline_contract["shard_index"]
    ids = set(json.loads(args.ids.read_text())) if args.ids else None
    source = [r for r in records(args.dataset_root / "metadata/seed_metadata_v004.parquet")
              if shard_for_capture_group(r["capture_group"], shard_count) == shard_index
              and (ids is None or r["id"] in ids)]
    expected_ids = {r["id"] for r in source}
    source_by_id = {r["id"]: r for r in source}
    contract = {"version": 1, "baseline_campaign": baseline_contract,
                "source_hashes": source_hashes(),
                "metadata_sha256": hashlib.sha256((args.dataset_root / "metadata/seed_metadata_v004.parquet").read_bytes()).hexdigest(),
                "model_signature": K1Model().signature, "retarget_version": RETARGET_VERSION,
                "workers": args.workers, "busy_workers": args.busy_workers,
                "expected_rows": len(source), "canary_ids": sorted(ids) if ids is not None else None,
                "gates": RECOVERY_GATES, "baseline_preserved": True,
                "mixed_solver_policy": "Each selected record retains its exact v7 or v8 solver provenance; all pass the same independent audit."}
    if contract["model_signature"] != baseline_contract["model_signature"]:
        raise ValueError("Baseline model contract changed")
    if contract["metadata_sha256"] != baseline_contract["metadata_sha256"]:
        raise ValueError("Baseline metadata contract changed")
    for folder in [args.output, args.output / "clips", args.output / "attempts"]:
        folder.mkdir(parents=True, exist_ok=True)
    contract, execution_id = bind_execution(args.output, contract, args.performance_parity)
    index = args.output / "index.jsonl"
    done = {}
    if index.exists():
        with index.open("r+") as f:
            for row in read_complete_lines(f):
                if row["id"] not in expected_ids or row["id"] in done:
                    raise ValueError("Unexpected or duplicate recovery ledger ID")
                done[row["id"]] = row
            f.truncate(f.tell())  # Only discard an interrupted, unterminated final write.
    submitted = set(done)
    pending = deque()
    last_report = last_active_check = 0.0
    active = is_active(args.follow_service)
    with (args.baseline / "index.jsonl").open() as upstream, index.open("a") as sink, \
            ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                initargs=(execution_id,)) as pool:
        futures = {}
        while len(done) < len(source):
            now = time.monotonic()
            if now - last_active_check > 15:
                active = is_active(args.follow_service)
                last_active_check = now
            for row in read_complete_lines(upstream):
                if row["id"] in expected_ids and row["id"] not in submitted:
                    expected = source_by_id[row["id"]]
                    for key in ["source_motion_id", "capture_group", "original_parent", "is_mirror", "split"]:
                        if row[key] != expected[key]:
                            raise ValueError(f"Baseline/source lineage mismatch: {key}")
                    pending.append(row)
                    submitted.add(row["id"])
            capacity = args.busy_workers if active and args.busy_workers else args.workers
            while pending and len(futures) < capacity:
                row = pending.popleft()
                future = pool.submit(recover, (row, str(args.dataset_root), str(args.baseline), str(args.output)))
                futures[future] = row["id"]
            completed, _ = wait(futures, timeout=2.0, return_when=FIRST_COMPLETED) if futures else (set(), set())
            for future in completed:
                row = future.result()  # Operational/programming errors stop the campaign.
                if row["id"] != futures.pop(future):
                    raise ValueError("Worker result identity changed")
                sink.write(json.dumps(row, allow_nan=False) + "\n")
                sink.flush()
                done[row["id"]] = row
            if now - last_report > 30:
                report(done, len(source), contract, args.output, True, execution_id)
                last_report = now
            if not futures and not pending and len(done) < len(source):
                if not active:
                    report(done, len(source), contract, args.output, False, execution_id)
                    raise RuntimeError(f"Baseline stopped with {len(source) - len(done)} assigned records unavailable")
                time.sleep(2)
    result = report(done, len(source), contract, args.output, False, execution_id)
    if not (result["all_categories_pass"] and result["all_controller_families_pass"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
