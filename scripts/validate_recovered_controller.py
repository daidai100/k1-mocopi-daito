#!/usr/bin/env python3
"""Follow audited recovery outputs with versioned, resumable controller replays."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from freeze_source import freeze_source

ROOT = Path(__file__).resolve().parents[1]
FROZEN, REVISION = freeze_source(ROOT, os.environ.get("K1_FROZEN_SOURCE"))
os.environ["K1_MOTION_ROOT"] = str(ROOT)
sys.path.insert(0, str(FROZEN))

from k1_motion.contracts import MotionClip  # noqa: E402
from k1_motion.control_validation import CONTROL_THRESHOLDS, MEASUREMENT_VERSION, replay_clip  # noqa: E402
from k1_motion.learning import Policy  # noqa: E402
from k1_motion.robot import K1Model  # noqa: E402

ROBOT = POLICY = None


def initialize(policy):
    global ROBOT, POLICY
    import torch

    torch.set_num_threads(1)
    ROBOT = K1Model()
    POLICY = Policy(policy, ROBOT.signature)


def one(job):
    row, source, output = job
    clip = MotionClip.load(Path(source) / row["reference_path"])
    for key in ("source_motion_id", "capture_group", "is_mirror", "split"):
        if clip.metadata[key] != row[key]:
            raise ValueError(f"Recovery payload/ledger lineage mismatch: {key}")
    result = replay_clip(ROBOT, POLICY, clip, Path(output) / "traces" / (row["id"] + ".npz"))
    return {
        **row,
        "controller_replay": result,
        "controller_replay_passed": result["clean_success"],
        "controller_audit_status": "replayed",
        "physics_qualified": False,
        "training_eligible": False,
        "scope": "One controller-specific native replay; family robustness and hardware qualification remain separate",
    }


def atomic_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def active(service):
    result = subprocess.run(["systemctl", "--user", "is-active", service], capture_output=True, text=True)
    return result.stdout.strip() in ("active", "activating", "reloading")


def summary(rows, contract, output, running):
    trials = [r for r in rows if r.get("controller_audit_status") == "replayed"]
    stats = {
        "replayed_originals": len(trials),
        "completed_originals": sum(r["controller_replay"]["completed"] for r in trials),
        "clean_originals": sum(r["controller_replay_passed"] for r in trials),
        "collision_originals": sum(r["controller_replay"]["self_collision_ticks"] > 0 for r in trials),
        "fall_originals": sum(r["controller_replay"]["fell"] for r in trials),
        "clean_replay_hours": sum(
            r["controller_replay"]["duration_s"] for r in trials if r["controller_replay_passed"]
        )
        / 3600,
        "status_counts": dict(Counter(r["controller_audit_status"] for r in rows)),
        "execution_errors": 0,
        "running": running,
        "updated_unix": time.time(),
        "contract": contract,
        "behaviorally_accepted": False,
        "hardware_verified": False,
    }
    atomic_json(output / "summary.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--follow-service")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Bound the number of original dynamic replays for a canary")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Worker and canary limits must be positive")
    args.source, args.output, args.policy = (p.resolve() for p in (args.source, args.output, args.policy))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces").mkdir(exist_ok=True)
    contract = {
        "version": MEASUREMENT_VERSION,
        "source_revision": REVISION,
        "source": str(args.source),
        "policy": str(args.policy),
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "policy_metadata_sha256": hashlib.sha256(args.policy.with_suffix(".json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "recovery_contract": json.loads((args.source / "campaign.json").read_text()),
        "thresholds": CONTROL_THRESHOLDS,
        "canary_limit": args.limit,
        "mirror_policy": "deferred; originals are the independent denominator",
    }
    path = args.output / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Controller audit contract changed; choose a new output version")
    atomic_json(path, contract)
    ledger = args.output / "index.jsonl"
    done = []
    if ledger.exists():
        with ledger.open("rb+") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    stream.truncate(offset)
                    break
                done.append(json.loads(line))
    ids = {r["id"] for r in done}
    submitted = sum(r["controller_audit_status"] == "replayed" for r in done)
    futures = {}
    last_report = 0.0
    with (
        ledger.open("a") as sink,
        (args.source / "index.jsonl").open() as source,
        ProcessPoolExecutor(
            max_workers=args.workers, initializer=initialize, initargs=(str(args.policy),)
        ) as pool,
    ):

        def save(row):
            done.append(row)
            ids.add(row["id"])
            sink.write(json.dumps(row) + "\n")
            sink.flush()

        while True:
            eof = False
            while len(futures) < args.workers * 2 and (args.limit is None or submitted < args.limit):
                offset = source.tell()
                line = source.readline()
                if not line or not line.endswith("\n"):
                    source.seek(offset)
                    eof = True
                    break
                row = json.loads(line)
                if row["id"] in ids:
                    continue
                if row["is_mirror"] or not row["kinematics_accepted"]:
                    status = "mirror_deferred" if row["is_mirror"] else "reference_rejected"
                    save({**row, "controller_audit_status": status, "controller_replay_passed": False})
                else:
                    if not row.get("recovery_audit", {}).get("accepted"):
                        raise ValueError("Accepted reference lacks independent recovery audit")
                    futures[pool.submit(one, (row, str(args.source), str(args.output)))] = row["id"]
                    ids.add(row["id"])
                    submitted += 1
            if futures:
                completed, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    try:
                        row = future.result()
                    except Exception as error:
                        stats = summary(done, contract, args.output, False)
                        stats.update(execution_errors=1, fatal_error=f"{type(error).__name__}: {error}")
                        atomic_json(args.output / "summary.json", stats)
                        raise
                    save(row)
                    del futures[future]
            if time.monotonic() - last_report > 15:
                stats = summary(done, contract, args.output, True)
                print(json.dumps({k: v for k, v in stats.items() if k != "contract"}), flush=True)
                last_report = time.monotonic()
            if not futures:
                if args.limit is not None and submitted >= args.limit:
                    break
                if eof and (not args.follow_service or not active(args.follow_service)):
                    expected = contract["recovery_contract"]["expected_rows"]
                    if args.follow_service and len(done) != expected:
                        stats = summary(done, contract, args.output, False)
                        stats.update(
                            execution_errors=1,
                            fatal_error="Upstream recovery stopped before all assigned records arrived",
                        )
                        atomic_json(args.output / "summary.json", stats)
                        raise RuntimeError(stats["fatal_error"])
                    break
                time.sleep(2)
    summary(done, contract, args.output, False)


if __name__ == "__main__":
    main()
