#!/usr/bin/env python3
"""Paired whole-recording retarget speed diagnostic; strict 500 Hz gates unchanged."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(os.environ.get("K1_MOTION_ROOT", Path(__file__).resolve().parents[1]))
if os.environ.get("K1_FROZEN_SOURCE"):
    sys.path.insert(0, str(Path(os.environ["K1_FROZEN_SOURCE"]).parent))
else:
    sys.path.insert(0, str(ROOT / "src"))
PROFILES = ("legacy-command-v1", "official-80-v1")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def convert_pair(job):
    from k1_motion.contracts import MotionClip
    from k1_motion.recovery_geometry import recovery_model
    from k1_motion.recovery_validation import retarget_recovery, audit_recovery
    from k1_motion.ground_reference import correct_walking_ground
    from k1_motion.reference_quality import audit_reference_consistency
    from k1_motion.math3d import rotation

    row, output = job
    output = Path(output)
    if row["is_mirror"] or row["split"] != "train":
        raise ValueError("Only training originals may enter this experiment")
    payload = Path(row["human_path"]).read_bytes()
    if row.get("human_sha256") and hashlib.sha256(payload).hexdigest() != row["human_sha256"]:
        raise ValueError("Canonical human payload changed")
    with np.load(row["human_path"]) as data:
        human = {k: data[k].copy() for k in ("times", "positions", "orientations")}
    robot = recovery_model("retarget")
    clips, reports, corrections, errors = {}, {}, {}, {}
    for profile in PROFILES:
        candidate, reports[profile] = retarget_recovery(
            robot, human, row, speed_profile=profile, control_tick_hold=True
        )
        candidate, corrections[profile] = correct_walking_ground(candidate)
        # Ground correction updates root positions: reconstruct the exact causal velocity path.
        from k1_motion.reference_velocity_clock import playback_derivatives

        dq, rv = playback_derivatives(candidate)
        candidate = MotionClip(
            candidate.times,
            {**candidate.values, "joint_velocity": dq, "root_velocity": rv},
            candidate.metadata,
            candidate.source_times,
            candidate.received_times,
        )
        destination = output / profile / (row["id"] + ".npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        candidate.save(destination)
        clips[profile] = MotionClip.load(destination)
        errors[profile] = (
            float(np.mean([r["rms_landmark_error_m"] for r in reports[profile]]))
            + corrections[profile]["mean_landmark_error_increase_bound_m"]
        )
    result = dict(
        id=row["id"],
        source_motion_id=row["source_motion_id"],
        capture_group=row["capture_group"],
        profiles={},
        whole_recording=True,
        physics_qualified=False,
        controller_success_used=False,
    )
    for profile in PROFILES:
        candidate = clips[profile]
        audit = audit_recovery(
            candidate,
            clips[PROFILES[0]],
            [{"rms_landmark_error_m": errors[profile]}],
            {"rms_landmark_error_m": errors[PROFILES[0]]},
            ground_profile=True,
            speed_profile=profile,
        )
        invalid = int(np.sum(~candidate.values["valid"].astype(bool)))
        reasons = list(audit["rejection_reasons"])
        if invalid:
            reasons.append("invalid_reference_ticks")
        if (
            corrections[profile]["nonfoot_ground_contact_samples"]
            or audit["ground_profile"]["max_nonfoot_penetration_m"] > 0.0001
        ):
            reasons.append("nonfoot_support_task_requires_separate_audit")
        upright = rotation(candidate.values["root_orientation"]).apply([0, 0, 1])[:, 2]
        if candidate.values["root_position"][:, 2].min() < 0.22 or upright.min() < 0.2:
            reasons.append("low_support_task_requires_separate_audit")
        if np.any(candidate.values["joint_position"] < robot.limits[:, 0] - 1e-8) or np.any(
            candidate.values["joint_position"] > robot.limits[:, 1] + 1e-8
        ):
            reasons.append("joint_limits")
        consistency = audit_reference_consistency(candidate, recovery_model("audit"))
        if not consistency["contact_consistent"]:
            reasons.append("reference_contact_or_clock_inconsistency")
        speed = np.abs(np.diff(candidate.values["joint_position"], axis=0) / robot.control_dt)
        item = dict(
            profile=profile,
            recovery_audit=audit,
            reference_consistency=consistency,
            ground_correction=corrections[profile],
            rejected_ticks=invalid,
            rejection_reasons=sorted(set(reasons)),
            kinematics_accepted=not reasons,
            physics_qualified=False,
            training_eligible=False,
            frames=len(candidate.times),
            duration_s=float(candidate.times[-1] - candidate.times[0]),
            max_joint_speed_rad_s=float(speed.max()),
            fraction_leg_ticks_at_legacy_cap=float(np.mean(np.any(speed[:, 10:] >= 6.0 - 1e-6, axis=-1))),
            attempt_reference_path=str((output / profile / (row["id"] + ".npz")).resolve()),
        )
        candidate.metadata.update(item)
        candidate.save(item["attempt_reference_path"])
        result["profiles"][profile] = item
    result["paired_strict_accepted"] = all(v["kinematics_accepted"] for v in result["profiles"].values())
    write(output / (row["id"] + ".json"), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    rows = json.loads(args.manifest.read_text())
    if not rows or len({r["id"] for r in rows}) != len(rows) or args.workers < 1:
        raise ValueError("Unique nonempty input manifest and positive workers required")
    output = args.output.resolve()
    from freeze_source import freeze_source

    snapshot, revision = freeze_source(ROOT, os.environ.get("K1_FROZEN_SOURCE"))
    contract = dict(
        version="paired-retarget-speed-20260926-v1",
        source_revision=revision,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        profiles=list(PROFILES),
        originals=len(rows),
        mirrors=0,
        workers=args.workers,
        strict_gates_unchanged=True,
        source_clock_unchanged=True,
        physics_qualified=False,
        training_admitted=0,
    )
    if (output / "campaign.json").exists() and json.loads((output / "campaign.json").read_text()) != contract:
        raise ValueError("Retarget campaign changed; use a new output")
    write(output / "campaign.json", contract)
    start = time.monotonic()
    results = []
    pending = []
    for row in rows:
        path = output / (row["id"] + ".json")
        if path.exists():
            results.append(json.loads(path.read_text()))
        else:
            pending.append((row, str(output)))
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(convert_pair, pending):
                results.append(result)
                write(
                    output / "status.json",
                    dict(
                        phase="converting",
                        processed=len(results),
                        originals=len(rows),
                        elapsed_s=time.monotonic() - start,
                    ),
                )
        summary = {
            **contract,
            "phase": "complete",
            "processed_originals": len(results),
            "paired_strict_accepted": sum(r["paired_strict_accepted"] for r in results),
            "profiles": {
                profile: dict(
                    strict_accepted=sum(r["profiles"][profile]["kinematics_accepted"] for r in results),
                    rejects=dict(
                        Counter(
                            reason for r in results for reason in r["profiles"][profile]["rejection_reasons"]
                        )
                    ),
                )
                for profile in PROFILES
            },
            "elapsed_s": time.monotonic() - start,
        }
        write(output / "summary.json", summary)
        write(output / "status.json", summary)
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        write(output / "status.json", dict(phase="failed", error=f"{type(error).__name__}: {error}"))
        raise


if __name__ == "__main__":
    main()
