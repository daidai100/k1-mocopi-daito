#!/usr/bin/env python3
"""Original-only rejection census and causal first-pose calibration diagnostics.

No reference admission or conversion is changed. Remote input is ledger-only;
the small, take-disjoint pose panel reads the already extracted local BVHs.
"""
# ruff: noqa: E402
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from k1_motion.adapters import BVH_BASIS, BVH_MAPS, Bvh
from k1_motion.contracts import MotionClip
from k1_motion.recovery_geometry import geometry_forward, recovery_model
from k1_motion.reference_admission import reference_rejections, take_family
from k1_motion.robot import K1Model

FAMILIES = {"turn", "squat", "crawl", "sit_or_kneel", "transition", "run", "idle_stance", "dance"}


def description(row):
    return " ".join([row.get("take_name", ""), *row.get("annotations", [])]).lower()


def support_category(row):
    if row["family"] != "sit_or_kneel":
        return row["family"]
    text = description(row)
    # "Hands on knees" is common in sitting descriptions, not evidence of
    # kneeling. Use explicit kneel labels and report this as an annotation subset.
    kneel = "kneel" in text
    furniture = bool(re.search(r"chair|bench|sofa|stool|couch|\bseat\b", text))
    if kneel:
        return "kneeling_with_furniture" if furniture else "kneeling_no_furniture_mentioned"
    return "seated_furniture" if furniture else "floor_sitting_or_unspecified"


def summarize(rows):
    reasons, source_errors, tick_reasons, pairs = Counter(), Counter(), Counter(), Counter()
    ground_only = []
    for row in rows:
        if row["kinematics_accepted"]:
            continue
        audit = row.get("recovery_audit", {})
        reject = audit.get("rejection_reasons", ["source_error"])
        reasons.update(reject)
        tick_reasons.update(row.get("rejection_counts", {}).keys())
        pairs.update(audit.get("collision_pairs", {}).keys())
        if reject == ["ground_penetration_on_command_path"]:
            ground_only.append(row)
        if not audit:
            source_errors.update([row.get("error", "unspecified")])
    depths = [r["recovery_audit"]["max_ground_penetration_m"] for r in ground_only]
    return {
        "originals": len(rows), "v4_strict_accepted": sum(r["kinematics_accepted"] for r in rows),
        "rejected": sum(not r["kinematics_accepted"] for r in rows),
        "reasons_clip_counts_overlapping": dict(reasons),
        "invalid_tick_reason_clip_counts_overlapping": dict(tick_reasons),
        "self_collision_pair_clip_counts": dict(pairs), "source_errors": dict(source_errors),
        "ground_only": len(ground_only),
        "ground_only_max_depth_median_mm": float(np.median(depths) * 1000) if depths else None,
        "ground_only_depth_at_most_50mm": sum(d <= .05 for d in depths),
        "external_support_description_count": sum(
            "external_support_not_configured" in reference_rejections(r) for r in rows),
        "capture_groups": len({r["capture_group"] for r in rows}),
        "related_take_families": len({take_family(r["capture_group"]) for r in rows}),
    }


def inspect_first_pose(job):
    row, dataset = job
    bvh = Bvh.load(Path(dataset) / row["source_path"])
    positions, _ = bvh.fk(translation_mode="add", frame_indices=[0])
    ids = [bvh.names.index(name) for name in BVH_MAPS["bones_seed"]]
    p = BVH_BASIS.apply(positions[0, ids] * .01)
    robot = K1Model()
    def legs(points):
        return np.array([np.linalg.norm(points[a] - points[b]) + np.linalg.norm(points[b] - points[c])
                         for a, b, c in ((9, 10, 11), (13, 14, 15))])
    human_legs, robot_legs = legs(p), legs(robot.neutral_landmarks)
    toe_floor = float(min(p[12, 2], p[16, 2]) - .02)
    height = float(p[0, 2] - toe_floor)
    limb_scale = float(robot_legs.mean() / human_legs.mean())
    return {
        "id": row["id"], "family": row["family"], "category": support_category(row),
        "capture_group": row["capture_group"], "take_family": take_family(row["capture_group"]),
        "annotations": row.get("annotations", []), "source_error": row.get("error"),
        "source_frames": len(bvh.values), "inspected_frame_indices": [0],
        "toe_derived_initial_root_height_m": height,
        "human_leg_lengths_m": human_legs.tolist(), "robot_leg_lengths_m": robot_legs.tolist(),
        "hypothetical_neutral_height_scale": float(robot.neutral_qpos[2] / height) if height else None,
        "pose_independent_leg_length_scale": limb_scale,
        "toe_derived_root_height_after_limb_scaling_m": height * limb_scale,
        "below_current_022m_root_gate_after_limb_scaling": height * limb_scale < .22,
        "minimum_wrist_relative_to_toe_floor_m": float(min(p[5, 2], p[8, 2]) - toe_floor),
        "minimum_knee_relative_to_toe_floor_m": float(min(p[10, 2], p[14, 2]) - toe_floor),
        "floor_estimate_validated": False, "conversion_or_admission_changed": False,
    }


def inspect_ground_bodies(job):
    row, source = job
    clip = MotionClip.load(Path(source) / row["attempt_reference_path"])
    robot = recovery_model("audit")
    model, data = robot.model, robot.data
    counts, depths = Counter(), {}
    for p, q, joints in zip(clip.values["root_position"], clip.values["root_orientation"],
                            clip.values["joint_position"]):
        data.qpos[:] = np.r_[p, q, joints]
        geometry_forward(model, data)
        bodies = set()
        for contact in data.contact[:data.ncon]:
            ids = model.geom_bodyid[[contact.geom1, contact.geom2]]
            if 0 not in ids or contact.dist >= -.005:
                continue
            body = model.body(int(max(ids))).name
            bodies.add(body)
            depths[body] = max(depths.get(body, 0), -float(contact.dist))
        counts.update(bodies)
    return {"id": row["id"], "category": support_category(row), "capture_group": row["capture_group"],
            "annotations": row.get("annotations", []), "diagnostic_hz": 50,
            "frames": len(clip.times), "ground_body_frame_counts_over_5mm": dict(counts),
            "ground_body_max_depths_m": depths,
            "root_height_min_m": float(clip.values["root_position"][:, 2].min()),
            "root_frames_below_022m": int((clip.values["root_position"][:, 2] < .22).sum()),
            "v4_rejection_reasons": row.get("recovery_audit", {}).get("rejection_reasons"),
            "stance_slip_p95_m_s": row.get("recovery_audit", {}).get("stance_slip_p95_m_s"),
            "conversion_or_admission_changed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-source")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poses-per-category", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new diagnostic output")
    if bool(args.remote_host) != bool(args.remote_source):
        raise ValueError("Both remote arguments are required")
    rows, sources, local_rows = [], [], []
    for source in args.source:
        text = (source / "index.jsonl").read_text()
        selected_rows = [r for line in text.splitlines() if not (r := json.loads(line))["is_mirror"]
                         and r["family"] in FAMILIES]
        rows.extend(selected_rows)
        local_rows.extend((r, str(source)) for r in selected_rows)
        sources.append({"source": str(source.resolve()),
                        "ledger_sha256": hashlib.sha256(text.encode()).hexdigest()})
    if args.remote_host:
        program = (
            "import json,sys;from pathlib import Path;p=Path(sys.argv[1]);"
            "s=json.loads((p/'summary.json').read_text());assert s['complete'] and not s['running'];"
            "[print(json.dumps(r)) for l in (p/'index.jsonl').open() "
            "if not (r:=json.loads(l))['is_mirror'] and r['family'] in json.loads(sys.argv[2])]"
        )
        remote = subprocess.run(["ssh", "-6", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                                 args.remote_host, shlex.join(["python3", "-c", program,
                                 args.remote_source, json.dumps(sorted(FAMILIES))])],
                                check=True, capture_output=True, text=True).stdout
        rows.extend(json.loads(line) for line in remote.splitlines())
        sources.append({"host": args.remote_host, "source": args.remote_source,
                        "selected_ledger_sha256": hashlib.sha256(remote.encode()).hexdigest()})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Overlapping original ledgers")
    selected, counts, used = [], Counter(), set()
    for row in sorted(rows, key=lambda r: r["id"]):
        category = support_category(row)
        group = take_family(row["capture_group"])
        if ("upright neutral" not in row.get("error", "") or category not in (
                "squat", "crawl", "kneeling_no_furniture_mentioned") or group in used
                or counts[category] >= args.poses_per_category):
            continue
        used.add(group)
        counts[category] += 1
        selected.append((row, str(args.dataset_root)))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        poses = list(executor.map(inspect_first_pose, selected))
        body_selected, body_counts, body_groups = [], Counter(), set()
        for row, source in sorted(local_rows, key=lambda job: job[0]["id"]):
            category, group = support_category(row), take_family(row["capture_group"])
            if (category not in ("squat", "crawl", "kneeling_no_furniture_mentioned", "turn")
                    or group in body_groups or body_counts[category] >= 4
                    or "ground_penetration_on_command_path" not in row.get("recovery_audit", {})
                    .get("rejection_reasons", [])):
                continue
            body_groups.add(group)
            body_counts[category] += 1
            body_selected.append((row, source))
        body_panel = list(executor.map(inspect_ground_bodies, body_selected))
    report = {
        "version": "grounded-and-low-support-rejection-census-v1", "sources": sources,
        "originals": len(rows), "mirrors_counted": False, "climbing_in_scope": False,
        "families": {family: summarize([r for r in rows if r["family"] == family])
                     for family in sorted(FAMILIES)},
        "sit_or_kneel_annotation_subsets": {
            category: summarize([r for r in rows if support_category(r) == category])
            for category in sorted({support_category(r) for r in rows if r["family"] == "sit_or_kneel"})},
        "first_pose_panel": poses,
        "pose_panel_selection": f"sorted original ID; up to {args.poses_per_category} distinct take families per category; calibration errors only",
        "kneeling_subset_definition": "Explicit kneel substring in take name or annotations; not generic hands-on-knees sitting",
        "ground_body_panel": body_panel,
        "ground_body_panel_selection": "sorted local-shard original ID; up to four distinct take families per category; ground failure",
        "physics_or_controller_success_measured": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "originals": len(rows), "pose_panel": len(poses),
                      "families": report["families"],
                      "sit_or_kneel_annotation_subsets": report["sit_or_kneel_annotation_subsets"]}, indent=2))


if __name__ == "__main__":
    main()
