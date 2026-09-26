#!/usr/bin/env python3
"""Correct imported semantic labels without changing any reference samples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
VERSION = "whole-body-source-label-correction-v1"


def corrected_labels(row):
    from k1_motion.motion_coverage import movement_tags
    tags = movement_tags(row)
    family = row["family"]
    # The historical classifier matched "strike", including striking a pose.
    # Only correct clear errors; generic fighting intent remains a broad label.
    if family == "punch" and "boxing_striking" not in tags:
        if "dance" in tags:
            family = "dance"
        elif "punch_self_" in row.get("take_name", ""):
            family = "gesture"
    return family, tags


def curate_row(row, output, revision):
    from k1_motion.contracts import MotionClip
    from k1_motion.reference_admission import reference_rejections
    family, tags = corrected_labels(row)
    if family == row["family"] and tags == row.get("movement_tags"):
        return row
    if (not row.get("kinematics_accepted") or not row.get("recovery_audit", {}).get("accepted")
            or reference_rejections(row) or row.get("rl_reference_audit")
            or row.get("low_pose_reference_audit")):
        raise ValueError("Label correction is restricted to strict-audited new references")
    clip = MotionClip.load(row["reference_path"])
    for key in ("id", "family", "capture_group", "split", "model_signature", "source_motion_id"):
        if clip.metadata[key] != row[key]:
            raise ValueError("Source payload/manifest mismatch")
    if not clip.values["valid"].all():
        raise ValueError("Invalid source samples")
    path = output / "clips" / (row["id"] + ".npz")
    if path.exists():
        raise ValueError("Refusing to overwrite a curated reference")
    changes = {"family": family, "movement_tags": tags,
               "source_family": row.get("source_family", row["family"]),
               "reference_path": str(path.resolve()),
               "label_curation": {"version": VERSION, "source_revision": revision,
                                  "parent_reference_path": row["reference_path"],
                                  "samples_and_clocks_unchanged": True}}
    updated = {**row, **changes}
    if reference_rejections(updated):
        raise ValueError("Label correction changed admission")
    clip.metadata.update(changes)
    clip.save(path)
    saved = MotionClip.load(path)
    for key, value in clip.values.items():
        np.testing.assert_array_equal(value, saved.values[key])
    np.testing.assert_array_equal(clip.times, saved.times)
    np.testing.assert_array_equal(clip.source_times, saved.source_times)
    return updated


def main():
    from freeze_source import freeze_source
    from prepare_broad_references import atomic_json, rows
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--base-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = list(rows(args.library / "index.jsonl"))
    base = list(rows(args.base_library / "index.jsonl"))
    if pool[:len(base)] != base:
        raise ValueError("Prior pool must remain unchanged")
    if len({r["id"] for r in pool}) != len(pool):
        raise ValueError("Duplicate input identities")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "clips").mkdir()
    (args.output / "pool").mkdir()
    snapshot, revision = freeze_source(ROOT)
    contract = {"version": VERSION, "source_revision": revision, "source_snapshot": str(snapshot.resolve()),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "parent_manifest": str((args.library / "index.jsonl").resolve()),
                "parent_manifest_sha256": hashlib.sha256((args.library / "index.jsonl").read_bytes()).hexdigest(),
                "preserved_base_manifest": str((args.base_library / "index.jsonl").resolve())}
    atomic_json(args.output / "campaign.json", contract)
    corrected = base + [curate_row(r, args.output, revision) for r in pool[len(base):]]
    index = args.output / "pool/index.jsonl"
    content = "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in corrected)
    index.write_text(content)
    changes = [{"id": a["id"], "split": a["split"], "source_family": a["family"],
                "family": b["family"], "tags_changed": a.get("movement_tags") != b.get("movement_tags")}
               for a, b in zip(pool, corrected) if a != b]
    report = {"complete": True, "base_originals_preserved": len(base),
              "all_originals_preserved": len(corrected), "samples_and_clocks_unchanged": True,
              "updated_metadata_originals": len(changes), "changes": changes,
              "family_changes": dict(Counter(a["family"] + " -> " + b["family"]
                  for a, b in zip(pool, corrected) if a["family"] != b["family"])),
              "pool_splits": dict(Counter(r["split"] for r in corrected)),
              "train_families": dict(Counter(r["family"] for r in corrected if r["split"] == "train")),
              "manifest_sha256": hashlib.sha256(content.encode()).hexdigest(),
              "physics_qualified": False, "controller_success_used_for_selection": False}
    atomic_json(args.output / "summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "changes"}, indent=2))


if __name__ == "__main__":
    main()
