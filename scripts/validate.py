#!/usr/bin/env python3
"""Inventory original human motion tracks; count each representation once."""
import argparse
from collections import Counter
import concurrent.futures
import datetime as dt
import io
import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = {"lafan1": 77, "bandai_namco": 3077, "kit_motion_language": 3911}


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def vector(text, size):
    if text is None:
        raise ValueError("Missing numeric field")
    data = np.fromstring(text, sep=" ")
    if data.size != size or not np.isfinite(data).all():
        raise ValueError(f"Expected {size} finite numbers, found {data.size}")
    return data


def bvh(path, dataset):
    text = path.read_text(encoding="utf-8-sig")
    header, body = re.split(r"(?m)^\s*MOTION\s*$", text, maxsplit=1)
    frames_match = re.search(r"Frames:\s*(\d+)", body)
    time_match = re.search(r"Frame Time:\s*([\d.eE+-]+)", body)
    if not frames_match or not time_match:
        raise ValueError("Missing BVH timing")
    count = int(frames_match.group(1))
    frame_time = float(time_match.group(1))
    if count < 2 or not np.isfinite(frame_time) or frame_time <= 0:
        raise ValueError("Invalid BVH timing")
    channels = sum(int(x) for x in re.findall(r"CHANNELS\s+(\d+)", header))
    joints = re.findall(r"(?:ROOT|JOINT)\s+(\S+)", header)
    if not channels or not joints or header.count("{") != header.count("}"):
        raise ValueError("Invalid BVH skeleton hierarchy")
    for offset in re.findall(r"OFFSET\s+([^\r\n]+)", header):
        vector(offset, 3)
    numeric = np.loadtxt(io.StringIO(body[time_match.end():]), ndmin=2)
    if numeric.shape != (count, channels) or not np.isfinite(numeric).all():
        raise ValueError(f"BVH shape/nonfinite mismatch: {numeric.shape}, expected {(count, channels)}")
    if dataset == "lafan1":
        take, subject = path.stem.rsplit("_", 1)
        category = re.sub(r"\d+$", "", take)
        group = f"lafan1/{take}"
        annotations = [category]
        subset = "LAFAN1"
    else:
        subset, category, style, clip = path.stem.split("_", 3)
        annotations = [category, style]
        subject = None  # These releases do not identify the actor per clip.
        group = f"bandai_namco/{subset}/{category}/{style}"
    sidecar = path.with_suffix(".json")
    return [{
        "dataset": dataset, "subset": subset, "path": str(path.relative_to(ROOT)),
        "format": "BVH", "source_motion_id": f"{dataset}/{path.stem}",
        "subject_id": subject, "capture_group": group, "annotations": annotations,
        "upstream_annotation": json.loads(sidecar.read_text()) if sidecar.exists() else None,
        "frames": count, "fps": 1.0 / frame_time, "frame_time_seconds": frame_time,
        "duration_seconds": count * frame_time, "joints": len(joints), "channels": channels,
        "root_joint": joints[0], "numerical_validation": "passed",
        "units": "source BVH units retained; confirm scale and axes before retargeting"
    }]


def mmm(path, dataset):
    tree = ET.parse(path)
    prefix = path.name.removesuffix("_mmm.xml")
    metadata = json.loads(path.with_name(prefix + "_meta.json").read_text())
    annotations = json.loads(path.with_name(prefix + "_annotations.json").read_text())
    raw = path.with_name(prefix + "_raw.c3d")
    if not raw.is_file() or raw.stat().st_size < 512:
        raise ValueError("Missing or undersized original C3D recording")
    source = metadata.get("source", {})
    origin = source.get("institution", {}).get("identifier", "unknown")
    database = source.get("database", {})
    records = []
    for track, motion in enumerate(tree.getroot().findall("Motion")):
        order = motion.find("JointOrder")
        if order is None:
            continue  # Environmental object tracks are preserved, not human hours.
        joints = [joint.attrib["name"] for joint in order.findall("Joint")]
        if not joints:
            continue
        frames = motion.findall("MotionFrames/MotionFrame")
        timestamps = []
        for frame in frames:
            timestamps.append(float(frame.findtext("Timestep")))
            vector(frame.findtext("RootPosition"), 3)
            vector(frame.findtext("RootRotation"), 3)
            vector(frame.findtext("JointPosition"), len(joints))
            for optional in ("JointVelocity", "JointAcceleration"):
                field = frame.findtext(optional)
                if field is not None:
                    vector(field, len(joints))
        timestamps = np.asarray(timestamps)
        increments = np.diff(timestamps)
        if len(frames) < 2 or not np.isfinite(timestamps).all() or np.any(increments <= 0):
            raise ValueError("Invalid or nonmonotonic MMM timestamps")
        step = float(np.median(increments))
        source_id = f"{origin}/{database.get('identifier', 'unknown')}/{database.get('motion_id', prefix)}/{database.get('motion_file_id', prefix)}/{track}"
        records.append({
            "dataset": dataset, "subset": origin, "path": str(path.relative_to(ROOT)),
            "format": "MMM", "track_index": track, "subject_id": motion.get("name"),
            "source_motion_id": source_id,
            "capture_group": f"{origin}/{database.get('identifier', 'unknown')}/{database.get('motion_id', prefix)}",
            "source_metadata": source, "annotations": annotations,
            "frames": len(frames), "fps": 1.0 / step, "frame_time_seconds": step,
            "duration_seconds": float(timestamps[-1] - timestamps[0] + step),
            "sample_interval_min": float(increments.min()), "sample_interval_max": float(increments.max()),
            "joints": len(joints), "numerical_validation": "passed",
            "raw_c3d": str(raw.relative_to(ROOT)), "raw_c3d_validation": "archive CRC and presence; marker reconstruction not evaluated",
            "units": "MMM source units retained (root position in mm, rotation/joints in radians)",
            "model_reference": motion.findtext("Model/File"),
            "height_m": motion.findtext("ModelProcessorConfig/Height")
        })
    if not records:
        raise ValueError("No human joint track found")
    return records


def inspect_one(args):
    path, dataset = args
    try:
        return {"records": (mmm(path, dataset) if path.name.endswith("_mmm.xml") else bvh(path, dataset)), "error": None}
    except Exception as error:
        return {"records": [], "error": {"path": str(path.relative_to(ROOT)), "error": repr(error)}}


def inspect_dataset(dataset):
    directory = ROOT / "data/raw" / dataset
    paths = sorted(directory.rglob("*_mmm.xml" if dataset == "kit_motion_language" else "*.bvh"))
    records, errors = [], []
    print(f"{dt.datetime.now().isoformat()} validating {dataset}: {len(paths)} human motion files", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as workers:
        for index, result in enumerate(workers.map(inspect_one, ((path, dataset) for path in paths), chunksize=8), 1):
            records.extend(result["records"])
            if result["error"]:
                errors.append(result["error"])
            if index % 500 == 0:
                print(f"{dataset}: {index}/{len(paths)} inspected", flush=True)
    seen = set()
    for record in records:
        record["duplicate_source_id"] = record["source_motion_id"] in seen
        record["counted_in_duration"] = not record["duplicate_source_id"]
        seen.add(record["source_motion_id"])
    counted = [r for r in records if r["counted_in_duration"]]
    inventory = ROOT / "manifests" / (dataset + ".motions.jsonl")
    with inventory.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "dataset": dataset, "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "motion_files": len(paths), "expected_motion_files": EXPECTED_FILES[dataset],
        "file_count_matches_upstream": len(paths) == EXPECTED_FILES[dataset],
        "valid_human_tracks": len(records), "unique_source_tracks": len(counted),
        "frames": sum(r["frames"] for r in counted),
        "hours": sum(r["duration_seconds"] for r in counted) / 3600,
        "fps_values": dict(Counter(round(r["fps"], 4) for r in records)),
        "subsets": dict(Counter(r["subset"] for r in records)),
        "failed_files": len(errors), "errors": errors,
        "inventory": str(inventory.relative_to(ROOT)),
        "validation_scope": "Finite values, dimensions, skeleton headers, timing, annotations and provenance; physical retargeting and control not yet evaluated"
    }
    save_json(ROOT / "manifests" / (dataset + ".validation.json"), summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-downloads", action="store_true")
    args = parser.parse_args()
    pending = set(EXPECTED_FILES)
    summaries = {}
    while pending:
        for dataset in sorted(pending):
            if not (ROOT / "manifests" / (dataset + ".download.json")).exists():
                continue
            cached = ROOT / "manifests" / (dataset + ".validation.json")
            summaries[dataset] = json.loads(cached.read_text()) if cached.exists() else inspect_dataset(dataset)
            pending.remove(dataset)
        failures = sum(item["failed_files"] for item in summaries.values())
        counted_hours = sum(item["hours"] for item in summaries.values())
        complete = not pending and not failures and all(item["file_count_matches_upstream"] for item in summaries.values())
        report = {"updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "download_and_numerical_validation_complete": complete,
                  "target_hours": 20, "validated_human_motion_hours": counted_hours,
                  "approximately_20_hours_acquired": complete and counted_hours >= 19,
                  "valid_human_tracks": sum(item["unique_source_tracks"] for item in summaries.values()),
                  "pending_datasets": sorted(pending), "failed_files": failures,
                  "datasets": summaries,
                  "duration_basis": "Sum of original human track frame durations; C3D/MMM representations counted once; no augmentation or mirrored copies added. Distinct people recorded simultaneously are separate human tracks, not independent studio sessions.",
                  "retargeted_to_k1": False, "controller_trained": False,
                  "training_readiness": "Raw human source data only. Skeleton/axis normalization, contact/environment screening, split design, retargeting and physics validation are still required."}
        save_json(ROOT / "manifests/corpus-summary.json", report)
        if not pending or not args.wait_for_downloads:
            break
        download_status = ROOT / "manifests/download-status.json"
        if download_status.exists() and json.loads(download_status.read_text()).get("state") == "failed":
            raise SystemExit("Acquisition failed; see download-status.json")
        time.sleep(20)
    if not pending and not report["approximately_20_hours_acquired"]:
        raise SystemExit("Corpus requires attention; see corpus-summary.json")


if __name__ == "__main__":
    main()
