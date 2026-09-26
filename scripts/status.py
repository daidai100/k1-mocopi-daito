#!/usr/bin/env python3
"""Show actual download and human-motion validation status without rehashing."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary_path = ROOT / "manifests/corpus-summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print("Corpus complete:", summary["download_and_numerical_validation_complete"])
        print(f"Validated human motion: {summary['validated_human_motion_hours']:.3f} hours; {summary['valid_human_tracks']:,} tracks")
        for name, item in summary["datasets"].items():
            print(f"  {name}: {item['hours']:.3f} hours, {item['unique_source_tracks']:,} tracks, {item['failed_files']} failed files")
        print("Pending:", ", ".join(summary["pending_datasets"]) or "none")
        print("Updated UTC:", summary["updated_utc"])
    else:
        print("Motion validation has not produced a summary yet.")
    progress_path = ROOT / "manifests/download-status.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        print("Transfer state:", progress["state"])
        chunks = progress.get("range_chunk_bytes", 0)
        for name, size in progress["archive_bytes"].items():
            if name.endswith(".part"):
                print(f"  {name}: {(size + chunks) / 1e9:.3f} GB including downloaded byte ranges")
        if progress["errors"]:
            print("Transfer errors:", progress["errors"])
    print("Data:", ROOT / "data/raw")
    print("These are human source recordings. K1 retargeting and controller training are pending.")


if __name__ == "__main__":
    main()
