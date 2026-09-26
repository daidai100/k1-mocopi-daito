#!/usr/bin/env python3
"""Portable, unmodified train pool plus the fixed validation canary."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--validation-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "clips").mkdir()
    rows, seen, size = [], set(), 0
    for directory, split in ((args.library, "train"), (args.validation_library, "validation")):
        for line in (directory / "index.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["split"] != split:
                continue
            if row["id"] in seen:
                raise ValueError("Duplicate clip identity across ablation splits")
            seen.add(row["id"])
            source = directory / row["reference_path"]
            relative = Path("clips")/(row["id"]+".npz")
            target = args.output / relative
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            row["reference_path"] = str(relative)
            size += target.stat().st_size
            rows.append(row)
    manifest = "".join(json.dumps(r, sort_keys=True)+"\n" for r in rows)
    (args.output / "index.jsonl").write_text(manifest)
    receipt = {"parent_library": str(args.library.resolve()),
               "validation_library": str(args.validation_library.resolve()),
               "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
               "splits": dict(Counter(r["split"] for r in rows)),
               "train_families": dict(Counter(r["family"] for r in rows if r["split"] == "train")),
               "clip_bytes": size, "trajectories_unchanged": True,
               "scope": "Paths rewritten only; training admission and payloads are unchanged"}
    (args.output / "staging.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
