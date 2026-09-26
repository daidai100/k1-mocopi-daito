#!/usr/bin/env python3
"""Build a new train-only library with versioned legacy velocity clock repair."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k1_motion.reference_velocity_clock import repair_training_library  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--train-ids",
        type=Path,
        required=True,
        help="Frozen curriculum JSON with train_ids, or a JSON list of IDs",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    selected = json.loads(args.train_ids.read_text())
    ids = selected["train_ids"] if isinstance(selected, dict) else selected
    report = repair_training_library(args.library, args.output, ids, workers=args.workers)
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "version",
                    "clips",
                    "repaired_clips",
                    "unchanged_clips",
                    "invalid_ticks",
                    "geometry_arrays_unchanged",
                    "physics_qualified",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
