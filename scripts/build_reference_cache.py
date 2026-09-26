#!/usr/bin/env python3
"""Preprocess the portable full train pool once on SSD for independent runs."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["K1_MOTION_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    from k1_motion.reference_cache import build_reference_cache
    from k1_motion.robot import K1Model
    torch.set_num_threads(1)
    print(json.dumps({"phase": "preprocessing_references", "library": args.library}), flush=True)
    print(json.dumps(build_reference_cache(args.library, args.output, K1Model()), indent=2), flush=True)


if __name__ == "__main__":
    main()
