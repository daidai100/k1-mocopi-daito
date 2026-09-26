#!/usr/bin/env python3
"""Retain a cache through proved preprocessing-identical runtime/PPO changes."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["K1_MOTION_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT/"src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-package", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from k1_motion.reference_cache import rebind_reference_cache
    from k1_motion.robot import K1Model
    torch.set_num_threads(1)
    print(json.dumps(rebind_reference_cache(args.cache, args.output, args.old_package, args.library, K1Model()),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
