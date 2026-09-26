#!/usr/bin/env python3
"""Audit/pad all train originals and retain every unpadded >10 s clip for training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    from k1_motion.standing_padding import prepare_library
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(prepare_library(args.source, args.output, workers=args.workers), indent=2), flush=True)


if __name__ == '__main__':
    main()
