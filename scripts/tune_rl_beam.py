#!/usr/bin/env python3
"""Run a bounded four-way source/thread sweep from the same durable checkpoints."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from run_server_ablation import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--baseline-bundle", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"phase": "running", "started_unix": time.time(), "settings": {}}
    cases = [("baseline56", args.baseline_bundle, [14]*4), ("optimized56", ROOT, [14]*4),
             ("optimized62", ROOT, [14, 16, 16, 16]), ("optimized64", ROOT, [16]*4)]
    try:
        for name, bundle, workers in cases:
            command = [sys.executable, str(ROOT / "scripts/resume_rl_beam.py"),
                       "--resume-campaign", str(args.campaign), "--bundle", str(bundle),
                       "--output", str(args.output / name), "--reference-cache", str(args.reference_cache),
                       "--workers", *map(str, workers), "--benchmark", "--iterations", str(args.iterations)]
            state.update(current=name, updated_unix=time.time())
            write_json(args.output / "status.json", state)
            with (args.output / (name+".log")).open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            result = json.loads((args.output / name / "status.json").read_text())
            if result["phase"] != "completed":
                raise RuntimeError(f"{name} did not pass finite-update/continuation/reload checks")
            state["settings"][name] = result["concurrent_result"]
            print(json.dumps({"setting": name, **result["concurrent_result"]}), flush=True)
        state.update(phase="completed", updated_unix=time.time())
        write_json(args.output / "status.json", state)
    except Exception as error:
        state.update(phase="failed", error=f"{type(error).__name__}: {error}", updated_unix=time.time())
        write_json(args.output / "status.json", state)
        raise


if __name__ == "__main__":
    main()
