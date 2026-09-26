#!/usr/bin/env python3
"""Compare the retained PPO weights from matched scalar/process/C++ CPU runs."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    baseline, candidate = [torch.load(p / "checkpoint.pt", map_location="cpu", weights_only=True)
                           for p in (args.baseline, args.candidate)]
    keys = ("model_signature", "reference_fingerprint", "task_version", "observation", "config",
            "action_settings", "reward_settings", "num_envs", "iteration", "transitions")
    if any(baseline[k] != candidate[k] for k in keys):
        raise ValueError("Unmatched checkpoint contracts")
    if baseline["model"].keys() != candidate["model"].keys():
        raise ValueError("Model fields differ")
    errors = {k: float((v-candidate["model"][k]).abs().max()) for k, v in baseline["model"].items()}
    exact = all(torch.equal(v, candidate["model"][k]) for k, v in baseline["model"].items())
    report = {"baseline": str(args.baseline), "candidate": str(args.candidate),
              "iteration": candidate["iteration"], "transitions": candidate["transitions"],
              "model_tensors": len(errors), "weights_and_normalization_exact": exact,
              "max_abs_error": max(errors.values()), "field_errors": errors,
              "scope": "Matched CPU-backend PPO checkpoint equivalence, not behavioral acceptance"}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "field_errors"}, indent=2))
    if not exact:
        raise ValueError("CPU backend checkpoint changed")


if __name__ == "__main__":
    main()
