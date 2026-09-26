#!/usr/bin/env python3
"""Evaluate frozen held-out references beside a local learner, never promote it."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    output = args.campaign / "validation"
    output.mkdir(exist_ok=True)
    lock = (output / "monitor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = json.loads((args.campaign / "source.json").read_text())
    environment = {**os.environ, "K1_FROZEN_SOURCE": str(Path(source["path"]) / "k1_motion"),
                   "PYTHONPATH": source["path"], "K1_MOTION_ROOT": str(ROOT),
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    panel = output / "panel.json"
    rows = [json.loads(line) for line in (args.library / "index.jsonl").read_text().splitlines()]
    rows = [{"id": r["id"], "capture_group": r["capture_group"], "family": r["family"],
             "reference_path": str((args.library / r["reference_path"]).resolve()),
             "cohort": "expanded_validation", "split": "validation"}
            for r in rows if r["split"] == "validation"]
    if not rows:
        raise ValueError("No held-out references")
    # The evaluator independently rejects related-take overlap against each
    # exported checkpoint's complete inherited training provenance.
    if panel.exists() and json.loads(panel.read_text()) != rows:
        raise ValueError("Frozen panel changed")
    if not panel.exists():
        write_json(panel, rows)
    state = {"panel": str(panel), "panel_sha256": hashlib.sha256(panel.read_bytes()).hexdigest(),
             "trials": len(rows), "source_revision": source["revision"],
             "behaviorally_accepted": False, "hardware_verified": False,
             "scope": "Independent scalar CPU runtime replay; not training reward or automatic promotion"}

    def update(**extra):
        state.update(updated_unix=time.time(), **extra)
        write_json(output / "status.json", state)

    def evaluate(checkpoint, name):
        destination = output / name
        summary = destination / "replay/summary.json"
        if summary.exists():
            return json.loads(summary.read_text())
        destination.mkdir(exist_ok=True)
        if (destination / "replay").exists():
            (destination / "replay").rename(destination / f"interrupted-replay-{time.time_ns()}")
        update(phase="evaluating", checkpoint=str(checkpoint), evaluation=name)
        with (destination / "evaluation.log").open("a") as log:
            subprocess.run([sys.executable, "-m", "k1_motion", "export", str(checkpoint),
                            "--output", str(destination / "actor.pt")], env=environment, cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_rl_reference_pilot.py"),
                            "--panel", str(panel), "--policy", str(destination / "actor.pt"),
                            "--output", str(destination / "replay"), "--workers", "2"],
                           env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        return json.loads(summary.read_text())

    try:
        baseline = evaluate(args.baseline, "baseline")
        update(baseline=baseline["all"])
        while True:
            training = args.campaign / "training"
            finished = (training / "report.json").exists()
            checkpoints = [p for p in sorted(training.glob("checkpoint-*.pt"))
                           if time.time()-p.stat().st_mtime > 10]
            for checkpoint in checkpoints:
                name = checkpoint.stem
                comparison_path = output / name / "comparison.json"
                if comparison_path.exists():
                    continue
                result = evaluate(checkpoint, name)
                if result["contract"]["panel_sha256"] != baseline["contract"]["panel_sha256"]:
                    raise ValueError("Comparison panel differs")
                comparison = {"baseline": baseline["all"], "candidate": result["all"],
                              "raw_completion_delta": result["all"]["completed"]-baseline["all"]["completed"],
                              "clean_delta": result["all"]["clean"]-baseline["all"]["clean"],
                              "execution_errors": result["execution_errors"], "automatically_promoted": False}
                write_json(comparison_path, comparison)
                update(latest_comparison=comparison, latest_checkpoint=str(checkpoint))
            update(phase="completed" if finished else "waiting_for_checkpoint")
            if finished:
                return
            campaign_state = json.loads((args.campaign / "status.json").read_text())
            if campaign_state.get("trainer_exit_code") not in (None, 0):
                update(phase="learner_failed")
                return
            time.sleep(20)
    except Exception as error:
        update(phase="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
