#!/usr/bin/env python3
"""Retain remote milestones and run the fixed CPU human panel beside GPU training."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from compare_human_panels import compare


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def collect(args):
    local = Path(args.local_run).resolve()
    local.mkdir(parents=True, exist_ok=True)
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    if args.control_path:
        ssh += ["-S", args.control_path]
    # Numbered checkpoints are written only at completed milestones. Fetch a
    # checkpoint only once it is older than 60 seconds, never the mutable latest.
    query = """
import json, sys, time
from pathlib import Path
root=Path(sys.argv[1]); now=time.time()
finished=(root/'report.json').exists()
files=[]
for p in root.glob('checkpoint-*.pt'):
    if finished or now-p.stat().st_mtime > 60:
        files.append(p.name)
print(json.dumps({'files':sorted(files),'finished':finished}))
"""
    result = subprocess.run(
        [*ssh, args.host, shlex.join(["python3", "-", args.remote_run])],
        input=query, text=True, capture_output=True, check=True, timeout=30,
    )
    remote = json.loads(result.stdout)
    for name in ["config.json", "metrics.jsonl", *remote["files"],
                 *(["report.json", "source.json", "actor.pt", "actor.json"] if remote["finished"] else [])]:
        destination = local / name
        if name.startswith("checkpoint-") and destination.exists():
            continue
        subprocess.run(
            ["rsync", "-a", "--timeout=30", "-e", shlex.join(ssh),
             f"{args.host}:{args.remote_run}/{name}", str(destination)],
            check=True, capture_output=True, text=True, timeout=90,
        )
    return local, remote["files"], remote["finished"]


def evaluate(args, local, name):
    iteration = int(name.removeprefix("checkpoint-").removesuffix(".pt"))
    output = local / f"validation-{iteration:06d}"
    comparison = output / "comparison.json"
    if comparison.exists():
        return json.loads(comparison.read_text())
    output.mkdir(exist_ok=True)
    environment = {
        **os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "PYTHONPATH": str(Path(args.source).resolve()),
        "K1_MOTION_ROOT": str(Path(__file__).resolve().parents[1]),
    }
    actor = output / "actor.pt"
    with (output / "evaluation.log").open("a") as log:
        if not (output / "export-report.json").exists():
            subprocess.run(
                [sys.executable, "-m", "k1_motion", "export", str(local / name), "--output", str(actor)],
                env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120,
            )
        export = json.loads((output / "export-report.json").read_text())
        if export["iteration"] != iteration or not export["finite"] or export["export_reload_max_error"] > 1e-6:
            raise ValueError("Milestone export does not match a finite, reloadable checkpoint")
        panel = output / "human-panel"
        if not (panel / "report.json").exists():
            if panel.exists():
                panel.rename(output / f"interrupted-panel-{time.time_ns()}")
            subprocess.run(
                [sys.executable, "-m", "k1_motion", "evaluate-panel", "--library", args.library,
                 "--panel", args.panel, "--policy", str(actor), "--output", str(panel),
                 "--workers", str(args.workers)],
                env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800,
            )
    result = compare(args.baseline, panel)
    result.update(checkpoint_iteration=iteration, evaluation_source=str(Path(args.source).resolve()))
    write_json(comparison, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="uh06814@login.t4.gsic.titech.ac.jp")
    parser.add_argument("--control-path")
    parser.add_argument("--remote-run", required=True)
    parser.add_argument("--local-run", required=True)
    parser.add_argument("--source", required=True, help="Frozen package parent for all CPU evaluations")
    parser.add_argument("--library", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    local = Path(args.local_run).resolve()
    local.mkdir(parents=True, exist_ok=True)
    lock = (local / "monitor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        state = {"observed_unix": time.time(), "remote_run": args.remote_run,
                 "behaviorally_accepted": False, "hardware_verified": False}
        try:
            local, files, finished = collect(args)
            state.update(phase="checking_milestones", learner_finished=finished)
            results = []
            for name in files:
                state.update(phase="evaluating", checkpoint=name)
                write_json(local / "monitor-status.json", state)
                result = evaluate(args, local, name)
                results.append({"iteration": result["checkpoint_iteration"],
                                "completed": result["comparison"]["completed"]["candidate"],
                                "strict_passes": result["comparison"]["tracking_passed"]["candidate"],
                                "trials": result["trials"]})
            state.update(phase="finished" if finished else "waiting_for_checkpoint", results=results)
            # This is a development ranking only. The original baseline remains
            # retained and no exported policy is promoted or overwritten.
            if results:
                state["best_observed_milestone"] = max(results, key=lambda x: (x["strict_passes"], x["completed"]))
            write_json(local / "monitor-status.json", state)
            print(json.dumps(state), flush=True)
            if finished or args.once:
                return
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
            state.update(phase="monitor_error", error=f"{type(error).__name__}: {error}")
            write_json(local / "monitor-status.json", state)
            print(json.dumps(state), flush=True)
            if args.once:
                raise
        time.sleep(60)


if __name__ == "__main__":
    main()
