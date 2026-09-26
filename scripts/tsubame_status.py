#!/usr/bin/env python3
"""Read persistent learner evidence and the interactive scheduler without a GPU login."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = r"""
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

job, run = sys.argv[1:]
root = Path.home() / "k1-motion"
operation = root / "operations" / f"allocation-{job}"
path = max(operation.glob("*-launch.json"), key=lambda p: p.stat().st_mtime) if run == "latest" else operation / f"{run}-launch.json"
receipt = json.loads(path.read_text())
output = Path(receipt["output"])
metrics_path = output / "metrics.jsonl"
metrics = [json.loads(s) for s in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
latest = metrics[-1] if metrics else None
scheduler = subprocess.run(["iqstat", "-j", job], text=True, capture_output=True)
config_path = output / "config.json"
config = json.loads(config_path.read_text()) if config_path.exists() else {}
evaluation_interval = config.get("config", {}).get("evaluation_interval", 0)
age = time.time() - metrics_path.stat().st_mtime if metrics_path.exists() else None
evaluation = sorted((output / "native-evaluations").glob("iteration-*.json"))
report = {
    "observed_unix": time.time(),
    "job_id": job,
    "node": receipt["node"],
    "allocation_active": scheduler.returncode == 0 and "job_state" in scheduler.stdout,
    "learner_pid_from_launch": receipt["pid"],
    "run": str(output),
    "source_revision": receipt["source_revision"],
    "start_iteration": receipt["start_iteration"],
    "target_iteration": receipt["target_iteration"],
    "latest_metrics": latest,
    "metrics_age_seconds": age,
    "num_envs": config.get("num_envs"),
    "stage": config.get("config", {}).get("stage"),
    "exit_code": receipt.get("exit_code"),
    "intentional_stop": receipt.get("intentional_stop", False),
    "action_settings": config.get("action_settings"),
    "reward_settings": config.get("reward_settings"),
    "bc_weight": config.get("config", {}).get("bc_weight"),
    "behaviorally_accepted": False,
    "hardware_verified": False,
    "scheduler_text": scheduler.stdout or scheduler.stderr,
}
if receipt.get("intentional_stop"):
    report["phase"] = "learner_intentionally_stopped"
elif "exit_code" in receipt:
    report["phase"] = "learner_finished" if receipt["exit_code"] == 0 else "learner_failed"
elif latest and evaluation_interval > 0 and (
    latest["iteration"] % evaluation_interval == 0 or latest["iteration"] == receipt["target_iteration"]
) and age > 20:
    report["phase"] = "native_evaluation_expected; awaiting its completion evidence"
elif latest and age < 60:
    report["phase"] = "learner_updates"
else:
    report["phase"] = "startup_or_stale_metrics; inspect log"
if metrics:
    rate = statistics.median(row["transitions_per_second"] for row in metrics[-20:])
    report["recent_transitions_per_second"] = rate
    report["remaining_learning_seconds_excluding_evaluation"] = (
        receipt["target_iteration"] - latest["iteration"]
    ) * config["num_envs"] * config["config"]["horizon"] / rate
if evaluation:
    data = json.loads(evaluation[-1].read_text())
    report["latest_native_evaluation"] = {
        "path": str(evaluation[-1]),
        "results": {k: {s: v[s] for s in ("completed", "total")} for k, v in data.items()},
        "scope": "native development replay; exported human-streaming acceptance is separate",
    }
print(json.dumps(report, allow_nan=False))
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="uh06814@login.t4.gsic.titech.ac.jp")
    parser.add_argument("--control-path")
    parser.add_argument("--job", required=True)
    parser.add_argument("--run", default="latest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    if args.control_path:
        command += ["-S", args.control_path]
    command += [args.host, shlex.join(["python3", "-", args.job, args.run])]
    result = subprocess.run(command, input=REMOTE, capture_output=True, text=True, check=True, timeout=30)
    report = json.loads(result.stdout)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("latest_metrics", "scheduler_text")}, indent=2
        )
    )
    if report["latest_metrics"]:
        print(
            json.dumps(
                {k: report["latest_metrics"][k] for k in ("iteration", "transitions", "loss", "body_rmse_m")}
            )
        )


if __name__ == "__main__":
    main()
