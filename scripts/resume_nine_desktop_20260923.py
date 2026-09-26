#!/usr/bin/env python3
"""Resume the interrupted desktop nine-run queue without replacing prior artifacts."""

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import torch


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "artifacts/nine-run-20260923/desktop"
TARGET = 3000


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_iteration(path):
    return int(torch.load(path, map_location="cpu", weights_only=True)["iteration"])


def verified_report(path, expected):
    if not path.is_file():
        return False
    report = read(path)
    return (report.get("finite_updates") is True
            and report.get("checkpoint_reload_max_error") == 0
            and report.get("last_metrics", {}).get("iteration") == expected)


def command_for(name, phase, output, *, resume=None, remaining=None):
    template = CAMPAIGN / "desktop_scale85_seed44" / f"{phase}-command.json"
    command = read(template)
    command[command.index("--output") + 1] = str(output)
    command[command.index("--reference-scale") + 1] = "0.95" if "scale95" in name else "0.85"
    command[command.index("--iterations") + 1] = str(remaining if resume else 25 if phase == "preflight" else TARGET)
    if resume:
        position = command.index("--initialize")
        command[position:position + 2] = ["--resume", str(resume)]
    return command


def launch(name, phase, output, command, *, resumed_from=None):
    directory = CAMPAIGN / name
    status_path = directory / "status.json"
    source = next((Path(read(CAMPAIGN / "plan.json")["bundle"]) / "artifacts/source-snapshots").glob("*/k1_motion"))
    env = {k: v for k, v in os.environ.items() if k not in
           ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
    env.update(CUDA_VISIBLE_DEVICES="0", K1_FROZEN_SOURCE=str(source), OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_WAIT_POLICY="PASSIVE", GOMP_SPINCOUNT="0")
    output.parent.mkdir(parents=True, exist_ok=True)
    command_path = directory / f"{output.name}-command.json"
    write(command_path, command)
    with (directory / f"{output.name}.log").open("w") as log:
        process = subprocess.Popen(command, cwd=read(CAMPAIGN / "plan.json")["bundle"], env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        status = dict(next(run for run in read(CAMPAIGN / "plan.json")["runs"] if run["name"] == name),
                      phase=phase, trainer_pid=process.pid, supervisor_pid=os.getpid(),
                      output=str(output), command_file=str(command_path),
                      resumed_from=str(resumed_from) if resumed_from else None,
                      started_at=datetime.now(timezone.utc).isoformat())
        write(status_path, status)
        code = process.wait()
    report = output / "report.json"
    if code or not verified_report(report, TARGET if phase == "training" else 25):
        status.update(phase="interrupted" if code in (-15, 143) else "failed",
                      trainer_exit_code=code, report_present=report.is_file())
        write(status_path, status)
        raise RuntimeError(f"{name} {phase} exited {code}; inspect {directory / (output.name + '.log')}")
    result = read(report)["last_metrics"]
    status.update(phase=f"{phase}_complete", trainer_exit_code=0,
                  updates=result["iteration"], transitions=result["transitions"],
                  adam_steps=result["optimizer_steps"])
    write(status_path, status)


def main():
    plan = read(CAMPAIGN / "plan.json")
    if plan["host"] != "desktop" or plan["target_updates_per_run"] != TARGET:
        raise ValueError("Unexpected desktop campaign plan")
    if not Path(plan["bundle"]).is_dir() or not Path(plan["inputs"]).is_dir():
        raise FileNotFoundError("Frozen bundle or inputs missing")
    for run in plan["runs"]:
        name = run["name"]
        directory = CAMPAIGN / name
        candidates = [directory / "training", *(path for path in directory.glob("training-*") if path.is_dir())]
        if any(verified_report(path / "report.json", TARGET) for path in candidates):
            continue
        checkpoints = [(checkpoint_iteration(path / "checkpoint.pt"), path / "checkpoint.pt")
                       for path in candidates if (path / "checkpoint.pt").is_file()]
        if checkpoints:
            iteration, checkpoint = max(checkpoints)
            if not 0 < iteration < TARGET:
                raise ValueError(f"Invalid resume iteration {iteration} for {name}")
            output = directory / f"training-resume-{iteration:06d}"
            if output.exists():
                output = directory / f"training-resume-{iteration:06d}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
            launch(name, "training", output,
                   command_for(name, "training", output, resume=checkpoint, remaining=TARGET - iteration),
                   resumed_from=checkpoint)
            continue
        preflight = directory / "preflight"
        if not verified_report(preflight / "report.json", 25):
            if preflight.exists():
                preflight = directory / f"preflight-retry-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
            launch(name, "preflight", preflight, command_for(name, "preflight", preflight))
        training = directory / "training"
        if training.exists():
            training = directory / f"training-restart-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        launch(name, "training", training, command_for(name, "training", training))
    write(CAMPAIGN / "final-status.json", {run["name"]: read(CAMPAIGN / run["name"] / "status.json")
                                           for run in plan["runs"]})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Desktop queue recovery failed: {exc}", file=sys.stderr, flush=True)
        raise
