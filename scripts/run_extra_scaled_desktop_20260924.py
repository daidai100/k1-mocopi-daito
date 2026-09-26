#!/usr/bin/env python3
"""Queue two additional scaled desktop learners after the existing nine-run queue."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "artifacts/nine-run-20260923/desktop"
OUTPUT = ROOT / "artifacts/desktop-extra-scales-20260924"
TARGET = 3000
RUNS = (("desktop_scale80_seed44", 0.8), ("desktop_scale70_seed44", 0.7))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def report_complete(path, iteration):
    if not path.is_file():
        return False
    report = read(path)
    return (report.get("finite_updates") is True
            and report.get("checkpoint_reload_max_error") == 0
            and report.get("last_metrics", {}).get("iteration") == iteration)


def checkpoint_iteration(path):
    return int(torch.load(path, map_location="cpu", weights_only=True)["iteration"])


def prepare():
    original = read(PRIOR / "plan.json")
    bundle = Path(original["bundle"])
    inputs = Path(original["inputs"])
    sources = list((bundle / "artifacts/source-snapshots").glob("*/k1_motion"))
    if len(sources) != 1 or sources[0].parent.name != original["source_revision"]:
        raise ValueError("Frozen training source differs from the existing campaign")
    template = read(PRIOR / "desktop_scale95_seed44/training-command.json")
    for path in (bundle / "initialize.pt", inputs / "reference-cache.pt",
                 inputs / "library/index.jsonl", Path(template[template.index("--action-settings") + 1])):
        if not path.is_file():
            raise FileNotFoundError(path)
    if read(PRIOR / "desktop_scale95_seed44/status.json")["reference_scale"] != 0.95:
        raise ValueError("Unexpected preceding desktop treatment")
    plan = {
        "version": "desktop-extra-scales-v1",
        "source_campaign": str(PRIOR),
        "source_revision": original["source_revision"],
        "frozen_bundle": str(bundle),
        "inputs": str(inputs),
        "ordering": "wait for scale 0.95, then scale 0.80, then scale 0.70",
        "target_updates_per_run": TARGET,
        "expected_transitions_per_run": TARGET * 2048 * 32,
        "reference_scale_contract": "k1-travel-sole-scale-v1",
        "corpus": original["corpus"],
        "runs": [{"name": name, "reference_scale": scale, "seed": 44,
                  "physical_gpu": 0, "phase": "queued"} for name, scale in RUNS],
    }
    plan_path = OUTPUT / "plan.json"
    if plan_path.exists():
        if read(plan_path) != plan:
            raise ValueError("Existing extra-scale plan differs")
    else:
        write(plan_path, plan)
    for run in plan["runs"]:
        status = OUTPUT / run["name"] / "status.json"
        if not status.exists():
            write(status, run)
    return bundle, sources[0], template


def wait_for_prior():
    status_path = PRIOR / "desktop_scale95_seed44/status.json"
    report_path = PRIOR / "desktop_scale95_seed44/training/report.json"
    while True:
        status = read(status_path)
        if status["phase"] == "training_complete" and report_complete(report_path, TARGET):
            return
        if status["phase"] == "failed":
            raise RuntimeError("Preceding scale 0.95 run did not complete")
        if status["phase"] == "interrupted":
            active = subprocess.run(
                ["systemctl", "--user", "is-active", "--quiet",
                 "k1-nine-desktop-resume-20260923.service"], check=False,
            ).returncode == 0
            if not active:
                raise RuntimeError("Preceding scale 0.95 run is interrupted")
        time.sleep(15)


def command(template, output, scale, *, updates, resume=None):
    result = template.copy()
    result[result.index("--output") + 1] = str(output)
    result[result.index("--reference-scale") + 1] = str(scale)
    result[result.index("--iterations") + 1] = str(updates)
    if resume is not None:
        start = result.index("--initialize")
        result[start:start + 2] = ["--resume", str(resume)]
    return result


def execute(name, scale, phase, output, template, bundle, source, *, resume=None, updates=None):
    directory = OUTPUT / name
    status_path = directory / "status.json"
    cmd = command(template, output, scale, updates=updates or (25 if phase == "preflight" else TARGET),
                  resume=resume)
    env = {key: value for key, value in os.environ.items()
           if key not in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
    env.update(CUDA_VISIBLE_DEVICES="0", K1_FROZEN_SOURCE=str(source), OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_WAIT_POLICY="PASSIVE",
               GOMP_SPINCOUNT="0")
    output.parent.mkdir(parents=True, exist_ok=True)
    command_file = directory / f"{output.name}-command.json"
    write(command_file, cmd)
    with (directory / f"{output.name}.log").open("w") as log:
        process = subprocess.Popen(cmd, cwd=bundle, env=env, stdout=log, stderr=subprocess.STDOUT)
        status = {"name": name, "reference_scale": scale, "seed": 44, "physical_gpu": 0,
                  "phase": phase, "trainer_pid": process.pid, "supervisor_pid": os.getpid(),
                  "started_at": datetime.now(timezone.utc).isoformat(), "output": str(output),
                  "command_file": str(command_file),
                  "resumed_from": str(resume) if resume else None}
        write(status_path, status)
        code = process.wait()
    expected = TARGET if phase == "training" else 25
    if code or not report_complete(output / "report.json", expected):
        status.update(phase="interrupted" if code in (-15, 143) else "failed",
                      trainer_exit_code=code)
        write(status_path, status)
        raise RuntimeError(f"{name} {phase} exited {code}; inspect {output.parent / (output.name + '.log')}")
    metrics = read(output / "report.json")["last_metrics"]
    status.update(phase=f"{phase}_complete", trainer_exit_code=0,
                  updates=metrics["iteration"], transitions=metrics["transitions"],
                  adam_steps=metrics["optimizer_steps"])
    write(status_path, status)


def main():
    bundle, source, template = prepare()
    wait_for_prior()
    for name, scale in RUNS:
        directory = OUTPUT / name
        candidates = [directory / "training", *(p for p in directory.glob("training-resume-*") if p.is_dir())]
        if any(report_complete(p / "report.json", TARGET) for p in candidates):
            continue
        checkpoints = [(checkpoint_iteration(p / "checkpoint.pt"), p / "checkpoint.pt")
                       for p in candidates if (p / "checkpoint.pt").is_file()]
        if checkpoints:
            iteration, checkpoint = max(checkpoints)
            if not 0 < iteration < TARGET:
                raise ValueError(f"Invalid checkpoint iteration {iteration} for {name}")
            output = directory / f"training-resume-{iteration:06d}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
            execute(name, scale, "training", output, template, bundle, source,
                    resume=checkpoint, updates=TARGET - iteration)
            continue
        preflight = directory / "preflight"
        if not report_complete(preflight / "report.json", 25):
            if preflight.exists():
                preflight = directory / f"preflight-retry-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
            execute(name, scale, "preflight", preflight, template, bundle, source, updates=25)
        training = directory / "training"
        if training.exists():
            training = directory / f"training-retry-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        execute(name, scale, "training", training, template, bundle, source)
    write(OUTPUT / "final-status.json", {name: read(OUTPUT / name / "status.json") for name, _ in RUNS})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Desktop extra-scale queue failed: {exc}", file=sys.stderr, flush=True)
        raise
