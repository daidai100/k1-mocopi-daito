#!/usr/bin/env python3
"""Shared held-out behavioral panel for three independent reward/size ablations."""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_server_ablation import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path, help="Reuse a verified identical-panel baseline")
    parser.add_argument("--baseline-only", action="store_true", help="Qualify the frozen baseline while learners are staged")
    parser.add_argument("--beam-size", type=int, default=0, help="Retain a behavioral checkpoint beam and controller-only anchors")
    parser.add_argument("--comparison-anchor", choices=("position", "pv"), default="position")
    args = parser.parse_args()
    if args.comparison_anchor == "pv" and not args.beam_size:
        raise ValueError("PV comparisons require the preserved controller anchors")
    deadline = time.monotonic()+120
    while not args.baseline_only and not (args.campaign/"status.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Campaign did not publish its initial status")
        time.sleep(1)
    output = args.campaign/"validation"
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/"monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    revision = json.loads((ROOT/"bundle.json").read_text())["source_revision"]
    source = ROOT/"artifacts/source-snapshots"/revision
    environment = {**os.environ, "K1_FROZEN_SOURCE": str(source/"k1_motion"),
                   "K1_MOTION_ROOT": str(ROOT), "PYTHONPATH": str(source)+os.pathsep+os.environ.get("PYTHONPATH", ""),
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    rows = [json.loads(line) for line in (ROOT/"library/index.jsonl").read_text().splitlines()]
    rows = [{"id": r["id"], "capture_group": r["capture_group"], "family": r["family"],
             "reference_path": str((ROOT/"library"/r["reference_path"]).resolve()),
             "cohort": "expanded_validation", "split": "validation"}
            for r in rows if r["split"] == "validation"]
    if len(rows) != 54 or len({r["id"] for r in rows}) != 54:
        raise ValueError("Expected the frozen 54-reference held-out panel")
    panel = output/"panel.json"
    if panel.exists() and json.loads(panel.read_text()) != rows:
        raise ValueError("Existing validation panel changed")
    write_json(panel, rows)
    if args.baseline_evaluation is not None and not (output/"baseline").exists():
        saved = json.loads((args.baseline_evaluation/"replay/summary.json").read_text())
        metadata = json.loads((args.baseline_evaluation/"actor.json").read_text())
        if (saved["contract"]["source_revision"] != revision
                or saved["contract"]["panel_sha256"] != hashlib.sha256(panel.read_bytes()).hexdigest()
                or saved["execution_errors"] or metadata["checkpoint_iteration"] != 2500):
            raise ValueError("Cannot reuse a different baseline or panel")
        shutil.copytree(args.baseline_evaluation, output/"baseline")
        write_json(output/"baseline/reuse.json", {"reused_from": str(args.baseline_evaluation),
                                                 "same_source_and_panel": True})
    state = {"panel_sha256": hashlib.sha256(panel.read_bytes()).hexdigest(), "trials": len(rows),
             "source_revision": revision, "behaviorally_accepted": False, "hardware_verified": False,
             "scope": "No-reset CPU exported-policy replay; family metrics, not reward ranking", "comparisons": {}}

    def update(**values):
        state.update(updated_unix=time.time(), **values)
        write_json(output/"status.json", state)
        if args.beam_size:
            from candidate_beam import candidate_beam
            candidates = state["comparisons"]
            beam = candidate_beam(candidates, args.beam_size)
            beam.update(updated_unix=time.time(), anchors=state.get("anchors", {}),
                        candidates=candidates, pending_evaluation=state.get("current") if state.get("phase") == "evaluating" else None)
            write_json(output/"beam.json", beam)

    def evaluate(checkpoint, key):
        destination = output/key
        summary = destination/"replay/summary.json"
        if not summary.exists():
            destination.mkdir(parents=True, exist_ok=True)
            if (destination/"replay").exists():
                (destination/"replay").rename(destination/f"interrupted-replay-{time.time_ns()}")
            update(phase="evaluating", current=key)
            with (destination/"evaluation.log").open("a") as log:
                subprocess.run([sys.executable, "-m", "k1_motion", "export", str(checkpoint),
                                "--output", str(destination/"actor.pt")], env=environment, cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
                subprocess.run([sys.executable, str(ROOT/"scripts/evaluate_rl_reference_pilot.py"),
                                "--panel", str(panel), "--policy", str(destination/"actor.pt"),
                                "--output", str(destination/"replay"), "--workers", "2"],
                               env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(summary.read_text())
        actor_metadata = json.loads((destination/"actor.json").read_text())
        result["training_exposure"] = {key: actor_metadata.get(key) for key in
                                        ("checkpoint_iteration", "checkpoint_transitions", "checkpoint_optimizer_steps")}
        result["checkpoint"] = str(checkpoint)
        result["actor"] = str(destination/"actor.pt")
        if result["contract"]["panel_sha256"] != state["panel_sha256"] or result["execution_errors"]:
            raise ValueError("Validation panel changed or trial execution failed")
        families = defaultdict(list)
        for row in rows:
            trial = json.loads((destination/"replay"/(row["id"]+".json")).read_text())
            families[row["family"]].append(trial)
        result["by_family"] = {family: {"trials": len(trials),
            "completed": sum(t["completed"] for t in trials), "clean": sum(t["clean_success"] for t in trials),
            "collision_trials": sum(t["self_collision_ticks"] > 0 for t in trials),
            "fell": sum(t["fell"] for t in trials),
            "mean_joint_rmse_rad": sum(t["joint_rmse_rad"] for t in trials)/len(trials),
            "mean_relative_body_rmse_m": sum(t["relative_body_rmse_m"] for t in trials)/len(trials)}
            for family, trials in families.items()}
        result["clean_ids"] = sorted(t["id"] if "id" in t else r["id"]
                                     for r in rows
                                     for t in [json.loads((destination/"replay"/(r["id"]+".json")).read_text())]
                                     if t["clean_success"])
        write_json(destination/"family-summary.json", result)
        return result

    try:
        baseline = evaluate(ROOT/"initialize.pt", "baseline")
        update(baseline=baseline["all"], baseline_by_family=baseline["by_family"])
        if args.beam_size:
            state["anchors"] = {"anchors/position_initializer": baseline}
            update()
            state["anchors"]["anchors/pv_initializer"] = evaluate(ROOT/"initialize-pv.pt", "baseline-pv")
            if args.comparison_anchor == "pv":
                baseline = state["anchors"]["anchors/pv_initializer"]
                state.update(baseline=baseline["all"], baseline_by_family=baseline["by_family"],
                             comparison_anchor="pv_initializer")
            update()
        if args.baseline_only:
            update(phase="baseline_completed")
            return
        while True:
            campaign = json.loads((args.campaign/"status.json").read_text())
            pending = False
            for name, run in campaign["runs"].items():
                directories = [Path(p) for p in run.get("training_directories", [args.campaign/name/"training"])]
                checkpoints = []
                for training in directories:
                    if (run.get("phase") == "failed" and (training/"checkpoint.pt").exists()
                            and not (training/"checkpoint-recovered-terminal.pt").exists()):
                        shutil.copy2(training/"checkpoint.pt", training/"checkpoint-recovered-terminal.partial")
                        (training/"checkpoint-recovered-terminal.partial").replace(training/"checkpoint-recovered-terminal.pt")
                    checkpoints.extend(sorted(training.glob("checkpoint-*.pt")))
                    if (training/"report.json").exists():
                        report = json.loads((training/"report.json").read_text())
                        iteration = report["last_metrics"]["iteration"]
                        if not (training/f"checkpoint-{iteration:06d}.pt").exists():
                            checkpoints.append(training/"checkpoint.pt")
                for checkpoint in checkpoints:
                    if time.time()-checkpoint.stat().st_mtime < 15:
                        pending = True
                        continue
                    key = name+"/"+checkpoint.stem
                    if key in state["comparisons"]:
                        continue
                    result = evaluate(checkpoint, key)
                    state["comparisons"][key] = {"all": result["all"], "by_family": result["by_family"],
                        "checkpoint": str(checkpoint), "actor": result["actor"], "clean_ids": result["clean_ids"],
                        "training_exposure": result["training_exposure"],
                        "raw_completion_delta": result["all"]["completed"]-baseline["all"]["completed"],
                        "clean_delta": result["all"]["clean"]-baseline["all"]["clean"],
                        "execution_errors": 0, "automatically_promoted": False}
                    update(latest=key)
            matched = defaultdict(dict)
            for key, result in state["comparisons"].items():
                transitions = result["training_exposure"]["checkpoint_transitions"]
                if transitions is not None:
                    matched[str(transitions)][key.split("/")[0]] = key
            state["matched_exposure_comparisons"] = {exposure: members for exposure, members in matched.items()
                                                       if len(members) == len(campaign["runs"])}
            phase = campaign["phase"]
            terminal = phase in ("completed", "completed_with_failures", "failed", "interrupted")
            update(phase="completed" if phase == "completed" and not pending else "learner_failed" if terminal and not pending
                   else "waiting_for_checkpoint")
            if terminal and not pending:
                return
            time.sleep(20)
    except Exception as error:
        update(phase="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
