#!/usr/bin/env python3
"""Continue a qualified RL preflight, then run the frozen controller panel."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--additional-iterations", type=int, default=295)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    preflight = json.loads((args.preflight / "report.json").read_text())
    if not preflight["finite_updates"] or preflight["checkpoint_reload_max_error"] != 0:
        raise ValueError("Training preflight did not qualify")
    if (args.output / "training").exists():
        raise ValueError("Campaign output already exists; preserve its artifacts and resume explicitly")
    settings = args.output / "action-settings.json"
    settings.write_text(json.dumps(preflight["action_settings"], indent=2) + "\n")
    source = json.loads((args.preflight / "source.json").read_text())
    environment = {**os.environ, "K1_FROZEN_SOURCE": str(Path(source["path"]) / "k1_motion")}
    configuration = preflight["config"]
    output = args.output / "training"
    status = {"source_revision": source["revision"], "preflight": str(args.preflight.resolve()),
              "library": preflight["library"], "target_iteration": preflight["last_metrics"]["iteration"]
              + args.additional_iterations, "data_role": "GMR references for RL; balance is learned",
              "behaviorally_accepted": False, "hardware_verified": False}

    def update(phase, **extra):
        status.update(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
        temporary = args.output / "status.partial"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(args.output / "status.json")

    command = [sys.executable, str(ROOT / "scripts/train_warp.py"),
               "--library", preflight["library"], "--output", str(output), "--stage", "student",
               "--resume", str(args.preflight / "checkpoint.pt"), "--iterations", str(args.additional_iterations),
               "--num-envs", str(preflight["num_envs"]), "--horizon", str(configuration["horizon"]),
               "--history", str(preflight["observation"]["history"]),
               "--hidden-sizes", *map(str, configuration["hidden_sizes"]),
               "--minibatch", str(configuration["minibatch"]), "--epochs", str(configuration["epochs"]),
               "--learning-rate", str(configuration["learning_rate"]),
               "--sampling", configuration["sampling"], "--matmul-precision", configuration["matmul_precision"],
               "--bc-weight", "0", "--action-settings", str(settings), "--reference-storage", "packed",
               "--self-collision-weight", str(preflight["reward_settings"]["self_collision_weight"]),
               "--root-velocity-weight", str(preflight["reward_settings"].get("root_velocity_weight", 0.5)),
               "--root-velocity-sigma", str(preflight["reward_settings"].get("root_velocity_sigma", 0.75)),
               "--evaluation-interval", "0", "--checkpoint-interval", "25",
               "--nconmax", "128", "--njmax", "1024", "--epa-horizon", "96", "--no-conditional-graphs",
               "--threads", "2", "--seed", str(configuration["seed"])]
    (args.output / "commands.json").write_text(json.dumps({"training": command}, indent=2) + "\n")
    try:
        update("training", command=command)
        with (args.output / "training.log").open("w") as log:
            subprocess.run(command, env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        update("controller_evaluation")
        evaluation = args.output / "evaluation"
        with (args.output / "evaluation.log").open("w") as log:
            subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_rl_reference_pilot.py"),
                            "--panel", str(args.panel), "--policy", str(output / "actor.pt"),
                            "--output", str(evaluation), "--workers", "2"],
                           env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        baseline = json.loads(args.baseline_evaluation.read_text())
        result = json.loads((evaluation / "summary.json").read_text())
        if baseline["contract"]["panel_sha256"] != result["contract"]["panel_sha256"]:
            raise ValueError("Evaluation panel changed")
        comparison = {"baseline": baseline, "candidate": result,
                      "clean_pass_delta": result["all"]["clean"] - baseline["all"]["clean"],
                      "completion_delta": result["all"]["completed"] - baseline["all"]["completed"],
                      "automatically_promoted": False,
                      "scope": "Bounded RL experiment; original controller export retained"}
        (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        update("completed", comparison=str(args.output / "comparison.json"))
    except Exception as error:
        update("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
