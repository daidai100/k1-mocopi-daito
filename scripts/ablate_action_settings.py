#!/usr/bin/env python3
"""Paired controller-authority diagnostics with unchanged exported actor weights."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys

from freeze_source import freeze_source


def replay(job):
    from k1_motion.evaluation_panel import evaluate_panel

    library, panel, output, actor, trial_ids = job
    report = evaluate_panel(library, panel, output, actor, trial_ids=trial_ids)
    trials = [json.loads(line) for line in (Path(output) / "trials.jsonl").read_text().splitlines()]
    return {
        "output": output,
        "settings": report["action_settings"],
        "completed": sum(t["completed"] for t in trials),
        "tracking_passed": sum(t["tracking_passed"] for t in trials),
        "trials": trials,
        "partial_panel": report["partial_panel"],
        "behaviorally_accepted": False,
    }


def ablation_metadata(robot, parent, overrides, actor_path, revision):
    """Change only declared settings and bind the effective actuator contract."""
    from k1_motion.actuation import action_settings
    from k1_motion.actuators import actuator_contract
    original = action_settings(robot, parent.get('action_settings'))
    settings = action_settings(robot, {**original, **overrides})
    changed = sorted(k for k in set(original) | set(settings) if original.get(k) != settings.get(k))
    return {**parent, 'action_settings': settings,
        'actuator_contract': actuator_contract(robot, settings) if settings.get('actuator_profile') else None,
        'control_ablation': dict(parent_actor=str(Path(actor_path).resolve()),
            parent_sha256=parent['sha256'], parent_action_settings=original,
            changed_settings=changed, actor_weights_unchanged=True,
            replay_source_revision=revision, trained_with_these_settings=False),
        'behaviorally_accepted': False, 'hardware_verified': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append", default=[], help="Residual radians:command rad/s")
    parser.add_argument("--settings-case", action="append", default=[], help="JSON file of parent-setting overrides")
    parser.add_argument("--trial", action="append")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, revision = freeze_source(root)
    sys.path.insert(0, str(source))
    os.environ["K1_MOTION_ROOT"] = str(root)
    from k1_motion.robot import K1Model
    from k1_motion.learning import Policy

    robot = K1Model()
    policy = Policy(args.policy, robot.signature)
    cases = []
    for case in args.case:
        residual, velocity = map(float, case.split(":"))
        cases.append({"residual_scale": residual, "command_velocity_limit": velocity})
    cases.extend(json.loads(Path(p).read_text()) for p in args.settings_case)
    if not cases:
        parser.error("At least one --case or --settings-case is required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for i, settings in enumerate(cases):
        directory = output / f"case-{i:02d}"
        directory.mkdir()
        actor = directory / "actor.pt"
        shutil.copyfile(args.policy, actor)
        metadata = ablation_metadata(robot, policy.metadata, settings, args.policy, revision)
        actor.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        jobs.append((args.library, args.panel, str(directory / "replay"), str(actor), args.trial))
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for result in pool.map(replay, jobs):
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "trials"}), flush=True)
    report = {
        "source_revision": revision,
        "actor": args.policy,
        "actor_sha256": policy.metadata["sha256"],
        "scope": "Controller settings diagnostic with unchanged actor weights; no training or promotion",
        "results": results,
        "behaviorally_accepted": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
