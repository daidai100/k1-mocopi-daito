#!/usr/bin/env python3
"""Freeze matched controller experiments; original references and physics stay pinned."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

from freeze_source import freeze_source
from run_warp_capacity_queue import build_commands


def controller_variants(parent):
    common = {**parent, "mask_inactive_actions": True, "ankle_prior_scale": 1.0}
    variants = [
        ("legacy_baseline", {"mask_inactive_actions": False}),
        ("masked_baseline", {}),
        ("velocity", {"target_velocity_scale": 1.0}),
        ("position", {"residual_scale": 0.4}),
        ("combined", {"residual_scale": 0.4, "target_velocity_scale": 1.0}),
        ("prior_half", {"ankle_prior_scale": 0.5}),
        ("prior_off", {"ankle_prior_scale": 0.0}),
        ("command_feedback", {}),
    ]
    return [
        dict(name=name, settings={**common, **delta}, command_feedback=name == "command_feedback")
        for name, delta in variants
    ]


def set_option(cmd, flag, value):
    if flag in cmd:
        cmd[cmd.index(flag) + 1] = str(value)
    else:
        cmd.extend([flag, str(value)])


def build_runs(
    output,
    variants,
    *,
    library,
    cache,
    initializer,
    updates,
    milestone,
    num_envs,
    preflight_updates=5,
    workers=24,
    seed=45,
):
    runs = []
    for variant in variants:
        name = variant["name"]
        run = build_commands(
            output,
            output / "scripts",
            library,
            cache,
            initializer,
            updates=updates,
            num_envs=num_envs,
            milestone=milestone,
            seed=seed,
            preflight_updates=preflight_updates,
        )[0]
        run.update(variant, backend="mujoco_cpp")
        for key in ("command", "preflight_command"):
            cmd = run[key]
            cmd[:] = [x.replace(str(output / "small"), str(output / name)) for x in cmd]
            i = cmd.index("--curriculum-manifest")
            del cmd[i : i + 2]
            for flag, value in [
                ("--backend", "mujoco_cpp"),
                ("--reward-profile", "survival-position-v2"),
                ("--action-settings", output / (name + "-controller.json")),
                ("--scene-transitions", output / "scenes.json"),
                ("--gamma", math.exp(-0.02 / 60)),
                ("--gae-lambda", ".99"),
                ("--cpu-workers", workers),
                ("--cpu-chunk-size", 4),
                ("--checkpoint-interval", min(50, milestone)),
            ]:
                set_option(cmd, flag, value)
            if variant["command_feedback"]:
                cmd.append("--command-feedback")
        if variant["command_feedback"]:
            run["actor_parameters"] += 660 * 512
            run["critic_parameters"] += 660 * 512
            run["total_parameters"] += 2 * 660 * 512
        runs.append(run)
    return runs


def prepare(args):
    root = Path(os.environ.get("K1_MOTION_ROOT", Path(__file__).resolve().parents[1]))
    out, prior = args.output.resolve(), args.prior_campaign.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshot, revision = freeze_source(root)
    (out / "src").symlink_to(snapshot, target_is_directory=True)
    shutil.copytree(root / "scripts", out / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    (out / "inputs").mkdir()
    library = (prior / "inputs/library").resolve(strict=True)
    (out / "inputs/library").symlink_to(library, target_is_directory=True)
    for filename in ("training-panel.json", "development-panel.json"):
        shutil.copy2(prior / "inputs" / filename, out / "inputs" / filename)
    shutil.copy2(prior / "initialize.pt", out / "initialize.pt")
    shutil.copy2(prior / "scenes.json", out / "scenes.json")
    sys.path.insert(0, str(snapshot))
    import torch
    from k1_motion.actuation import action_settings
    from k1_motion.robot import K1Model
    from k1_motion.reference_cache import rebind_reference_cache
    from k1_motion.observations import observation_contract
    from k1_motion.model_transfer import initialize_model, checkpoint_hidden_sizes
    from k1_motion.learning import ActorCritic
    from k1_motion.actuators import actuator_contract
    from k1_motion.world_objective import WorldBodyTracking, SafetyObjective
    from k1_motion.scene_transitions import transition_contract

    torch.set_num_threads(1)
    torch.manual_seed(45)
    old_plan = json.loads((prior / "plan.json").read_text())
    cmd = old_plan["runs"][0]["command"]
    cache = Path(cmd[cmd.index("--reference-cache") + 1])
    new_cache = out / "reference-cache.pt"
    rebind_reference_cache(cache, new_cache, old_plan["frozen_source"], library, K1Model())
    saved = torch.load(out / "initialize.pt", map_location="cpu", weights_only=True)
    robot = K1Model()
    parent = action_settings(
        robot, json.loads((root / "configs/controller-pv-official80-guard03-speed-v1.json").read_text())
    )
    if saved.get("action_chunk") or saved["observation"].get("preview_horizon_s") != 0.3:
        raise ValueError("Requires the common single-step preview initializer")
    variants = controller_variants(parent)
    runs = build_runs(
        out,
        variants,
        library=library,
        cache=new_cache,
        initializer=out / "initialize.pt",
        updates=args.updates,
        milestone=args.milestone,
        num_envs=args.num_envs,
        preflight_updates=args.preflight_updates,
        workers=args.workers,
    )
    scene = transition_contract(json.loads((out / "scenes.json").read_text()), 0.02)
    reward = dict(
        self_collision_weight=0.0,
        profile="survival-position-v2",
        safety=SafetyObjective().contract,
        spatial_tracking=WorldBodyTracking("survival-position-v2").contract,
    )
    for run in runs:
        config = action_settings(robot, run["settings"])
        (out / (run["name"] + "-controller.json")).write_text(json.dumps(config, indent=2) + "\n")
        # Evaluate the inherited weights under each changed controller before any update.
        obs = observation_contract(10, "preview", command_feedback=run["command_feedback"])
        critic_size = saved["critic_size"] + obs["size"] - saved["observation"]["size"]
        model = ActorCritic(obs["size"], critic_size, checkpoint_hidden_sizes(saved))
        transfer = initialize_model(model, saved, obs)
        initial = {
            **saved,
            "model": model.state_dict(),
            "actor_size": obs["size"],
            "critic_size": critic_size,
            "observation": obs,
            "action_settings": config,
            "physics_contract": {"actuator": actuator_contract(robot, config)},
            "iteration": 0,
            "transitions": 0,
            "optimizer_steps": 0,
            "source_revision": revision,
            "initialization_transfer": transfer,
            "optimizer_resume_supported": False,
        }
        initial.pop("optimizer", None)
        path = out / (run["name"] + "-initial.pt")
        torch.save(initial, path)
        run["baseline_checkpoint"] = str(path)
        run["expected_training_contract"] = dict(
            action_settings=config,
            observation=obs,
            reward_settings=reward,
            scene_transitions=scene,
            gamma=math.exp(-0.02 / 60),
            gae_lambda=0.99,
        )
    rows = [json.loads(x) for x in (library / "index.jsonl").read_text().splitlines()]
    if any(r["split"] != "train" or r["is_mirror"] for r in rows):
        raise ValueError("Requires training originals only")
    plan = dict(
        version="controller-authority-20260926-v1",
        output=str(out),
        scripts=str(out / "scripts"),
        source_revision=revision,
        frozen_source=str(snapshot / "k1_motion"),
        initializer=str(out / "initialize.pt"),
        initializer_sha256=hashlib.sha256((out / "initialize.pt").read_bytes()).hexdigest(),
        library_manifest_sha256=hashlib.sha256((library / "index.jsonl").read_bytes()).hexdigest(),
        originals=len(rows),
        by_family=dict(Counter(r["family"] for r in rows)),
        updates=args.updates,
        num_envs=args.num_envs,
        horizon=32,
        preflight_updates=args.preflight_updates,
        milestone=args.milestone,
        evaluation_workers=6,
        seed=45,
        backend="mujoco_cpp",
        require_full_coverage=False,
        runs=runs,
        authority_trace=True,
        stopping="equal transition budgets; execution/nonfinite errors stop queue; retain per-controller champions",
        confirmation_panel_used=False,
        behaviorally_accepted=False,
        hardware_verified=False,
    )
    manifest = root / "retarget-sources/manifest.json"
    if manifest.exists():
        plan['post_experiments'] = [dict(name='retarget_speed',
            command=[sys.executable,str(out/'scripts/run_authority_retarget.py'),
                '--manifest',str(manifest),'--output',str(out/'retarget-speed'),'--workers','6'],
            log=str(out/'retarget-speed.log'),receipt=str(out/'retarget-speed/summary.json'))]
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"output": str(out), "source_revision": revision, "runs": [r["name"] for r in runs]}))
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prior-campaign", type=Path, required=True)
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--milestone", type=int, default=250)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--preflight-updates", type=int, default=5)
    p.add_argument("--workers", type=int, default=24)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
