"""Freeze a matched three-setting survival-position-v2 benchmark on the server."""

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prior-bundle", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--initializer", type=Path, required=True)
    args = p.parse_args()
    root = Path(os.environ.get("K1_MOTION_ROOT", Path(__file__).resolve().parents[1]))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    prior = args.prior_bundle.resolve()
    snapshot, revision = freeze_source(root)
    (out / "src").symlink_to(snapshot, target_is_directory=True)
    shutil.copytree(root / "scripts", out / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(prior / "inputs", out / "inputs")
    if (out / "inputs/library").is_symlink():
        (out / "inputs/library").unlink()
    if not (out / "inputs/library").exists():
        (out / "inputs/library").symlink_to(prior / "library", target_is_directory=True)
    shutil.copy2(args.initializer, out / "initialize.pt")
    shutil.copy2(prior / "controller.json", out / "controller.json")
    shutil.copy2(root / "configs/shuffled-scenes-120s.json", out / "scenes.json")
    sys.path.insert(0, str(snapshot))
    import torch
    from k1_motion.scene_transitions import transition_contract
    from k1_motion.world_objective import WorldBodyTracking, SafetyObjective

    saved = torch.load(out / "initialize.pt", map_location="cpu", weights_only=True)
    if saved["actor_size"] != 1680 or saved["critic_size"] != 1820 or saved.get("action_chunk"):
        raise ValueError("Expected original single-step preview actor")
    rows = [json.loads(x) for x in (prior / "library/index.jsonl").read_text().splitlines()]
    if len(rows) != 2751 or any(r["split"] != "train" or r["is_mirror"] for r in rows):
        raise ValueError("Unexpected long-original corpus")
    scene = transition_contract(json.loads((out / "scenes.json").read_text()), 0.02)
    gamma = math.exp(-0.02 / 60.0)
    variants = [
        ("position_baseline", {}),
        ("velocity_emphasis", {"weights": {"root_xy_position": 5.0, "root_velocity": 6.0}}),
        ("position_catchup", {"weights": {"root_xy_error": 1.0}}),
    ]
    runs = []
    for name, settings in variants:
        run = build_commands(
            out,
            out / "scripts",
            prior / "library",
            args.cache.resolve(),
            out / "initialize.pt",
            updates=2000,
            num_envs=2048,
            milestone=1000,
            seed=45,
            preflight_updates=3,
        )[0]
        run.update(name=name, backend="mujoco_cpp")
        settings_path = out / (name + "-reward.json")
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        for key in ("command", "preflight_command"):
            cmd = run[key]
            for flag, value in [("--backend", "mujoco_cpp"), ("--reward-profile", "survival-position-v2")]:
                cmd[cmd.index(flag) + 1] = value
            i = cmd.index("--curriculum-manifest")
            del cmd[i : i + 2]
            cmd[:] = [x.replace(str(out / "small"), str(out / name)) for x in cmd]
            cmd.extend(
                [
                    "--world-reward-settings",
                    str(settings_path),
                    "--scene-transitions",
                    str(out / "scenes.json"),
                    "--gamma",
                    str(gamma),
                    "--gae-lambda",
                    ".99",
                    "--cpu-workers",
                    "20",
                    "--cpu-chunk-size",
                    "4",
                ]
            )
        reward = dict(
            self_collision_weight=0.0,
            safety=SafetyObjective().contract,
            spatial_tracking=WorldBodyTracking("survival-position-v2", settings).contract,
            profile="survival-position-v2",
        )
        run["expected_training_contract"] = dict(
            reward_settings=reward, scene_transitions=scene, gamma=gamma, gae_lambda=0.99
        )
        runs.append(run)
    plan = dict(
        version="translation-rewards-20260925-v1",
        output=str(out),
        scripts=str(out / "scripts"),
        source_revision=revision,
        frozen_source=str(snapshot / "k1_motion"),
        initializer=str(out / "initialize.pt"),
        initializer_sha256=hashlib.sha256((out / "initialize.pt").read_bytes()).hexdigest(),
        library_manifest_sha256=hashlib.sha256((prior / "library/index.jsonl").read_bytes()).hexdigest(),
        originals=len(rows),
        by_family=dict(Counter(r["family"] for r in rows)),
        updates=2000,
        num_envs=2048,
        horizon=32,
        preflight_updates=3,
        milestone=1000,
        evaluation_workers=4,
        seed=45,
        backend="mujoco_cpp",
        training_physics="native MuJoCo float64, 500 Hz, 20 CPU workers",
        learner="R9700 HIP_VISIBLE_DEVICES=0",
        require_full_coverage=False,
        curriculum=None,
        sampling="fixed shuffled scene roles; actual exposure logged",
        gamma=gamma,
        gae_lambda=0.99,
        runs=runs,
        behaviorally_accepted=False,
        hardware_verified=False,
        confirmation_panel_used=False,
    )
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (out / "status.json").write_text(
        json.dumps(dict(phase="queued", runs={r["name"]: {"phase": "queued"} for r in runs}), indent=2) + "\n"
    )
    print(json.dumps(dict(output=str(out), source_revision=revision, runs=[r["name"] for r in runs])))


if __name__ == "__main__":
    main()
