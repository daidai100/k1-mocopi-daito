#!/usr/bin/env python3
"""Run with an Isaac Lab interpreter. No rendering/cameras or robot communication.

Startup and asset compatibility are reported separately from physics/learning.
Use --output to retain the failure stage even when the runtime is incompatible.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=ROOT / "reports/isaac-preflight.json")
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--upstream-unmodified", action="store_true")
args = parser.parse_args()
report = {
    "backend": "isaaclab",
    "headless": True,
    "rendering": False,
    "stage": "imports",
    "passed": False,
    "physics_passed": False,
    "learning_passed": False,
    "versions": {},
    "python": sys.executable,
}
args.output.parent.mkdir(parents=True, exist_ok=True)


def save():
    args.output.write_text(json.dumps(report, indent=2) + "\n")


app = None
try:
    for package in ("torch", "isaacsim", "isaaclab"):
        report["versions"][package] = importlib.metadata.version(package)
    for path in ("src", "third_party/booster_assets/src", "third_party/booster_train/source/booster_train"):
        sys.path.insert(0, str(ROOT / path))
    from isaaclab.app import AppLauncher

    report["stage"] = "application_startup"
    save()
    start = time.monotonic()
    app = AppLauncher(headless=True, enable_cameras=False).app
    report["startup_seconds"] = time.monotonic() - start
    report["stage"] = "booster_task_import"
    save()
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from booster_train.assets.robots.booster import BOOSTER_K1_CFG

    report["gpu"] = torch.cuda.get_device_name(0)
    report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    report["stage"] = "robot_spawn"
    save()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.002, device="cuda:0"))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    if args.upstream_unmodified:
        cfg = BOOSTER_K1_CFG.replace(prim_path="/World/K1")
    else:
        from k1_motion.isaac_compat import make_robot_cfg

        cfg = make_robot_cfg(ROOT).replace(prim_path="/World/K1")
        report["compatibility_adapter"] = "k1_motion.isaac_compat.make_robot_cfg"
    robot = Articulation(cfg)
    sim.reset()
    report["joints"] = robot.joint_names
    report["stage"] = "physics"
    save()
    for _ in range(args.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())
        if not torch.isfinite(robot.data.root_state_w).all():
            raise ValueError("Nonfinite K1 physics state")
    report["physics_passed"] = True
    report["stage"] = "physics_complete_learning_not_tested"
    report["final_root_state"] = robot.data.root_state_w.cpu().tolist()
    report["passed"] = True
except Exception as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
    report["traceback"] = traceback.format_exc()
finally:
    save()
    print(json.dumps(report, indent=2), flush=True)
    if app:
        from k1_motion.isaac_shutdown import shutdown

        shutdown(app, 0 if report["passed"] else 1, args.output.with_suffix(".shutdown.json"))
if not report["passed"]:
    raise SystemExit(1)
