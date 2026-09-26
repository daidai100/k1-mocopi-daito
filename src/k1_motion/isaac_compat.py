"""Adapter for the pinned Booster task vs the newer pinned K1 asset naming.

Call after AppLauncher. This does not patch the upstream checkout or global env.
The same explicit PD values and joint effort limits as MuJoCo are used here.
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .contracts import JOINT_NAMES


def make_robot_cfg(root):
    from isaaclab.actuators import IdealPDActuatorCfg
    from booster_train.assets.robots.booster import BOOSTER_K1_CFG

    root = Path(root)
    config = json.loads((root / "configs/k1.json").read_text())
    xml = ET.parse(root / config["model"])
    motors = {n.get("joint"): float(n.get("forcerange").split()[1]) for n in xml.findall(".//motor")}
    armatures = {n.get("name"): float(n.get("armature", 0)) for n in xml.findall(".//joint")}
    cfg = BOOSTER_K1_CFG.copy()
    if config.get("shared_collision_contract"):
        from .simulator_assets import shared_collision_urdf

        cfg.spawn.asset_path = str(shared_collision_urdf(root, root / config["model"]))
    cfg.init_state.pos = (0.0, 0.0, 0.5435)
    cfg.init_state.joint_pos = dict(zip(JOINT_NAMES, config["neutral"]))
    cfg.actuators = {
        "whole_body": IdealPDActuatorCfg(
            joint_names_expr=list(JOINT_NAMES),
            effort_limit=motors,
            effort_limit_sim=motors,
            velocity_limit=dict(zip(JOINT_NAMES, config["velocity_limit"])),
            velocity_limit_sim=dict(zip(JOINT_NAMES, config["velocity_limit"])),
            stiffness=dict(zip(JOINT_NAMES, config["kp"])),
            damping=dict(zip(JOINT_NAMES, config["kd"])),
            armature={n: armatures[n] for n in JOINT_NAMES},
        )
    }
    return cfg
