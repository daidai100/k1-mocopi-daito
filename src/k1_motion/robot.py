"""One explicitly named K1 model/actuator contract for retargeting and control."""

import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .contracts import JOINT_NAMES, LANDMARKS, Reference, RobotState

ROOT = Path(os.environ.get("K1_MOTION_ROOT", Path(__file__).resolve().parents[2]))
SITES = {
    "pelvis": ("trunk", (0, 0, 0)),
    "chest": ("trunk", (0, 0, 0.185)),
    "head": ("aahead_pitch_link", (0, 0, 0.08)),
    "left_shoulder": ("left_shoulder_roll_link", (0, 0, 0)),
    "left_elbow": ("left_elbow_yaw_link", (0, 0, 0)),
    "left_wrist": ("left_elbow_yaw_link", (0, 0.145, 0)),
    "right_shoulder": ("right_shoulder_roll_link", (0, 0, 0)),
    "right_elbow": ("right_elbow_yaw_link", (0, 0, 0)),
    "right_wrist": ("right_elbow_yaw_link", (0, -0.145, 0)),
    "left_hip": ("left_hip_roll_link", (0, 0, 0)),
    "left_knee": ("left_knee_pitch_link", (0, 0, 0)),
    "left_ankle": ("left_ankle_roll_link", (0, 0, 0)),
    "left_toe": ("left_ankle_roll_link", (0.095, 0, -0.02)),
    "right_hip": ("right_hip_roll_link", (0, 0, 0)),
    "right_knee": ("right_knee_pitch_link", (0, 0, 0)),
    "right_ankle": ("right_ankle_roll_link", (0, 0, 0)),
    "right_toe": ("right_ankle_roll_link", (0.095, 0, -0.02)),
}


class K1Model:
    def __init__(self, config_path=None):
        path = Path(config_path or ROOT / "configs/k1.json")
        self.config = json.loads(path.read_text())
        model_path = Path(self.config["model"])
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"K1 assets missing: run python scripts/setup_assets.py ({model_path})")
        self.model_path = model_path
        tree = ET.parse(model_path)
        xml = tree.getroot()
        xml.find("compiler").set("meshdir", str(model_path.parent / "meshes"))
        xml.find("option").set("timestep", str(self.config["physics_dt"]))
        xml.find("option").set("integrator", "implicitfast")
        ground = xml.find(".//geom[@name='ground']")
        ground.set("condim", "3")
        ground.set("friction", f"{self.config['ground_friction']} 0.005 0.0001")
        if self.config.get("shared_collision_contract"):
            # MuJoCo combines same-priority friction with max(), so changing only
            # the floor left the robot's default coefficient 1.0 in effect.
            for geom in xml.findall(".//geom"):
                geom.set("friction", f"{self.config['ground_friction']} 0.005 0.0001")
        for name, (body, pos) in SITES.items():
            node = xml.find(f".//body[@name='{body}']")
            if node is None:
                raise ValueError(f"K1 body contract mismatch: {body}")
            ET.SubElement(
                node,
                "site",
                name=f"motion_{name}",
                pos=" ".join(map(str, pos)),
                size="0.008",
                rgba="0.2 0.7 1 1",
                group="3",
            )
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        names = tuple(self.model.joint(i).name for i in range(1, self.model.njnt))
        actuator_names = tuple(self.model.actuator(i).name for i in range(self.model.nu))
        if (
            names != JOINT_NAMES
            or actuator_names != JOINT_NAMES
            or (self.model.nq, self.model.nv) != (29, 28)
        ):
            raise ValueError(f"Unexpected K1 layout: {names}, actuators={actuator_names}")
        self.site_ids = np.array([self.model.site(f"motion_{name}").id for name in LANDMARKS])
        self.limits = self.model.jnt_range[1:].copy()
        self.effort = self.model.actuator_forcerange[:, 1].copy()
        self.neutral = np.array(self.config["neutral"], float)
        self.kp, self.kd = np.array(self.config["kp"]), np.array(self.config["kd"])
        self.velocity_limit = np.array(self.config["velocity_limit"])
        self.control_dt = float(self.config["control_dt"])
        self.substeps = round(self.control_dt / self.model.opt.timestep)
        if not np.isclose(self.substeps * self.model.opt.timestep, self.control_dt):
            raise ValueError("Control period must be an integer multiple of physics dt")
        for key in ("neutral", "kp", "kd", "velocity_limit"):
            value = np.asarray(self.config[key])
            if value.shape != (22,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid K1 {key}")
        self.signature = hashlib.sha256(
            model_path.read_bytes() + json.dumps(self.config, sort_keys=True).encode()
        ).hexdigest()
        self.data.qpos[:3] = [0, 0, 0.6]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[7:] = self.neutral
        mujoco.mj_forward(self.model, self.data)
        # Put the soles at the floor without an initial impact; ankle origin is 38mm above sole.
        self.data.qpos[2] += 0.038 - min(self.landmarks()[11, 2], self.landmarks()[15, 2])
        mujoco.mj_forward(self.model, self.data)
        self.neutral_qpos = self.data.qpos.copy()
        self.neutral_landmarks = self.landmarks()

    def landmarks(self, data=None):
        return (data or self.data).site_xpos[self.site_ids].copy()

    def reset(self, reference=None):
        mujoco.mj_resetData(self.model, self.data)
        if reference is None:
            self.data.qpos[:] = self.neutral_qpos
        else:
            self.data.qpos[:] = np.r_[
                reference.root_position, reference.root_orientation, reference.joint_position
            ]
        mujoco.mj_forward(self.model, self.data)

    def neutral_reference(self, t=0.0):
        return Reference(
            t,
            t,
            self.neutral_qpos[:3],
            self.neutral_qpos[3:7],
            np.zeros(6),
            self.neutral,
            np.zeros(22),
            self.neutral_landmarks,
            np.ones(2),
            np.ones(2),
        )

    def state(self, now=None):
        return RobotState(
            self.data.time if now is None else now,
            self.data.qpos[7:],
            self.data.qvel[6:],
            self.data.qpos[3:7],
            self.data.qvel[3:6],
            root_position=self.data.qpos[:3],
            linear_velocity=self.data.qvel[:3],
        )

    def step(self, targets, damping_only=False, settings=None):
        if settings or getattr(self, '_actuator_profile', None):
            from .servo import step_pd
            return step_pd(self, targets, damping_only=damping_only,
                           settings=settings or self._actuator_settings)
        from .actuation import action_settings, command_velocity_limits
        from .actuators import SAFETY_FIELDS, safety_sample, accumulate_safety
        operating_speed = command_velocity_limits(self, action_settings(self))
        safety = np.zeros(len(SAFETY_FIELDS))
        targets = np.asarray(targets)
        if targets.shape != (22,) or not np.isfinite(targets).all():
            raise ValueError("Invalid motor target")
        targets = np.clip(targets, self.limits[:, 0], self.limits[:, 1])
        saturation = 0.0
        effort_sum = 0.0
        self_collision = False
        for _ in range(self.substeps):
            torque = (
                0 if damping_only else self.kp * (targets - self.data.qpos[7:])
            ) - self.kd * self.data.qvel[6:]
            saturation += np.mean(np.abs(torque) >= self.effort)
            self.data.ctrl[:] = np.clip(torque, -self.effort, self.effort)
            effort_sum += float(np.mean((self.data.ctrl / self.effort) ** 2))
            mujoco.mj_step(self.model, self.data)
            accumulate_safety(safety, safety_sample(self.data.qpos[7:], self.data.qvel[6:], torque,
                              self.effort, self.limits, operating_speed, self.velocity_limit), self.substeps)
            self_collision |= any(
                c.dist <= 0 and 0 not in self.model.geom_bodyid[[c.geom1, c.geom2]]
                for c in self.data.contact[: self.data.ncon]
            )
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Nonfinite simulated robot state")
        return {
            "effort_saturation": saturation / self.substeps,
            "effort": effort_sum / self.substeps,
            "self_collision": self_collision,
            "safety": dict(zip(SAFETY_FIELDS, safety)),
        }

    def contact_metrics(self):
        feet = [self.model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
        contact = np.zeros(2)
        self_collision = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.dist > 0:
                continue
            bodies = self.model.geom_bodyid[[c.geom1, c.geom2]]
            if 0 in bodies:
                for j, foot in enumerate(feet):
                    contact[j] = max(contact[j], int(foot in bodies))
            else:
                self_collision += 1
        return contact, self_collision
