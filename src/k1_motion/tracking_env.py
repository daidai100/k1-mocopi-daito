"""One motion-conditioned task and shared actor contract across physics backends."""

import copy
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from .contracts import JOINT_NAMES
from .learning import MotionLibrary
from .observations import History, frame_tensor, quat_apply, quat_inv, targets_tensor, observation_contract
from .robot import K1Model, ROOT, SITES
from .math3d import rotation
from .actuation import (action_settings as resolve_action_settings, command_velocity_limits,
                        target_velocities_tensor, position_guard_tensor)
from .servo import gains, step_pd

TASK_VERSION = "relative-body-tracking-v3"
LOW_SUPPORT_TASK_VERSION = "relative-body-tracking-v3-low-support-v1"

# Gather only fields consumed by this phase. Packed reference gathering launches
# device kernels too; fetching all reward/reset fields for an actor command is
# wasted work. These sets do not change which reference timestamps are visible.
COMMAND_KEYS = ("root_orientation", "root_velocity", "joint_position", "joint_velocity", "contacts", "age")
OBSERVATION_KEYS = (*COMMAND_KEYS, "root_position", "valid", "contact_confidence")
RESET_KEYS = ("root_position", "root_orientation", "root_velocity", "joint_position", "joint_velocity",
              "reset_height_offset")
REWARD_KEYS = ("joint_position", "root_position", "root_orientation", "landmarks", "body_orientation",
               "landmark_velocity", "minimum_tracking_height", "minimum_tracking_upright")


class MujocoPhysics:
    def __init__(self, spec, num_envs, device, action_settings=None):
        self.spec, self.num_envs, self.device = spec, num_envs, device
        self.action_settings = action_settings
        from .actuators import SAFETY_FIELDS, actuator_contract, configure_robot_actuators
        self.safety_per_env = {key: torch.zeros(num_envs, device=device) for key in SAFETY_FIELDS}
        self.contract = {"backend": "scalar-mujoco", "physics_dt": float(spec.model.opt.timestep),
                         "actuator": actuator_contract(spec, action_settings)}
        self.robots = []
        for _ in range(num_envs):
            robot = copy.copy(spec)
            robot.data = mujoco.MjData(spec.model)
            configure_robot_actuators(robot, action_settings)
            robot.reset()
            self.robots.append(robot)

    def reset(self, ids, ref):
        for value in self.safety_per_env.values():
            value[ids] = 0
        ids = ids.cpu().tolist()
        for i, index in enumerate(ids):
            robot = self.robots[index]
            mujoco.mj_resetData(robot.model, robot.data)
            robot.data.qpos[:] = np.concatenate(
                [ref[k][i].cpu().numpy() for k in ("root_position", "root_orientation", "joint_position")]
            )
            # Reference initialization is a declared episode reset, never used in evaluation recovery.
            velocity = ref["root_velocity"][i].cpu().numpy()
            robot.data.qvel[:3] = velocity[:3]
            robot.data.qvel[3:6] = rotation(robot.data.qpos[3:7]).inv().apply(velocity[3:])
            robot.data.qvel[6:] = ref["joint_velocity"][i].cpu().numpy()
            mujoco.mj_forward(robot.model, robot.data)

    def step(self, target, velocity=None):
        velocities = [None] * self.num_envs if velocity is None else velocity.detach().cpu().numpy()
        metrics = [step_pd(r, q, v, settings=self.action_settings)
                   for r, q, v in zip(self.robots, target.detach().cpu().numpy(), velocities)]
        self.effort = float(np.mean([m["effort"] for m in metrics]))
        self.saturation = float(np.mean([m["effort_saturation"] for m in metrics]))
        self.effort_per_env = torch.tensor([m["effort"] for m in metrics], device=self.device)
        self.self_collision_per_env = torch.tensor(
            [float(m["self_collision"]) for m in metrics], device=self.device
        )
        for key in self.safety_per_env:
            self.safety_per_env[key] = torch.tensor([m["safety"][key] for m in metrics],
                                                   dtype=torch.float32, device=self.device)

    def state(self):
        # mj_step integrates qpos after computing site/body transforms. Publish
        # reward kinematics at the same timestamp as qpos without rerunning the
        # constraint solver or changing the last solved contact/effort metrics.
        for robot in self.robots:
            mujoco.mj_kinematics(robot.model, robot.data)

        def tensor(values):
            return torch.as_tensor(np.stack(values), dtype=torch.float32, device=self.device)

        return {
            "q": tensor([r.data.qpos[7:] for r in self.robots]),
            "dq": tensor([r.data.qvel[6:] for r in self.robots]),
            "orientation": tensor([r.data.qpos[3:7] for r in self.robots]),
            "omega": tensor([r.data.qvel[3:6] for r in self.robots]),
            "position": tensor([r.data.qpos[:3] for r in self.robots]),
            "velocity": tensor([r.data.qvel[:3] for r in self.robots]),
            "landmarks": tensor([r.landmarks() for r in self.robots]),
            "body_orientation": tensor([r.data.xquat[r.model.site_bodyid[r.site_ids]] for r in self.robots]),
        }

    def close(self):
        pass

    def foot_support(self):
        from .support import foot_support
        measured = [foot_support(r.model, r.data) for r in self.robots]
        return {key: torch.as_tensor(np.stack([row[key] for row in measured]), device=self.device,
                                    dtype=torch.bool if key == 'contact' else torch.float32)
                for key in measured[0]}


class IsaacPhysics:
    """Must be constructed after AppLauncher; GPU physics without step rendering."""

    def __init__(self, spec, num_envs, device, action_settings=None):
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.assets import AssetBaseCfg
        import isaaclab.sim as sim_utils
        from isaaclab.utils import configclass
        from .isaac_compat import make_robot_cfg

        self.spec, self.num_envs, self.device = spec, num_envs, device
        cfg = make_robot_cfg(ROOT)
        kp, kd = gains(spec, action_settings)
        cfg.actuators["whole_body"].stiffness = dict(zip(JOINT_NAMES, kp.tolist()))
        cfg.actuators["whole_body"].damping = dict(zip(JOINT_NAMES, kd.tolist()))

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path="/World/Ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=spec.config["ground_friction"],
                        dynamic_friction=spec.config["ground_friction"],
                    )
                ),
            )
            robot = cfg.replace(prim_path="{ENV_REGEX_NS}/K1")

        simulation_cfg = sim_utils.SimulationCfg(dt=spec.config["physics_dt"], device=str(device))
        if spec.config.get("shared_collision_contract"):
            simulation_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=spec.config["ground_friction"],
                dynamic_friction=spec.config["ground_friction"],
            )
        self.sim = sim_utils.SimulationContext(simulation_cfg)
        self.scene = InteractiveScene(SceneCfg(num_envs=num_envs, env_spacing=4.0, replicate_physics=True))
        self.robot = self.scene["robot"]
        self.sim.reset()
        materials = self.robot.root_physx_view.get_material_properties()
        before_materials = torch.unique(materials.reshape(-1, 3), dim=0).tolist()
        if spec.config.get("shared_collision_contract"):
            materials[..., :2] = spec.config["ground_friction"]
            materials[..., 2] = 0.0
            self.robot.root_physx_view.set_material_properties(
                materials, torch.arange(num_envs, device="cpu")
            )
        actual_materials = self.robot.root_physx_view.get_material_properties()
        self.contract = {
            "asset": cfg.spawn.asset_path,
            "materials_before_override": before_materials,
            "materials_static_dynamic_restitution": torch.unique(
                actual_materials.reshape(-1, 3), dim=0
            ).tolist(),
            "collision_shapes_per_robot": actual_materials.shape[1],
        }
        self.ids = [self.robot.joint_names.index(n) for n in JOINT_NAMES]
        if len(set(self.ids)) != 22:
            raise ValueError("Isaac K1 joint mapping is not bijective")
        self.origins = self.scene.env_origins
        self.saturation = self.effort = 0.0
        self.body_ids = [self.robot.body_names.index(body) for body, _ in SITES.values()]
        self.site_offsets = torch.tensor([offset for _, offset in SITES.values()], device=self.device)

    def reset(self, ids, ref):
        root = self.robot.data.default_root_state[ids].clone()
        root[:, :3] = ref["root_position"] + self.origins[ids]
        root[:, 3:7] = ref["root_orientation"]
        root[:, 7:] = ref["root_velocity"]
        self.robot.write_root_link_state_to_sim(root, env_ids=ids)
        q = self.robot.data.default_joint_pos[ids].clone()
        q[:, self.ids] = ref["joint_position"]
        dq = torch.zeros_like(q)
        dq[:, self.ids] = ref["joint_velocity"]
        self.robot.write_joint_state_to_sim(q, dq, env_ids=ids)
        self.scene.reset(ids)

    def step(self, target, velocity=None):
        velocity = torch.zeros_like(target) if velocity is None else velocity
        for _ in range(self.spec.substeps):
            self.robot.set_joint_position_target(target, joint_ids=self.ids)
            self.robot.set_joint_velocity_target(velocity, joint_ids=self.ids)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.spec.config["physics_dt"])
        torque = self.robot.data.applied_torque[:, self.ids]
        effort = torch.as_tensor(self.spec.effort, dtype=torch.float32, device=self.device)
        self.effort = (torque / effort).square().mean()
        self.effort_per_env = (torque / effort).square().mean(-1)
        self.saturation = (torque.abs() >= effort * 0.999).float().mean()

    def state(self):
        d = self.robot.data
        orientation = d.body_link_quat_w[:, self.body_ids]
        landmarks = d.body_link_pos_w[:, self.body_ids] + quat_apply(orientation, self.site_offsets)
        return {
            "q": d.joint_pos[:, self.ids].clone(),
            "dq": d.joint_vel[:, self.ids].clone(),
            "orientation": d.root_quat_w.clone(),
            "omega": d.root_ang_vel_b.clone(),
            "position": d.root_pos_w.clone() - self.origins,
            "velocity": d.root_link_lin_vel_w.clone(),
            "landmarks": landmarks - self.origins[:, None],
            "body_orientation": orientation.clone(),
        }

    def close(self):
        # Standalone AppLauncher owns Kit teardown. sim.stop() can synchronously
        # engage a renderer on this installed runtime even after headless stepping.
        pass


class TrackerEnv:
    def __init__(
        self,
        directory,
        num_envs=8,
        device="cuda:0",
        backend="mujoco",
        history=4,
        corruption=False,
        physics_options=None,
        self_collision_weight=0.0,
        library=None,
        action_settings=None,
        candidate_training=False,
        root_velocity_weight=None,
        root_velocity_sigma=None,
        reference_storage="padded",
        reward_profile="legacy",
        reference_cache=None,
        reference_scale=None,
        observation_profile="causal",
        tracking_huber=False,
        first_collision_penalty=0.,
        arm_workers=8,
        preview_horizon_s=.3,
        safety_profile=None,
        world_reward_settings=None,
        scene_transitions=None,
        command_feedback=False,
    ):
        from .tracking_rewards import REWARD_PROFILES
        if reward_profile not in REWARD_PROFILES:
            raise ValueError("Unknown tracking reward profile")
        if reward_profile != "legacy" and (root_velocity_weight is not None or root_velocity_sigma is not None):
            raise ValueError("Root velocity overrides apply only to the legacy reward")
        from .spatial_rewards import SERIES_PROFILES, SpatialTrackingReward
        from .simple_rewards import SCREEN_PROFILES, SimpleTrackingReward
        from .world_objective import WORLD_PROFILES, WorldBodyTracking, SafetyObjective
        if safety_profile not in (None, 'casual-safe-v1'):
            raise ValueError('Unknown safety profile')
        if safety_profile and (self_collision_weight or tracking_huber or first_collision_penalty):
            raise ValueError('Safety profile declares its common costs; extra costs are not allowed')
        if safety_profile and backend == 'isaac':
            raise ValueError('Safety profile requires MuJoCo substep measurements')
        if reward_profile in SERIES_PROFILES and (tracking_huber or first_collision_penalty):
            raise ValueError("Spatial reward profiles require their declared common costs")
        if reward_profile in ('spatial-s4-v1', 'causal-balanced-v1') and backend not in ('mujoco', 'mujoco_cpp', 'warp'):
            raise ValueError("Reward contact measurement requires a qualified MuJoCo backend")
        if not np.isfinite(self_collision_weight) or self_collision_weight < 0:
            raise ValueError("Self-collision reward weight must be finite and nonnegative")
        if (self_collision_weight or first_collision_penalty) and backend == "isaac":
            raise ValueError("Self-contact cost is qualified only for MuJoCo backends")
        self.device = torch.device(device)
        if observation_profile == 'preview' and corruption:
            raise ValueError('Preview requires a buffer-aware corruption model; legacy corruption is unsupported')
        self.spec = K1Model()
        from .scene_transitions import transition_contract
        self.scene_transition_contract = (transition_contract(scene_transitions, self.spec.control_dt)
                                          if scene_transitions is not None else None)
        if scene_transitions is not None and corruption:
            raise ValueError('Scene transitions require a buffer-aware corruption model; corruption is unsupported')
        self.action_settings = resolve_action_settings(self.spec, action_settings)
        from .actuation import active_action_mask
        self.action_mask = torch.tensor(active_action_mask(self.action_settings), device=self.device)
        self.command_feedback = command_feedback
        self.arm_feedback = None
        if self.action_settings.get("arm_collision_clearance", 0):
            from .collision_feedback import ArmCollisionFeedback
            if backend in ("mujoco_cpp", "warp"):
                from .parallel_collision_feedback import ParallelArmCollisionFeedback
                self.arm_feedback = ParallelArmCollisionFeedback(
                    self.spec, self.action_settings["arm_collision_clearance"],
                    (physics_options or {}).get("workers", 32) if backend == "mujoco_cpp" else arm_workers)
            else:
                self.arm_feedback = ArmCollisionFeedback(self.spec, self.action_settings["arm_collision_clearance"])
        self.command_step_limit = torch.tensor(
            command_velocity_limits(self.spec, self.action_settings) * self.spec.control_dt,
            device=self.device,
            dtype=torch.float32,
        )
        margin = np.asarray(self.action_settings.get('command_position_margin_rad', 0.))
        self.position_margin = (torch.tensor(margin, dtype=torch.float32, device=self.device)
                                if np.any(margin > 0) else None)
        self.num_envs, self.history_length = num_envs, history
        self.observation_profile = observation_profile
        self.preview_horizon_s = preview_horizon_s
        self.observation = observation_contract(history, observation_profile, preview_horizon_s, command_feedback)
        self.arm_workers = arm_workers
        self.backend_name = backend
        self.reward_settings = {"self_collision_weight": float(self_collision_weight)}
        self.safety_objective = SafetyObjective() if safety_profile else None
        if self.safety_objective:
            self.reward_settings['safety'] = self.safety_objective.contract
        self.spatial_reward = (SpatialTrackingReward(reward_profile, num_envs, self.device, self.spec.control_dt)
                               if reward_profile in SERIES_PROFILES else None)
        if reward_profile in SCREEN_PROFILES:
            self.spatial_reward = SimpleTrackingReward(reward_profile, num_envs, self.device, self.spec.control_dt)
        if reward_profile in WORLD_PROFILES:
            self.spatial_reward = WorldBodyTracking(reward_profile, settings=world_reward_settings)
        elif world_reward_settings is not None:
            raise ValueError('World reward settings require a world reward profile')
        if self.spatial_reward is not None:
            self.reward_settings['spatial_tracking'] = self.spatial_reward.contract
        from .motion_costs import TrackingCosts
        self.tracking_costs = TrackingCosts(num_envs, self.device, self.spec.control_dt,
            velocity_weight=2. if tracking_huber else 0., displacement_weight=1. if tracking_huber else 0.,
            first_collision_penalty=first_collision_penalty)
        self.extra_tracking_costs = bool(tracking_huber or first_collision_penalty)
        if tracking_huber:
            self.reward_settings["tracking_huber"] = {"version": 1, "velocity_weight": 2.,
                "displacement_weight": 1., "velocity_scale_m_s": .3, "displacement_scale_m": .15,
                "window_seconds": .5, "delta": 1., "time_scaled": True}
        if first_collision_penalty:
            self.reward_settings["first_collision_penalty"] = float(first_collision_penalty)
        if reward_profile != "legacy":
            self.reward_settings["profile"] = reward_profile
        for name, value, default in (("root_velocity_weight", root_velocity_weight, 0.5),
                                     ("root_velocity_sigma", root_velocity_sigma, 0.75)):
            if value is not None:
                if not np.isfinite(value) or value < 0 or (name.endswith("sigma") and value == 0):
                    raise ValueError("Invalid root velocity reward setting")
                if value != default:
                    self.reward_settings[name] = float(value)
        self.library_directory = str(directory)
        if reference_cache is not None:
            if library is not None:
                raise ValueError("Choose a shared library or a reference cache")
            from .reference_cache import load_reference_cache
            library = load_reference_cache(reference_cache, directory, self.spec, self.device,
                                           candidate_training, reference_storage)
        self.library = (
            library
            if library is not None
            else MotionLibrary(directory, self.spec, self.device, split="train",
                               candidate_training=candidate_training, storage=reference_storage)
        )
        if reference_scale is not None:
            from .reference_scale import scale_library
            self.reward_settings['reference_scale'] = scale_library(self.library, self.spec, reference_scale)
        if torch.device(self.library.device) != self.device or any(
            row.get("model_signature") != self.spec.signature for row in self.library.rows
        ):
            raise ValueError("Shared motion library device or robot contract differs")
        if backend == "warp":
            from .warp_physics import WarpPhysics

            self.physics = WarpPhysics(self.spec, num_envs, self.device,
                                       action_settings=self.action_settings, **(physics_options or {}))
        elif backend == "mujoco_cpp":
            from .cpu_physics import CpuParallelPhysics

            self.physics = CpuParallelPhysics(self.spec, num_envs, self.device,
                                              action_settings=self.action_settings, **(physics_options or {}))
        elif backend in ("isaac", "mujoco"):
            if physics_options:
                raise ValueError("Physics options are currently specific to MuJoCo Warp")
            self.physics = (IsaacPhysics if backend == "isaac" else MujocoPhysics)(
                self.spec, num_envs, self.device, action_settings=self.action_settings
            )
        else:
            raise ValueError(f"Unknown physics backend: {backend}")
        self.clips = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.frames = torch.zeros_like(self.clips)
        self.history = History(num_envs, history, self.device, self.observation["frame_size"])
        self.previous_action = torch.zeros((num_envs, 22), device=self.device)
        self.previous_target = torch.zeros_like(self.previous_action)
        self.previous_velocity = torch.zeros_like(self.previous_action)
        self.previous_guard_delta = torch.zeros_like(self.previous_action)
        self.last_command = {}
        self.limits = torch.tensor(self.spec.limits, dtype=torch.float32, device=self.device)
        self.neutral = torch.tensor(self.spec.neutral, dtype=torch.float32, device=self.device)
        self.corruption = corruption
        self.command_frames = torch.zeros_like(self.clips)
        self.episode_steps = torch.zeros_like(self.clips)
        self.episode_joint_error_auc = torch.zeros(num_envs, device=self.device)
        self.scene_steps = torch.zeros_like(self.clips)
        self.task_version = (LOW_SUPPORT_TASK_VERSION if any("low_pose_reference_audit" in r
                                                            for r in self.library.rows) else TASK_VERSION)
        self.previous_landmarks = torch.zeros((num_envs, 17, 3), device=self.device)
        self.previous_body_orientation = torch.zeros((num_envs, 17, 4), device=self.device)
        self.curriculum = None
        self.scene_transitions = None
        if self.scene_transition_contract is not None:
            from .scene_transitions import SceneTransitions
            from .shuffled_scenes import ShuffledScenes
            manager = (ShuffledScenes if self.scene_transition_contract['version'] == 'shuffled-scenes-v1'
                       else SceneTransitions)
            try:
                self.scene_transitions = manager(self, self.scene_transition_contract)
            except BaseException:
                self.close()
                raise

    def reference_frames(self, frames, keys=None):
        if self.scene_transitions is not None:
            return self.scene_transitions.frames(frames, keys)
        return self.library.frames(self.clips, frames, keys=keys)

    def reset(self, ids=None, clips=None, frames=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        if self.scene_transitions is not None:
            self.scene_transitions.reset(ids)
        self.clips[ids] = ((self.curriculum.sample_clips(len(ids)) if self.curriculum is not None
                           else self.library.sample(len(ids))) if clips is None else clips)
        shuffled = self.scene_transitions is not None and hasattr(self.scene_transitions, 'sample_starts')
        if shuffled and clips is None:
            self.clips[ids] = self.scene_transitions.sample_starts(ids)
        # Random reference-state initialization teaches clip interiors as well as starts.
        start = (
            torch.rand(len(ids), device=self.device) * (self.library.lengths[self.clips[ids]] - 2)
        ).long()
        # Starts are explicitly represented, alongside moving reference interiors.
        start = torch.where(torch.rand(len(ids), device=self.device) < 0.25, 0, start)
        if self.curriculum is not None and frames is None:
            start = self.curriculum.sample_frames(self.clips[ids])
        if shuffled:
            start.zero_()
        if frames is not None:
            start = frames
        self.frames[ids] = start
        self.command_frames[ids] = start
        ref = self.library.frames(self.clips[ids], self.frames[ids], keys=RESET_KEYS)
        if self.scene_transitions is not None:
            from .scene_transitions import transform_reference
            ref = transform_reference(ref, self.scene_transitions.yaw[ids], self.scene_transitions.shift[ids])
        reset_ref = {**ref, "root_position": ref["root_position"].clone()}
        # A reference can contain bounded residual foot penetration. Apply a
        # measured lift only to episode initialization, on every backend; the
        # reference, rewards and uninterrupted physical trajectory stay intact.
        reset_ref["root_position"][:, 2] += ref["reset_height_offset"]
        self.physics.reset(ids, reset_ref)
        self.previous_action[ids] = 0
        self.previous_target[ids] = ref["joint_position"]
        self.previous_velocity[ids] = 0
        self.previous_guard_delta[ids] = 0
        self.episode_steps[ids] = 0
        self.episode_joint_error_auc[ids] = 0
        self.scene_steps[ids] = 0
        if self.curriculum is not None:
            self.curriculum.reset_environments(ids)
        state = self.physics.state()
        self.tracking_costs.reset(ids, state["position"][ids], ref["root_position"])
        if self.spatial_reward is not None:
            self.spatial_reward.reset(ids, state['position'][ids], ref['root_position'])
        self.previous_landmarks[ids] = state["landmarks"][ids]
        self.previous_body_orientation[ids] = state["body_orientation"][ids]
        self.history.reset(ids)
        return self.observe()

    def observe(self):
        state = self.physics.state()
        actual = self.reference_frames(self.frames, keys=("landmarks", "root_position"))
        ref = self.reference_frames(self.command_frames, keys=OBSERVATION_KEYS)
        age = ref["age"] + (self.frames - self.command_frames).float() * self.spec.control_dt
        visible = frame_tensor(
            state["q"],
            state["dq"],
            state["orientation"],
            state["omega"],
            self.previous_action,
            ref,
            age,
            self.neutral,
            validate=False,  # Physics finite-state gate and learner checks stay enabled.
            profile=self.observation_profile, position=state["position"], velocity=state["velocity"],
            command_state=(self.previous_target, self.previous_velocity, self.previous_guard_delta)
                          if self.command_feedback else None,
        )
        causal = self.history.push(visible)
        if self.observation_profile == 'preview':
            from .observations import append_preview_tensor
            offsets = (5, 10, 15)
            preview = [self.reference_frames(self.command_frames+offset,
                keys=('joint_position', 'root_position', 'root_orientation', 'root_velocity', 'contacts', 'valid'))
                for offset in offsets]
            available = torch.stack([self.command_frames+offset < self.library.lengths[self.clips]
                                     for offset in offsets], dim=1)
            if self.preview_horizon_s == 0:
                available.zero_()
            causal = append_preview_tensor(causal, state['q'], state['orientation'],
                                           state['position'], preview, available)
        # Future references and contact truth remain privileged.
        future = [
            self.reference_frames(self.frames + offset, keys=("joint_position", "root_velocity"))
            for offset in (1, 5, 10)
        ]
        body_error = (actual["landmarks"] - actual["root_position"][:, None]) - (
            state["landmarks"] - state["position"][:, None]
        )
        privileged = torch.cat(
            [
                causal,
                quat_apply(quat_inv(state["orientation"]), state["velocity"]),
                state["position"][:, 2:3],
                actual["root_position"][:, 2:3] - state["position"][:, 2:3],
                body_error.flatten(1),
                *[frame["joint_position"] - state["q"] for frame in future],
                *[frame["root_velocity"] for frame in future],
            ],
            -1,
        )
        return causal, privileged

    def step(self, action, auto_reset=True, actuation_mask=None):
        action = action * self.action_mask
        command_trace = {}
        state = self.physics.state()
        ref = self.reference_frames(self.command_frames, keys=COMMAND_KEYS)
        source_tick = self.frames >= 0
        targets = targets_tensor(
            action, ref, state["orientation"], state["omega"], self.limits, self.action_settings["residual_scale"],
            self.action_settings.get("upper_body_residual_scale", 1.0),
            self.action_settings.get("imu_reference_rate_scale", 0.0),
            self.action_settings.get("ankle_prior_scale", 1.0), diagnostics=command_trace,
        )
        cap = self.command_step_limit
        targets = torch.maximum(
            torch.minimum(targets, self.previous_target + cap), self.previous_target - cap
        )
        command_trace['slew_target'] = targets
        fused_feedback = (self.arm_feedback is not None and actuation_mask is None
                          and self.position_margin is None
                          and getattr(self.physics, "fuse_arm_feedback", False))
        if self.arm_feedback is not None and not fused_feedback:
            targets = self.arm_feedback.project_batch(state["q"], state["dq"], targets)
            targets = targets.clamp(self.limits[:, 0], self.limits[:, 1])
            targets = torch.maximum(
                torch.minimum(targets, self.previous_target + cap), self.previous_target - cap
            )
        age = ref["age"] + (self.frames - self.command_frames).float() * self.spec.control_dt
        velocities = target_velocities_tensor(ref, cap / self.spec.control_dt,
                                              self.action_settings, age)
        command_trace.update(arm_target=targets, raw_velocity=velocities)
        if self.position_margin is not None:
            targets, velocities = position_guard_tensor(targets, velocities, self.previous_target,
                state['q'], self.limits, cap, self.position_margin)
        if actuation_mask is not None:
            # A finished diagnostic trial remains a recorded failure/completion.
            # Damping the terminal body avoids driving it through more poses on
            # the floor while other independent trials finish; it is never reset.
            targets = torch.where(actuation_mask[:, None], targets, state["q"])
            velocities = torch.where(actuation_mask[:, None], velocities, torch.zeros_like(velocities))
        if fused_feedback:
            targets = self.physics.step_projected(targets, velocities, self.previous_target, self.arm_feedback)
            command_trace['arm_target'] = targets
        else:
            self.physics.step(targets, velocities)
        command_trace.update(executed_target=targets, executed_velocity=velocities)
        self.last_command = {k: v.detach().clone() for k, v in command_trace.items()}
        state = self.physics.state()
        self.frames += 1
        self.episode_steps += 1
        self.scene_steps += source_tick
        if self.scene_transitions is not None:
            self.scene_transitions.episode_bridge_steps += ~source_tick
        beyondmimic = self.reward_settings.get("profile") == "beyondmimic-causal-v1"
        support_reward = self.reward_settings.get('profile') in ('spatial-s4-v1', 'causal-balanced-v1')
        actual = self.reference_frames(self.frames,
            keys=(*REWARD_KEYS, "root_velocity", *(("contacts", "contact_confidence", "joint_velocity")
                                                  if support_reward else ())))
        qerr = (state["q"] - actual["joint_position"]).square().mean(-1)
        zerr = (state["position"][:, 2] - actual["root_position"][:, 2]).square()
        angle = 2 * (state["orientation"] * actual["root_orientation"]).sum(-1).abs().clamp(max=1).acos()
        body_difference = (state["landmarks"] - state["position"][:, None]) - (
            actual["landmarks"] - actual["root_position"][:, None]
        )
        body_error = body_difference.square().sum(-1).mean(-1)
        landmark_velocity = (state["landmarks"] - self.previous_landmarks) / self.spec.control_dt
        smooth = ((action - self.previous_action).square()*self.action_mask).sum(-1)/self.action_mask.sum()
        if self.spatial_reward is not None:
            tracking, components = self.spatial_reward.step(state, actual, landmark_velocity,
                self.physics.foot_support() if support_reward else None)
            reward = (tracking - 0.1*smooth - 0.02*self.physics.effort_per_env) * self.spec.control_dt
        elif beyondmimic:
            from .tracking_rewards import beyondmimic_causal_tracking
            previous_reference = self.reference_frames(self.frames-1, keys=("body_orientation",))
            tracking, components = beyondmimic_causal_tracking(
                state, actual, self.previous_body_orientation, previous_reference["body_orientation"],
                landmark_velocity, self.spec.control_dt)
            reward = (tracking - 0.1*smooth - 0.02*self.physics.effort_per_env) * self.spec.control_dt
        else:
            verr = (state["velocity"] - actual["root_velocity"][:, :3]).square().sum(-1)
            feet_error = body_difference[:, [11, 15]].square().sum(-1).mean(-1)
            body_angle = (
                2 * (state["body_orientation"] * actual["body_orientation"]).sum(-1).abs().clamp(max=1).acos()
            )
            body_velocity_error = (landmark_velocity - actual["landmark_velocity"]).square().sum(-1).mean(-1)
            components = {
                "joint": (-qerr / 0.3**2).exp(),
                "height": (-zerr / 0.08**2).exp(),
                "orientation": (-angle.square() / 0.4**2).exp(),
                "body_position": (-body_error / 0.12**2).exp(),
                "feet_position": (-feet_error / 0.08**2).exp(),
                "body_orientation": (-body_angle.square().mean(-1) / 0.5**2).exp(),
                "velocity": (-verr / self.reward_settings.get("root_velocity_sigma", 0.75)**2).exp(),
                "body_velocity": (-body_velocity_error / 1.5**2).exp(),
            }
            reward = (
                0.5 * components["joint"]
                + components["height"]
                + components["orientation"]
                + components["body_position"]
                + components["feet_position"]
                + 0.5 * components["body_orientation"]
                + self.reward_settings.get("root_velocity_weight", 0.5) * components["velocity"]
                + 0.5 * components["body_velocity"]
                - 0.1 * smooth
                - 0.02 * self.physics.effort_per_env
            ) * self.spec.control_dt
        collision = getattr(self.physics, "self_collision_per_env", None)
        if self.safety_objective is not None:
            cost, safety_parts = self.safety_objective.step(
                collision, getattr(self.physics, 'safety_per_env', None))
            reward -= cost*self.spec.control_dt
            components.update(safety_parts)
            for name, value in self.physics.safety_per_env.items():
                components['measured_safety/'+name] = value
        if self.reward_settings["self_collision_weight"]:
            reward -= self.reward_settings["self_collision_weight"] * collision * self.spec.control_dt
        if self.extra_tracking_costs:
            cost, cost_components = self.tracking_costs.step(state["position"], state["velocity"],
                actual["root_position"], actual["root_velocity"][:, :3],
                torch.zeros_like(reward) if collision is None else collision)
            reward -= cost
            components.update(cost_components)
        up = torch.zeros_like(state["omega"])
        up[:, 2] = 1
        upright = quat_apply(state["orientation"], up)[:, 2]
        fallen = ((state["position"][:, 2] < actual["minimum_tracking_height"])
                  | (upright < actual["minimum_tracking_upright"]))
        finite = torch.isfinite(torch.cat([v.flatten(1) for v in state.values()], -1)).all(-1)
        fallen |= ~finite
        tracking_failed = (angle > 1.0) | (zerr > 0.2**2) | (body_error > 0.3**2)
        tracking_failed &= self.episode_steps > 5
        survival = self.reward_settings.get('profile') in ('survival-position-v1', 'survival-position-v2')
        if survival and self.spatial_reward.contract.get('tracking_termination') is False:
            components['tracking/legacy_violation'] = tracking_failed.float()
            # Keep sampling recovery from every finite upright state. Tracking
            # error affects the dense objective, never an error gate or timer.
            tracking_failed = torch.zeros_like(fallen)
        failed = fallen | tracking_failed
        if survival:
            objective = self.spatial_reward.contract
            alive_reward = (~failed).float()*objective['survival_per_second']*self.spec.control_dt
            penalty = torch.where(fallen, objective['fall_penalty'],
                torch.where(tracking_failed, objective['tracking_failure_penalty'], 0.))
            reward = torch.nan_to_num(reward, nan=0., posinf=0., neginf=0.)+alive_reward-penalty
            self.episode_joint_error_auc += torch.nan_to_num(
                components['joint_mae_rad'], nan=0., posinf=0.)*self.spec.control_dt
            components.update(survival_per_second=alive_reward/self.spec.control_dt,
                              failure_event_penalty=penalty)
        else:
            reward[failed] -= self.safety_objective.contract['failure_penalty'] if self.safety_objective else 0.3
        self.previous_action.copy_(action)
        self.previous_target.copy_(targets)
        self.previous_velocity.copy_(velocities)
        self.previous_guard_delta.copy_(targets-command_trace['raw_target'])
        self.previous_landmarks.copy_(state["landmarks"])
        self.previous_body_orientation.copy_(state["body_orientation"])
        clip_end = self.frames >= self.library.lengths[self.clips] - 1
        transitioned = torch.zeros_like(clip_end)
        episode_limit = torch.zeros_like(clip_end)
        pending = []
        if self.scene_transitions is not None:
            episode_limit = self.episode_steps >= self.scene_transitions.episode_steps
            eligible = clip_end & ~failed & ~episode_limit
            if actuation_mask is not None:
                eligible &= actuation_mask
            pending = self.scene_transitions.prepare(eligible)
            transitioned = self.scene_transitions.pending_mask(pending)
        done = failed | episode_limit | (clip_end & ~transitioned)
        if auto_reset:
            if self.scene_transitions is None:
                self.library.record_steps(self.clips, self.episode_steps, done)
            else:
                self.library.record_steps(self.clips[source_tick], self.scene_steps[source_tick],
                                          (done | clip_end)[source_tick])
            if self.curriculum is not None:
                signals = None
                if self.curriculum.fidelity_phases:
                    safety = getattr(self.physics, 'safety_per_env', {})
                    if collision is None or any(k not in safety for k in
                            ('joint_limit_fraction', 'operating_speed_fraction')):
                        raise ValueError('Fidelity phase curriculum requires measured safety signals')
                    signals = dict(
                        root_velocity_error_squared=(state['velocity']-actual['root_velocity'][:, :3]).square().sum(-1),
                        world_position_error_squared=(state['landmarks']-actual['landmarks']).square().sum(-1).mean(-1),
                        collision=collision, joint_limit_fraction=safety['joint_limit_fraction'],
                        operating_speed_fraction=safety['operating_speed_fraction'])
                self.curriculum.record(self.clips, self.frames, failed, signals=signals,
                    episode_steps=self.episode_steps, done=done, clip_end=clip_end,
                    source_mask=source_tick, scene_steps=self.scene_steps)
                if self.scene_transitions is not None:
                    bridge_end = (~source_tick & (self.frames == 0)).nonzero(as_tuple=False).flatten()
                    self.curriculum.reset_environments(bridge_end)
        # Explicit engineering assumptions: 0-40ms delay and 5% single-frame loss.
        # This never grants future observations and is not a model of Sony capture latency.
        if self.corruption:
            candidate = (self.frames - torch.randint(0, 3, self.frames.shape, device=self.device)).clamp(
                min=0
            )
            candidate = torch.maximum(candidate, self.command_frames)
            self.command_frames = torch.where(
                torch.rand(self.num_envs, device=self.device) < 0.05, self.command_frames, candidate
            )
        else:
            self.command_frames.copy_(self.frames)
        info = {
            "falls": fallen.sum(),
            "tracking_failures": tracking_failed.sum(),
            "completed": (clip_end & ~failed).sum(),
            "ended_episode_steps": (self.episode_steps * done).sum(),
            "ended_episodes": done.sum(),
            "joint_rmse_rad": qerr.mean().sqrt(),
            "body_rmse_m": body_error.mean().sqrt(),
            "orientation_error_rad": angle.mean(),
            "effort_saturation": self.physics.saturation,
            "corruption_assumptions": self.corruption,
            "reward_components": {k: v.mean() for k, v in components.items()},
        }
        world_error = (state['landmarks']-actual['landmarks']).square().sum(-1)
        from .actuation import command_metrics
        info['command_metrics'] = command_metrics(command_trace)
        info['reward_components']['diagnostic/world_body_rmse_m'] = world_error.mean().sqrt()
        if collision is not None:
            info["self_collision_fraction"] = collision.mean()
        if self.scene_transitions is not None:
            info.update(scene_transition_starts=transitioned.sum(),
                        scene_transition_completions=(~source_tick & (self.frames == 0) & ~failed).sum(),
                        scene_transition_failures=(~source_tick & failed).sum(),
                        scene_transition_unavailable=(clip_end & ~failed & ~episode_limit & ~transitioned).sum(),
                        scene_episode_limits=(episode_limit & ~failed).sum(),
                        ended_episode_bridge_steps=(self.scene_transitions.episode_bridge_steps*done).sum(),
                        scene_bridge_steps=(~source_tick).sum())
        if survival:
            info.update(ended_joint_error_auc_rad_s=(self.episode_joint_error_auc*done).sum(),
                        ended_joint_error_mean_rad=(self.episode_joint_error_auc*done).sum()
                            / ((self.episode_steps*done).sum()*self.spec.control_dt).clamp(min=self.spec.control_dt))
        self.last_step = {
            "failed": failed,
            "fallen": fallen,
            "clip_end": clip_end,
            "qerr": qerr,
            "body_error": body_error,
            "angle": angle,
            **({'survival_reward': alive_reward, 'failure_penalty': penalty} if survival else {}),
        }
        if self.scene_transitions is not None:
            self.scene_transitions.commit(pending)
        ids = done.nonzero(as_tuple=False).flatten() if auto_reset else []
        if len(ids):
            causal, privileged = self.reset(ids)
        else:
            causal, privileged = self.observe()
        return causal, privileged, reward, done, info

    def close(self):
        if hasattr(self, "evaluation_environment"):
            self.evaluation_environment.close()
        self.physics.close()
        if hasattr(self.arm_feedback, "close"):
            self.arm_feedback.close()


def add_upstream_paths():
    for path in ("third_party/booster_assets/src", "third_party/booster_train/source/booster_train"):
        sys.path.insert(0, str(Path(ROOT) / path))
