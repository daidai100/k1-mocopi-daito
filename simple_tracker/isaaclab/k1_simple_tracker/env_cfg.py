"""Environment config for the simple multi-clip K1 tracker.

Scene, robot, actuators (incl. 2-8 physics-step delay), domain randomisation and
terrain follow booster_train's K1 BeyondMimic setup so the sim2sim/sim2real path
through booster_deploy stays the same.  What changed is the command (library of
clips), the observations (no world-frame reference terms) and the rewards
(7 terms, fixed sigmas).
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from booster_train.assets.robots.booster import BOOSTER_K1_CFG, K1_ACTION_SCALE

from . import mdp

LIBRARY_FILE = os.environ.get(
    "K1_TRACKER_LIBRARY", os.path.expanduser("~/ws/k1-mocopi-data/k1_bandai_library.npz")
)

TRACKED_BODIES = [
    "Trunk",
    "Head_2",
    "Left_Hip_Roll",
    "Left_Shank",
    "left_foot_link",
    "Right_Hip_Roll",
    "Right_Shank",
    "right_foot_link",
    "Left_Arm_2",
    "Left_Arm_3",
    "left_hand_link",
    "Right_Arm_2",
    "Right_Arm_3",
    "right_hand_link",
]
END_EFFECTORS = ["left_hand_link", "right_hand_link", "left_foot_link", "right_foot_link"]

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(10.0, 10.0),
            border_width=20.0,
            num_rows=5,
            num_cols=10,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=False,
            sub_terrains={
                "nearly_flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.8, noise_range=(0.0, 0.005), noise_step=0.005, border_width=0.25
                ),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2, noise_range=(-0.015, 0.015), noise_step=0.005, border_width=0.25
                ),
            },
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0
    )


@configclass
class CommandsCfg:
    motion = mdp.MultiMotionCommandCfg(
        asset_name="robot",
        library_file=LIBRARY_FILE,
        anchor_body_name="Trunk",
        body_names=TRACKED_BODIES,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=True, scale=K1_ACTION_SCALE
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable inputs only.  Order is the deploy contract (see simple_tracker/README.md)."""

        # reference (54): available from a live retargeted mocopi stream
        ref_joint = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        ref_root_gravity = ObsTerm(
            func=mdp.ref_root_gravity_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.03, n_max=0.03)
        )
        ref_root_lin_vel = ObsTerm(
            func=mdp.ref_root_lin_vel_h, params={"command_name": "motion"}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        ref_root_ang_vel = ObsTerm(
            func=mdp.ref_root_ang_vel_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        ref_root_height = ObsTerm(
            func=mdp.ref_root_height, params={"command_name": "motion"}, noise=Unoise(n_min=-0.02, n_max=0.02)
        )
        # proprioception (72): IMU + joint encoders + previous action
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Asymmetric critic: privileged sim state, no noise."""

        ref_joint = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        ref_root_gravity = ObsTerm(func=mdp.ref_root_gravity_b, params={"command_name": "motion"})
        ref_root_lin_vel = ObsTerm(func=mdp.ref_root_lin_vel_h, params={"command_name": "motion"})
        ref_root_ang_vel = ObsTerm(func=mdp.ref_root_ang_vel_b, params={"command_name": "motion"})
        ref_root_height = ObsTerm(func=mdp.ref_root_height, params={"command_name": "motion"})
        ref_body_pos = ObsTerm(func=mdp.ref_body_pos_b, params={"command_name": "motion"})
        robot_body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        robot_root_height = ObsTerm(func=mdp.robot_root_height, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.3, 0.6),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """BeyondMimic's tracking terms minus the two world-frame anchor terms (7 terms total)."""

    body_pos = RewTerm(func=mdp.tracking_body_pos, weight=1.0, params={"command_name": "motion", "std": 0.3})
    body_ori = RewTerm(func=mdp.tracking_body_ori, weight=1.0, params={"command_name": "motion", "std": 0.4})
    body_lin_vel = RewTerm(func=mdp.tracking_body_lin_vel, weight=1.0, params={"command_name": "motion", "std": 1.0})
    body_ang_vel = RewTerm(
        func=mdp.tracking_body_ang_vel, weight=1.0, params={"command_name": "motion", "std": 3.14}
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"^(?!left_hand_link$)(?!right_hand_link$)(?!left_foot_link$)(?!right_foot_link$).+$"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    motion_end = DoneTerm(func=mdp.motion_end, params={"command_name": "motion"}, time_out=True)
    anchor_height = DoneTerm(func=mdp.bad_anchor_height, params={"command_name": "motion", "threshold": 0.25})
    anchor_gravity = DoneTerm(func=mdp.bad_anchor_gravity, params={"command_name": "motion", "threshold": 0.8})
    ee_height = DoneTerm(
        func=mdp.bad_body_height, params={"command_name": "motion", "threshold": 0.25, "body_names": END_EFFECTORS}
    )


@configclass
class K1SimpleTrackerEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


@configclass
class K1SimpleTrackerPlayEnvCfg(K1SimpleTrackerEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.commands.motion.play = True
        self.events.push_robot = None
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 60.0
