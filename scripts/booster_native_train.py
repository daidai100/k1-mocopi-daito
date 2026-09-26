#!/usr/bin/env python3
"""Run pinned Booster Train's K1 BeyondMimic task with an explicit settings receipt."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for source in ("src", "third_party/booster_assets/src", "third_party/booster_train/source/booster_train"):
    sys.path.insert(0, str(ROOT / source))

from isaaclab.app import AppLauncher  # noqa: E402 - project paths precede Isaac imports

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, required=True)
parser.add_argument("--iterations", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--diagnose-reset", action="store_true")
args = parser.parse_args()
app = AppLauncher(headless=True, enable_cameras=False).app
try:
    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import booster_train.tasks  # noqa: F401 - register native task
    from booster_train.tasks.manager_based.beyond_mimic.robots.k1.mj_dance_004.env_cfg import (
        FlatWoStateEstimationEnvCfg,
    )
    from booster_train.tasks.manager_based.beyond_mimic.robots.k1.mj_dance_004.ppo_cfg import PPORunnerCfg
    from k1_motion.isaac_compat import make_robot_cfg
    from k1_motion.booster_command import ResetSafeMotionCommand

    motion = args.motion.resolve()
    if not motion.is_file() or args.num_envs < 1 or args.iterations < 1:
        raise ValueError("Expected converted motion and positive training counts")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env_cfg = FlatWoStateEstimationEnvCfg()
    # The pinned Booster Train task names an older K1 articulation. Map every
    # tracked body onto the pinned Booster Assets K1 used by this repository.
    body_names = {
        "Trunk": "trunk", "Head_2": "aahead_pitch_link",
        "Left_Hip_Roll": "left_hip_roll_link", "Left_Shank": "left_knee_pitch_link",
        "left_foot_link": "left_ankle_roll_link",
        "Right_Hip_Roll": "right_hip_roll_link", "Right_Shank": "right_knee_pitch_link",
        "right_foot_link": "right_ankle_roll_link",
        "Left_Arm_2": "left_shoulder_roll_link", "Left_Arm_3": "left_elbow_pitch_link",
        "left_hand_link": "left_elbow_yaw_link",
        "Right_Arm_2": "right_shoulder_roll_link", "Right_Arm_3": "right_elbow_pitch_link",
        "right_hand_link": "right_elbow_yaw_link",
    }
    env_cfg.scene.robot = make_robot_cfg(ROOT).replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.actions.joint_pos.scale = 0.25
    env_cfg.commands.motion.anchor_body_name = "trunk"
    env_cfg.commands.motion.class_type = ResetSafeMotionCommand
    env_cfg.commands.motion.body_names = [body_names[name] for name in env_cfg.commands.motion.body_names]
    env_cfg.events.base_com.params["asset_cfg"].body_names = "trunk"
    env_cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
    ]
    for term in (env_cfg.rewards.motion_foot_ori, env_cfg.rewards.motion_hand_ori,
                 env_cfg.rewards.motion_foot_pos, env_cfg.rewards.motion_hand_pos,
                 env_cfg.rewards.motion_trunk_ori, env_cfg.rewards.motion_trunk_pos,
                 env_cfg.terminations.ee_body_pos):
        term.params["body_names"] = [body_names[name] for name in term.params["body_names"]]
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands.motion.motion_file = str(motion)
    env_cfg.commands.motion.tail_len = 0
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    env_cfg.sim.dt = 0.002
    env_cfg.decimation = 10
    env_cfg.sim.render_interval = 10
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    cfg = PPORunnerCfg()
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.max_iterations = args.iterations
    cfg.num_steps_per_env = 32
    cfg.save_interval = 25
    cfg.experiment_name = "k1_native_beyondmimic"
    cfg.empirical_normalization = True
    cfg.policy.actor_hidden_dims = [512, 256]
    cfg.policy.critic_hidden_dims = [512, 256]
    cfg.policy.init_noise_std = 0.4
    cfg.algorithm.num_learning_epochs = 4
    batch = args.num_envs * cfg.num_steps_per_env
    if batch % 4096:
        raise ValueError("num-envs * 32 must be divisible by 4096 for the requested minibatch")
    cfg.algorithm.num_mini_batches = batch // 4096
    cfg.algorithm.learning_rate = 1e-5
    cfg.algorithm.gamma = 0.99
    cfg.algorithm.lam = 0.95
    cfg.algorithm.clip_param = 0.2
    cfg.algorithm.desired_kl = 0.02
    receipt = {
        "framework": "Booster Train native K1 BeyondMimic + RSL-RL",
        "task": "Booster-K1-MJ_Dance_004-v0", "motion": str(motion),
        "num_envs": args.num_envs, "iterations": args.iterations, "seed": args.seed,
        "physics_dt": env_cfg.sim.dt, "control_dt": env_cfg.sim.dt * env_cfg.decimation,
        "horizon": cfg.num_steps_per_env, "epochs": cfg.algorithm.num_learning_epochs,
        "minibatch": batch // cfg.algorithm.num_mini_batches,
        "learning_rate": cfg.algorithm.learning_rate, "actor_hidden_dims": cfg.policy.actor_hidden_dims,
        "critic_hidden_dims": cfg.policy.critic_hidden_dims,
        "motion_sampling": "single_clip_native_adaptive_phase", "checkpoint_compatible_with_k1_motion": False,
        "body_name_adapter": body_names, "robot_cfg": "k1_motion.isaac_compat.make_robot_cfg",
    }
    (output / "settings.json").write_text(json.dumps(receipt, indent=2) + "\n")
    env = gym.make(receipt["task"], cfg=env_cfg)
    try:
        if args.diagnose_reset:
            env.reset()
            command = env.unwrapped.command_manager.get_term("motion")
            error = (command.body_pos_relative_w - command.robot_body_pos_w).abs()[..., 2]
            names = command.cfg.body_names
            report = {name: {"mean_z_error_m": error[:, index].mean().item(),
                             "max_z_error_m": error[:, index].max().item(),
                             "target_z_first": command.body_pos_relative_w[0, index, 2].item(),
                             "robot_z_first": command.robot_body_pos_w[0, index, 2].item()}
                      for index, name in enumerate(names)}
            (output / "reset_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        else:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg.clip_actions)
            runner = OnPolicyRunner(wrapped, cfg.to_dict(), log_dir=str(output), device=cfg.device)
            runner.learn(num_learning_iterations=cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()
finally:
    app.close()
