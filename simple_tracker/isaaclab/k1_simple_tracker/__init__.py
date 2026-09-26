"""Simple multi-clip motion tracker for Booster K1 (Isaac Lab task registration)."""

import gymnasium as gym

gym.register(
    id="K1-Simple-Tracker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1SimpleTrackerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:K1SimpleTrackerPPORunnerCfg",
    },
)

gym.register(
    id="K1-Simple-Tracker-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1SimpleTrackerPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:K1SimpleTrackerPPORunnerCfg",
    },
)
