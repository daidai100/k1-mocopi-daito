"""Versioned retarget-only speed bounds; physical robot limits stay unchanged."""
import numpy as np

LEGACY_SPEED_PROFILE = "legacy-command-v1"
OFFICIAL80_SPEED_PROFILE = "official-80-v1"


def retarget_speed_contract(robot, profile=None):
    profile = profile or LEGACY_SPEED_PROFILE
    nominal = np.asarray(robot.velocity_limit, dtype=float)
    if nominal.shape != (22,) or not np.isfinite(nominal).all() or np.any(nominal <= 0):
        raise ValueError("Invalid nominal joint limits for retarget speed profile")
    if profile == LEGACY_SPEED_PROFILE:
        limits = np.minimum(nominal, robot.config["command_velocity_limit"])
    elif profile == OFFICIAL80_SPEED_PROFILE:
        limits = .8 * nominal
    else:
        raise ValueError(f"Unknown retarget speed profile: {profile}")
    return {
        "version": "k1-retarget-speed-contract-v1", "profile": profile,
        "model_signature": robot.signature,
        "nominal_joint_velocity_limits_rad_s": nominal.tolist(),
        "joint_velocity_limits_rad_s": limits.tolist(),
        "scope": "Retarget pose-step bounds only; physical model and actuator limits unchanged",
    }


def retarget_speed_metadata(robot, base_version, profile=None):
    contract = retarget_speed_contract(robot, profile)
    if contract["profile"] == LEGACY_SPEED_PROFILE:
        return {"retarget_version": base_version}
    return {"retarget_version": f"{base_version}+{contract['profile']}",
            "retarget_speed_contract": contract, "model_signature": robot.signature}
