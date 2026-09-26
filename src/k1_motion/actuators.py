"""Versioned port of Booster Train's K1 simulated torque-speed envelope.

Source: BoosterRobotics/booster_train commit 651b7a53f2ffaf2d5629d0604d065cc385e29c6b,
assets/robots/{actuator,booster}.py. This ports peak/knee/speed/armature values and
the symmetric clipping formula, not its controller gains or randomized delay.
It is a manufacturer training model, not hardware qualification.
"""
import copy

import mujoco
import numpy as np

ACTUATOR_PROFILE = 'booster-train-k1-actuator-v1'
SOURCE_COMMIT = '651b7a53f2ffaf2d5629d0604d065cc385e29c6b'
SAFETY_FIELDS = ('operating_speed_fraction', 'operating_speed_max_ratio',
                 'nominal_speed_fraction', 'nominal_speed_max_ratio',
                 'joint_limit_fraction', 'joint_limit_max_error', 'torque_saturation',
                 'operating_speed_excess_squared')


def manufacturer_parameters():
    """Return independent arrays in our fixed 22-joint order (joint-output units)."""
    effort = np.array([6.]*2+[14.]*8+[68.,76.,38.3,112.,38.3,38.3]*2)
    velocity = np.array([7.85]*2+[33.51]*8+[14.66,12.57,17.59,12.57,17.59,17.59]*2)
    knee = np.minimum([10.47]*2+[5.24]*8+[1.88,2.62,7.85,2.09,7.85,7.85]*2, velocity)
    armature = np.array([.001]*10+[.0478125,.0339552,.0282528,.095625,.0565056,.0565056]*2)
    natural_frequency = np.array([10.]*10+[4.]*12)
    damping_ratio = np.array([2.]*10+[1.5,1.5,1.5,1.,1.5,1.5]*2)
    omega = 2*3.1415926535*natural_frequency
    return dict(effort=effort, velocity=velocity, knee=knee, armature=armature,
                manufacturer_kp=armature*omega**2, manufacturer_kd=2*damping_ratio*armature*omega)


def actuator_parameters(robot, settings=None):
    profile = (settings or {}).get('actuator_profile')
    if profile is None:
        return dict(effort=robot.effort.copy(), velocity=robot.velocity_limit.copy(),
                    knee=robot.velocity_limit.copy(), armature=robot.model.dof_armature[6:].copy())
    if profile != ACTUATOR_PROFILE:
        raise ValueError(f'Unknown actuator profile: {profile}')
    result = manufacturer_parameters()
    if not np.array_equal(robot.velocity_limit, result['velocity']):
        raise ValueError('Official actuator profile requires the pinned K1 joint-speed mapping')
    return result


def torque_envelope(velocity, effort, nominal_velocity, knee_velocity):
    """Official symmetric T-N limit; braking is also zero at/above nominal speed."""
    nominal_velocity = np.asarray(nominal_velocity, dtype=float)
    knee = np.minimum(np.maximum(knee_velocity, 0.), nominal_velocity)
    denominator = np.maximum(nominal_velocity-knee, 1e-6)
    limit = np.clip(np.asarray(effort)*(nominal_velocity-np.abs(velocity))/denominator, 0., effort)
    return np.where(nominal_velocity <= 0., 0., np.where(np.isfinite(nominal_velocity), limit, effort))


def speed_guard_torque(torque, velocity, operating_speed):
    """Taper only accelerating motor torque over the final 10% of operating speed.

    Call after the manufacturer torque envelope. Requested braking is unchanged;
    measured state and external forces are unchanged, so this is not a hard speed
    guarantee. Both limits and velocity are joint-output radians per second.
    """
    torque, velocity = np.asarray(torque), np.asarray(velocity)
    operating_speed = np.asarray(operating_speed)
    factor = np.clip((operating_speed-np.abs(velocity))/(.1*operating_speed), 0., 1.)
    return np.where(torque*velocity > 0., torque*factor, torque)


def effective_model(robot, settings=None):
    """Copy only for the opt-in dynamics override; never change the reference model."""
    parameters = actuator_parameters(robot, settings)
    if (settings or {}).get('actuator_profile') is None:
        return robot.model
    model = copy.copy(robot.model)
    model.actuator_forcerange[:, 0] = -parameters['effort']
    model.actuator_forcerange[:, 1] = parameters['effort']
    model.dof_armature[6:] = parameters['armature']
    # Armature also enters cached reference inertia and constraint weights.
    # Refresh these before either native stepping or converting the model to Warp.
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return model


def configure_robot_actuators(robot, settings=None):
    """One-time scalar-instance copy, preserving state and controller gains."""
    profile = (settings or {}).get('actuator_profile')
    if profile is None:
        return
    if getattr(robot, '_actuator_profile', None) == profile:
        return
    model = effective_model(robot, settings)
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, robot.data)
    robot.model, robot.data = model, data
    robot.effort = actuator_parameters(robot, settings)['effort']
    robot._actuator_profile = profile
    robot._actuator_settings = dict(settings)


def actuator_contract(robot, settings=None):
    from .actuation import action_settings, command_velocity_limits
    values = action_settings(robot, settings)
    parameters = actuator_parameters(robot, values)
    profile = values.get('actuator_profile')
    contract = dict(profile=profile or 'legacy-fixed-effort-v1',
                source_commit=SOURCE_COMMIT if profile else None,
                peak_effort_nm=parameters['effort'].tolist(),
                nominal_velocity_rad_s=parameters['velocity'].tolist(),
                knee_velocity_rad_s=parameters['knee'].tolist() if profile else None,
                armature_kg_m2=parameters['armature'].tolist(),
                operating_velocity_rad_s=command_velocity_limits(robot, values).tolist(),
                gains='existing_K1_PV_gains_with_explicit_action_scales',
                command_delay_physics_steps=0,
                manufacturer_random_delay_imported=False,
                reference_model_signature=robot.signature,
                safety_sampling='post-integration at every physics substep, including final control-interval state',
                safety_dt=float(robot.model.opt.timestep),
                fractions='mean over joints and physics substeps; max metrics cover the same post-step samples',
                hardware_verified=False)
    if values.get('operating_speed_guard', False):
        contract['operating_speed_guard'] = dict(version='accelerating-torque-taper-v1',
            onset_fraction=.9, zero_accelerating_torque_fraction=1.,
            operation='after manufacturer torque envelope at every physics substep',
            feedback='measured pre-integration joint velocity',
            braking='unchanged within manufacturer envelope', state_clipping=False,
            guarantee='motor torque restriction only; external dynamics may still exceed operating speed')
    return contract


def safety_sample(q, dq, requested_torque, torque_limit, joint_limits, operating_speed, nominal_speed):
    operating_ratio = np.abs(dq)/operating_speed
    nominal_ratio = np.abs(dq)/nominal_speed
    joint_error = np.maximum(np.maximum(joint_limits[:,0]-q, q-joint_limits[:,1]), 0.)
    return np.array([np.mean(operating_ratio > 1.), np.max(operating_ratio),
                     np.mean(nominal_ratio > 1.), np.max(nominal_ratio),
                     np.mean(joint_error > 0.), np.max(joint_error),
                     np.mean(np.abs(requested_torque) > torque_limit),
                     np.mean(np.maximum(operating_ratio-1., 0.)**2)])


def accumulate_safety(total, sample, substeps):
    for index in range(len(SAFETY_FIELDS)):
        if index in (1,3,5):
            total[index] = max(total[index], sample[index])
        else:
            total[index] += sample[index]/substeps
