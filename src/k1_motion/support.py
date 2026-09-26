"""Read-only scalar oracle for the native foot-ground support measurement."""
import mujoco
import numpy as np

from .spatial_rewards import SUPPORT_CONTRACT


def foot_support(model, data):
    bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side+'_ankle_roll_link')
              for side in ('left', 'right')]
    if min(bodies) < 1:
        raise ValueError('Missing foot bodies for contact measurement')
    force_sum, weighted_speed = np.zeros(2), np.zeros(2)
    force, jacobian = np.zeros(6), np.zeros((3, model.nv))
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.efc_address < 0:
            continue
        first, second = model.geom_bodyid[[contact.geom1, contact.geom2]]
        body = second if first == 0 else first if second == 0 else -1
        if body not in bodies:
            continue
        foot = bodies.index(body)
        mujoco.mj_contactForce(model, data, index, force)
        normal_force = max(float(force[0]), 0.)
        if normal_force == 0:
            continue
        mujoco.mj_jac(model, data, jacobian, None, contact.pos, body)
        velocity = jacobian @ data.qvel
        normal = contact.frame[:3]
        tangent = velocity - normal * np.dot(normal, velocity)
        force_sum[foot] += normal_force
        weighted_speed[foot] += normal_force * np.dot(tangent, tangent)
    return dict(contact=force_sum > SUPPORT_CONTRACT['normal_force_threshold_n'],
                slip_speed=np.sqrt(weighted_speed/np.maximum(force_sum, 1e-12)), normal_force=force_sum)
