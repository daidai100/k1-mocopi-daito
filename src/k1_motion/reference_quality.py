"""Stable reference-contact diagnostics, separate from dynamic qualification."""
from itertools import product

import mujoco
import numpy as np

from .math3d import rotation
from .reference_velocity_clock import velocity_errors


def audit_reference_consistency(clip, robot):
    model=robot.model
    data=mujoco.MjData(model)
    feet=[model.body(side+'_ankle_roll_link').id for side in ('left','right')]
    geoms=[(foot,g) for foot in feet for g in range(model.ngeom)
        if model.geom_bodyid[g]==foot and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not geoms or any(model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_BOX for _,g in geoms):
        raise ValueError('Stable contact audit requires the declared K1 foot boxes')
    signs=np.asarray(list(product((-1.,1.),repeat=3)))
    velocities=[]
    fk_error=0.
    v=clip.values
    for i in range(len(clip.times)):
        data.qpos[:]=np.r_[v['root_position'][i],v['root_orientation'][i],v['joint_position'][i]]
        data.qvel[:3]=v['root_velocity'][i,:3]
        data.qvel[3:6]=rotation(v['root_orientation'][i]).inv().apply(v['root_velocity'][i,3:])
        data.qvel[6:]=v['joint_velocity'][i]
        mujoco.mj_forward(model,data)
        fk_error=max(fk_error,float(np.linalg.norm(data.site_xpos[robot.site_ids]-v['landmarks'][i],axis=-1).max()))
        for foot,g in geoms:
            corners=data.geom_xpos[g]+(signs*model.geom_size[g])@data.geom_xmat[g].reshape(3,3).T
            for point in corners[corners[:,2]<=.001]:
                jac=np.zeros((3,model.nv))
                mujoco.mj_jac(model,data,jac,None,point,foot)
                velocities.append(float(np.linalg.norm((jac@data.qvel)[:2])))
    mean=float(np.mean(velocities)) if velocities else None
    derivative_error=max(velocity_errors(clip).values())
    return dict(version='stable-near-floor-reference-consistency-v1',frames=len(clip.times),
        audit_hz=50,near_floor_samples=len(velocities),near_floor_tangent_mean_m_s=mean,
        near_floor_tangent_p95_m_s=float(np.percentile(velocities,95)) if velocities else None,
        playback_velocity_max_error=derivative_error,landmark_fk_max_error_m=fk_error,
        contact_consistent=mean is not None and mean<=.2 and derivative_error<=1e-8 and fk_error<=1e-8,
        gate=dict(mean_near_floor_tangent_m_s=.2,playback_velocity_max_error=1e-8,landmark_fk_max_error_m=1e-8),
        contact_definition='Every foot-box corner at or below floor + 1 mm; includes exact zero contact',
        scope='Kinematic consistency only; retain the separate 500 Hz admission audit; not dynamic solvability',
        physics_qualified=False)
