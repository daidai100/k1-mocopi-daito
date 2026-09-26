"""GPU MuJoCo with the same K1 MJCF, PD loop, limits, and task tensors.

Zero-copy buffers synchronize a dedicated graph stream with Torch's current
stream. The reset graph uses a mutable world mask; episode resets never replace
device arrays. Buffer overflows are latched and checked before learner updates.
"""

from importlib.metadata import version

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from mujoco_warp._src.support import contact_force_fn, jac_dof
from mujoco_warp._src.types import vec5

from .observations import quat_apply, quat_inv
from .warp_compat import epa_horizon_capacity
from .servo import gains
from .actuators import SAFETY_FIELDS, actuator_parameters, actuator_contract, effective_model
from .actuation import action_settings as resolve_settings, command_velocity_limits
from .spatial_rewards import SUPPORT_CONTRACT


@wp.kernel(enable_backward=False)
def _pd(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    target: wp.array2d[float],
    target_velocity: wp.array2d[float],
    kp: wp.array[float],
    kd: wp.array[float],
    effort: wp.array[float],
    limits: wp.array2d[float],
    fraction: float,
    nominal_velocity: wp.array[float],
    knee_velocity: wp.array[float],
    command_velocity: wp.array[float],
    dynamic_actuator: int,
    ctrl: wp.array2d[float],
    power: wp.array2d[float],
    saturation: wp.array2d[float],
    demand: wp.array2d[float],
    available: wp.array2d[float],
):
    world, joint = wp.tid()
    desired = wp.clamp(target[world, joint], limits[joint, 0], limits[joint, 1])
    torque = kp[joint] * (desired - qpos[world, joint + 7])
    desired_velocity = target_velocity[world, joint]
    limit = effort[joint]
    if dynamic_actuator != 0:
        desired_velocity = wp.clamp(desired_velocity, -command_velocity[joint], command_velocity[joint])
        denominator = wp.max(nominal_velocity[joint]-knee_velocity[joint], 1.0e-6)
        limit = wp.clamp(effort[joint]*(nominal_velocity[joint]-wp.abs(qvel[world,joint+6]))/denominator,
                         0.0, effort[joint])
    torque += kd[joint] * (desired_velocity - qvel[world, joint + 6])
    bounded = wp.clamp(torque, -limit, limit)
    ctrl[world, joint] = bounded
    demand[world,joint] = torque
    available[world,joint] = limit
    normalized = bounded / effort[joint]
    power[world, joint] += fraction * normalized * normalized
    if wp.abs(torque) >= limit:
        saturation[world, joint] += fraction


@wp.kernel(enable_backward=False)
def _speed_guard(
    qvel: wp.array2d[float], operating: wp.array[float], effort: wp.array[float],
    fraction: float, ctrl: wp.array2d[float], power: wp.array2d[float],
):
    world, joint = wp.tid()
    before = ctrl[world, joint]
    velocity = qvel[world, joint+6]
    if before*velocity > 0.0:
        factor = wp.clamp((operating[joint]-wp.abs(velocity))/(0.1*operating[joint]), 0.0, 1.0)
        after = before*factor
        ctrl[world, joint] = after
        old_normalized, new_normalized = before/effort[joint], after/effort[joint]
        # _pd already accumulated unguarded effort. Correct only this substep;
        # demand/envelope saturation retain their original physical meaning.
        power[world, joint] -= fraction*old_normalized*old_normalized
        power[world, joint] += fraction*new_normalized*new_normalized


@wp.kernel(enable_backward=False)
def _safety(
    qpos: wp.array2d[float], qvel: wp.array2d[float],
    demand: wp.array2d[float], available: wp.array2d[float],
    operating: wp.array[float], nominal: wp.array[float], limits: wp.array2d[float],
    fraction: float, result: wp.array2d[float],
):
    world = wp.tid()
    operating_count = float(0.0)
    nominal_count = float(0.0)
    joint_count = float(0.0)
    saturation_count = float(0.0)
    operating_max = float(0.0)
    nominal_max = float(0.0)
    joint_max = float(0.0)
    excess_squared = float(0.0)
    for joint in range(22):
        operating_ratio = wp.abs(qvel[world,joint+6])/operating[joint]
        nominal_ratio = wp.abs(qvel[world,joint+6])/nominal[joint]
        joint_error = wp.max(wp.max(limits[joint,0]-qpos[world,joint+7],
                                   qpos[world,joint+7]-limits[joint,1]), 0.0)
        if operating_ratio > 1.0:
            operating_count += 1.0
        if nominal_ratio > 1.0:
            nominal_count += 1.0
        if joint_error > 0.0:
            joint_count += 1.0
        if wp.abs(demand[world,joint]) > available[world,joint]:
            saturation_count += 1.0
        operating_max = wp.max(operating_max, operating_ratio)
        nominal_max = wp.max(nominal_max, nominal_ratio)
        joint_max = wp.max(joint_max, joint_error)
        excess = wp.max(operating_ratio-1.0, 0.0)
        excess_squared += excess*excess
    result[world,0] += fraction*operating_count/22.0
    result[world,1] = wp.max(result[world,1],operating_max)
    result[world,2] += fraction*nominal_count/22.0
    result[world,3] = wp.max(result[world,3],nominal_max)
    result[world,4] += fraction*joint_count/22.0
    result[world,5] = wp.max(result[world,5],joint_max)
    result[world,6] += fraction*saturation_count/22.0
    result[world,7] += fraction*excess_squared/22.0


@wp.kernel(enable_backward=False)
def _self_contact(
    count: wp.array[int],
    distance: wp.array[float],
    geoms: wp.array[wp.vec2i],
    worlds: wp.array[int],
    geom_body: wp.array[int],
    collision: wp.array[float],
):
    index = wp.tid()
    if index < count[0] and distance[index] <= 0.0:
        pair = geoms[index]
        if pair[0] >= 0 and pair[1] >= 0:
            if geom_body[pair[0]] != 0 and geom_body[pair[1]] != 0:
                wp.atomic_max(collision, worlds[index], 1.0)


@wp.kernel(enable_backward=False)
def _foot_support(
    count: wp.array[int], geoms: wp.array[wp.vec2i], worlds: wp.array[int],
    points: wp.array[wp.vec3], frames: wp.array[wp.mat33], friction: wp.array[vec5],
    dimensions: wp.array[int], addresses: wp.array2d[int], forces: wp.array2d[float],
    cone: int, njmax: int, geom_body: wp.array[int], left: int, right: int,
    body_parent: wp.array[int], body_root: wp.array[int], dof_body: wp.array[int],
    ancestor: wp.array2d[int], com: wp.array2d[wp.vec3], cdof: wp.array2d[wp.spatial_vector],
    qvel: wp.array2d[float], sums: wp.array2d[float],
):
    index = wp.tid()
    if index >= count[0] or addresses[index, 0] < 0:
        return
    pair = geoms[index]
    if pair[0] < 0 or pair[1] < 0:
        return
    first, second = geom_body[pair[0]], geom_body[pair[1]]
    body = int(-1)
    if first == 0:
        body = second
    elif second == 0:
        body = first
    foot = int(-1)
    if body == left:
        foot = 0
    elif body == right:
        foot = 1
    if foot < 0:
        return
    world = worlds[index]
    force = contact_force_fn(cone, frames, friction, dimensions, addresses, forces,
                             njmax, count, world, index, False)
    normal_force = wp.max(force[0], 0.0)
    if normal_force <= 0.0:
        return
    velocity = wp.vec3(0.0)
    for dof in range(qvel.shape[1]):
        jacp, jacr = jac_dof(body_parent, body_root, dof_body, ancestor, com, cdof,
                             points[index], body, dof, world)
        velocity += jacp*qvel[world, dof]
    normal = frames[index][0]
    tangent = velocity-normal*wp.dot(normal, velocity)
    wp.atomic_add(sums, world, foot, normal_force)
    wp.atomic_add(sums, world, foot+2, normal_force*wp.dot(tangent, tangent))


class WarpPhysics:
    def __init__(self, spec, num_envs, device, nconmax=64, njmax=256, conditional_graphs=True,
                 epa_horizon=None, action_settings=None):
        self.spec, self.num_envs, self.device = spec, num_envs, torch.device(device)
        settings = resolve_settings(spec, action_settings)
        parameters = actuator_parameters(spec, settings)
        self.dynamic_actuator = int('actuator_profile' in settings)
        self.operating_speed_guard = settings.get('operating_speed_guard', False)
        model = effective_model(spec, settings)
        initial_data = mujoco.MjData(model)
        mujoco.mj_copyData(initial_data, model, spec.data)
        if self.device.type != "cuda" or min(num_envs, nconmax, njmax) < 1:
            raise ValueError("MuJoCo Warp requires CUDA and positive world/buffer sizes")
        if epa_horizon is not None and spec.model.nflex:
            raise ValueError("EPA horizon override is qualified only for rigid K1 collision geometry")
        wp.init()
        # CUDA graph capture cannot use Torch's legacy default (null) stream.
        self.torch_stream = torch.cuda.Stream(device=self.device)
        self.torch_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.stream = wp.stream_from_torch(self.torch_stream)
        with (
            epa_horizon_capacity(epa_horizon) as horizon_capacity,
            torch.cuda.stream(self.torch_stream),
            wp.ScopedStream(self.stream),
        ):
            self.model = mjw.put_model(model)
            self.model.opt.graph_conditional = conditional_graphs
            self.data = mjw.put_data(model, initial_data, nworld=num_envs, nconmax=nconmax, njmax=njmax)
            self.model.opt.warn_overflow = False  # The latched error below is fatal, never silently ignored.
            self.target_wp = wp.zeros((num_envs, 22), dtype=float)
            self.target_velocity_wp = wp.zeros((num_envs, 22), dtype=float)
            self.power_wp = wp.zeros((num_envs, 22), dtype=float)
            self.saturation_wp = wp.zeros((num_envs, 22), dtype=float)
            self.self_collision_wp = wp.zeros(num_envs, dtype=float)
            self.demand_wp = wp.zeros((num_envs,22), dtype=float)
            self.available_wp = wp.zeros((num_envs,22), dtype=float)
            self.safety_wp = wp.zeros((num_envs,len(SAFETY_FIELDS)), dtype=float)
            self.support_wp = wp.zeros((num_envs, 4), dtype=float)
            self.support_sums = wp.to_torch(self.support_wp)
            self.foot_bodies = [model.body(side+'_ankle_roll_link').id for side in ('left', 'right')]
            self.reset_wp = wp.zeros(num_envs, dtype=bool)
            kp, kd = gains(spec, action_settings)
            self.kp = wp.array(kp.astype(np.float32), dtype=float)
            self.kd = wp.array(kd.astype(np.float32), dtype=float)
            self.effort_limit = wp.array(parameters['effort'].astype(np.float32), dtype=float)
            self.nominal_velocity = wp.array(parameters['velocity'].astype(np.float32), dtype=float)
            self.knee_velocity = wp.array(parameters['knee'].astype(np.float32), dtype=float)
            self.operating_velocity = wp.array(command_velocity_limits(spec, settings).astype(np.float32), dtype=float)
            self.limits = wp.array(spec.limits.astype(np.float32), dtype=float)
            self.target = wp.to_torch(self.target_wp)
            self.target_velocity = wp.to_torch(self.target_velocity_wp)
            self.power = wp.to_torch(self.power_wp)
            self.saturated = wp.to_torch(self.saturation_wp)
            self.self_collision_per_env = wp.to_torch(self.self_collision_wp)
            self.safety = wp.to_torch(self.safety_wp)
            self.safety_per_env = {key:self.safety[:,index] for index,key in enumerate(SAFETY_FIELDS)}
            self.reset_mask = wp.to_torch(self.reset_wp)
            self.qpos = wp.to_torch(self.data.qpos)
            self.qvel = wp.to_torch(self.data.qvel)
            self.overflow = wp.to_torch(self.data.overflow)
            self.overflow_latch = torch.zeros_like(self.overflow)
            self.target[:] = torch.as_tensor(spec.neutral, dtype=torch.float32, device=self.device)
            # Load every kernel before capture; graph launches then have no JIT work.
            mjw.reset_data(self.model, self.data, self.reset_wp)
            self._control_step()
            mjw.kinematics(self.model, self.data)
            self._measure_support()
            with wp.ScopedCapture(stream=self.stream) as capture:
                mjw.reset_data(self.model, self.data, self.reset_wp)
            self.reset_graph = capture.graph
            with wp.ScopedCapture(stream=self.stream) as capture:
                self._physics_step()
            self.step_graph = capture.graph
            with wp.ScopedCapture(stream=self.stream) as capture:
                mjw.kinematics(self.model, self.data)
            self.kinematics_graph = capture.graph
            with wp.ScopedCapture(stream=self.stream) as capture:
                self._measure_support()
            self.support_graph = capture.graph
        torch.cuda.current_stream(self.device).wait_stream(self.torch_stream)
        self.site_ids = torch.tensor(spec.site_ids, device=self.device)
        self.body_ids = torch.tensor(spec.model.site_bodyid[spec.site_ids], device=self.device)
        self.effort = self.saturation = torch.zeros((), device=self.device)
        self.effort_per_env = torch.zeros(num_envs, device=self.device)
        self.contract = {
            "implementation": "mujoco-warp-k1-pv-v2",
            "action_settings": action_settings,
            "model_signature": spec.signature,
            "mujoco": mujoco.__version__,
            "mujoco_warp": version("mujoco-warp"),
            "warp": wp.__version__,
            "physics_dt": spec.model.opt.timestep,
            "control_dt": spec.control_dt,
            "nconmax": nconmax,
            "njmax": njmax,
            "epa_horizon": horizon_capacity,
            "integrator": int(spec.model.opt.integrator),
            "solver": int(spec.model.opt.solver),
            "solver_iterations": spec.model.opt.iterations,
            "solver_tolerance": spec.model.opt.tolerance,
            "overflow_check": "latched across episode resets; fatal before PPO update or validation report",
            "derived_state": "kinematics refreshed after integration",
            "physics_steps_per_graph": 1,
            "conditional_graphs": conditional_graphs,
            "self_contact_measurement": "any non-ground penetration during the control interval",
            "support": {**SUPPORT_CONTRACT, "device": "cuda"},
            "actuator": actuator_contract(spec, settings),
        }
        self.validate()

    def _control_step(self):
        self.power_wp.zero_()
        self.saturation_wp.zero_()
        self.self_collision_wp.zero_()
        self.safety_wp.zero_()
        for _ in range(self.spec.substeps):
            self._physics_step()
        mjw.kinematics(self.model, self.data)

    def _physics_step(self):
        wp.launch(
            _pd,
            dim=(self.num_envs, 22),
            inputs=[
                self.data.qpos,
                self.data.qvel,
                self.target_wp,
                self.target_velocity_wp,
                self.kp,
                self.kd,
                self.effort_limit,
                self.limits,
                1.0 / self.spec.substeps,
                self.nominal_velocity,
                self.knee_velocity,
                self.operating_velocity,
                self.dynamic_actuator,
                self.data.ctrl,
                self.power_wp,
                self.saturation_wp,
                self.demand_wp,
                self.available_wp,
            ],
        )
        if self.operating_speed_guard:
            wp.launch(_speed_guard, dim=(self.num_envs, 22),
                      inputs=[self.data.qvel, self.operating_velocity, self.effort_limit,
                              1.0/self.spec.substeps, self.data.ctrl, self.power_wp])
        mjw.step(self.model, self.data)
        wp.launch(_safety, dim=self.num_envs,
                  inputs=[self.data.qpos, self.data.qvel, self.demand_wp, self.available_wp,
                          self.operating_velocity, self.nominal_velocity, self.limits,
                          1.0/self.spec.substeps, self.safety_wp])
        wp.launch(
            _self_contact,
            dim=self.data.contact.dist.shape[0],
            inputs=[
                self.data.nacon,
                self.data.contact.dist,
                self.data.contact.geom,
                self.data.contact.worldid,
                self.model.geom_bodyid,
                self.self_collision_wp,
            ],
        )

    def reset(self, ids, ref):
        self.safety[ids] = 0
        self.support_sums[ids] = 0
        self.overflow_latch.bitwise_or_(self.overflow)
        self.reset_mask.zero_()
        self.reset_mask[ids] = True
        self._launch(self.reset_graph)
        self.qpos[ids] = torch.cat(
            [ref[k] for k in ("root_position", "root_orientation", "joint_position")], -1
        )
        velocity = ref["root_velocity"]
        self.qvel[ids, :3] = velocity[:, :3]
        self.qvel[ids, 3:6] = quat_apply(quat_inv(ref["root_orientation"]), velocity[:, 3:])
        self.qvel[ids, 6:] = ref["joint_velocity"]
        self._launch(self.kinematics_graph)

    def _launch(self, graph, repetitions=1, kinematics=False):
        current = torch.cuda.current_stream(self.device)
        self.torch_stream.wait_stream(current)
        with wp.ScopedStream(self.stream):
            for _ in range(repetitions):
                wp.capture_launch(graph)
            if kinematics:
                wp.capture_launch(self.kinematics_graph)
        current.wait_stream(self.torch_stream)

    def step(self, target, velocity=None):
        self.target.copy_(target.detach())
        if velocity is None:
            self.target_velocity.zero_()
        else:
            self.target_velocity.copy_(velocity.detach())
        self.power.zero_()
        self.saturated.zero_()
        self.self_collision_per_env.zero_()
        self.safety.zero_()
        self._launch(self.step_graph, self.spec.substeps, kinematics=True)
        self._launch(self.support_graph)
        self.overflow_latch.bitwise_or_(self.overflow)
        self.effort_per_env = self.power.mean(-1)
        self.effort = self.effort_per_env.mean()
        self.saturation = self.saturated.mean()

    def _measure_support(self):
        self.support_wp.zero_()
        model, data, contact = self.model, self.data, self.data.contact
        wp.launch(_foot_support, dim=contact.dist.shape[0], inputs=[
            data.nacon, contact.geom, contact.worldid, contact.pos, contact.frame, contact.friction,
            contact.dim, contact.efc_address, data.efc.force, model.opt.cone, data.njmax,
            model.geom_bodyid, *self.foot_bodies, model.body_parentid, model.body_rootid,
            model.dof_bodyid, model.body_isdofancestor, data.subtree_com, data.cdof,
            data.qvel, self.support_wp])

    def foot_support(self):
        force = self.support_sums[:, :2]
        return dict(normal_force=force.clone(), contact=force > SUPPORT_CONTRACT['normal_force_threshold_n'],
                    slip_speed=(self.support_sums[:, 2:]/force.clamp_min(1e-12)).clamp_min(0).sqrt())

    def state(self):
        return {
            "q": self.qpos[:, 7:].clone(),
            "dq": self.qvel[:, 6:].clone(),
            "orientation": self.qpos[:, 3:7].clone(),
            "omega": self.qvel[:, 3:6].clone(),
            "position": self.qpos[:, :3].clone(),
            "velocity": self.qvel[:, :3].clone(),
            "landmarks": wp.to_torch(self.data.site_xpos)[:, self.site_ids].clone(),
            "body_orientation": wp.to_torch(self.data.xquat)[:, self.body_ids].clone(),
        }

    def validate(self):
        errors = self.overflow_latch | self.overflow
        if torch.any(errors):
            raise RuntimeError(
                f"MuJoCo Warp buffer overflow; affected worlds: {torch.count_nonzero(errors).item()}; "
                f"overflow bitmasks: {torch.unique(errors[errors != 0]).cpu().tolist()}"
            )
        if not torch.isfinite(self.qpos).all() or not torch.isfinite(self.qvel).all():
            raise FloatingPointError("Nonfinite MuJoCo Warp physics state")

    def close(self):
        self.validate()
        torch.cuda.synchronize(self.device)
