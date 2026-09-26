"""Published reward kinematics must describe the integrated control-tick state.

Failures covered: old site positions alongside new qpos, reset-to-first-step
velocity bias, altered servo trajectory/contact/effort from refreshing FK, and
partial resets changing the surviving world's state.
"""
import mujoco
import numpy as np
import pytest
import torch

from k1_motion.cpu_physics import CpuParallelPhysics
from k1_motion.robot import K1Model
from k1_motion.servo import step_pd
from k1_motion.tracking_env import MujocoPhysics


@pytest.mark.parametrize('backend', ['scalar', 'cpp'])
def test_published_kinematics_match_qpos_without_changing_physics(backend):
    torch.set_num_threads(1)
    spec = K1Model()
    physics = (MujocoPhysics(spec, 2, 'cpu') if backend == 'scalar'
               else CpuParallelPhysics(spec, 2, 'cpu', workers=1))
    reference = spec.neutral_reference()
    fields = ('root_position', 'root_orientation', 'joint_position', 'root_velocity', 'joint_velocity')
    ref = {key: torch.tensor(np.tile(getattr(reference, key), (2, 1)), dtype=torch.float32)
           for key in fields}
    ref['root_position'][0, 2] = 1.2
    ref['root_velocity'][0, 0] = 1.
    ref['root_velocity'][0, 5] = .5
    ref['joint_velocity'][0] = torch.linspace(-.3, .3, 22)
    # Unrefreshed production servo is the oracle for unchanged physical state.
    oracle = MujocoPhysics(spec, 2, 'cpu')
    fresh = mujoco.MjData(spec.model)
    try:
        for item in (physics, oracle):
            item.reset(torch.arange(2), ref)
        for tick in range(12):
            target = ref['joint_position'].clone()
            target[:, 2] += .1*np.sin(tick*.3)
            velocity = torch.zeros_like(target)
            expected_metrics = [step_pd(robot, q.numpy(), v.numpy())
                                for robot, q, v in zip(oracle.robots, target, velocity)]
            physics.step(target, velocity)
            state = physics.state()
            for index, robot in enumerate(oracle.robots):
                # No fresh FK or forward dynamics is run on the oracle.
                actual_qpos = np.r_[state['position'][index].numpy(),
                                    state['orientation'][index].numpy(), state['q'][index].numpy()]
                actual_qvel = np.r_[state['velocity'][index].numpy(),
                                    state['omega'][index].numpy(), state['dq'][index].numpy()]
                np.testing.assert_array_equal(actual_qpos, robot.data.qpos.astype(np.float32))
                np.testing.assert_array_equal(actual_qvel, robot.data.qvel.astype(np.float32))
                fresh.qpos[:] = actual_qpos
                mujoco.mj_kinematics(spec.model, fresh)
                np.testing.assert_allclose(state['landmarks'][index].numpy(),
                                           fresh.site_xpos[spec.site_ids], rtol=0, atol=2e-7)
                np.testing.assert_allclose(state['body_orientation'][index].numpy(),
                                           fresh.xquat[spec.model.site_bodyid[spec.site_ids]],
                                           rtol=0, atol=2e-7)
                assert float(physics.self_collision_per_env[index]) == float(expected_metrics[index]['self_collision'])
                assert float(physics.effort_per_env[index]) == pytest.approx(expected_metrics[index]['effort'], abs=1e-7)
            if tick == 5:
                surviving = {key: value[0].clone() for key, value in state.items()}
                ids = torch.tensor([1])
                for item in (physics, oracle):
                    item.reset(ids, {key: value[ids] for key, value in ref.items()})
                for key, value in surviving.items():
                    torch.testing.assert_close(physics.state()[key][0], value, rtol=0, atol=0)
    finally:
        physics.close()
        oracle.close()
