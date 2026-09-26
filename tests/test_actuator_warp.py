"""Bounded real Warp/native qualification of the optional actuator contract."""
import numpy as np
import pytest
import torch
import warp as wp

from k1_motion.actuators import ACTUATOR_PROFILE, SAFETY_FIELDS, manufacturer_parameters, torque_envelope
from k1_motion.cpu_physics import CpuParallelPhysics
from k1_motion.robot import K1Model
from k1_motion.warp_physics import WarpPhysics, _pd
from test_actuator_contract import reset_reference, settings


def test_warp_pd_kernel_matches_official_envelope_at_both_signed_speed_boundaries():
    wp.init()
    device = 'cpu'
    params = manufacturer_parameters()
    speed = np.stack([np.zeros(22), params['knee'],
                      (params['knee']+params['velocity'])/2, params['velocity'],
                      params['velocity']*1.1, -params['velocity']*.9]).astype(np.float32)
    worlds = len(speed)
    qpos = np.zeros((worlds,29),np.float32)
    qvel = np.zeros((worlds,28),np.float32)
    qvel[:,6:] = speed
    def a(x):
        return wp.array(np.asarray(x,np.float32),dtype=float,device=device)
    ctrl,power,saturation,demand,available = [wp.zeros((worlds,22),dtype=float,device=device) for _ in range(5)]
    wp.launch(_pd,dim=(worlds,22),device=device,inputs=[
        a(qpos),a(qvel),a(np.ones((worlds,22))),a(np.zeros((worlds,22))),
        a(np.full(22,1000)),a(np.zeros(22)),a(params['effort']),a(np.tile([-2,2],(22,1))),1.,
        a(params['velocity']),a(params['knee']),a(.8*params['velocity']),1,
        ctrl,power,saturation,demand,available])
    # Compare the same represented limits: the head's clamped knee creates a
    # 1e-6-wide edge, so a float32-vs-float64 nominal value changes that boundary.
    expected = torque_envelope(speed.astype(float), params['effort'].astype(np.float32),
                               params['velocity'].astype(np.float32),params['knee'].astype(np.float32))
    np.testing.assert_allclose(available.numpy(),expected,rtol=2e-6,atol=2e-5)
    np.testing.assert_allclose(ctrl.numpy(),expected,rtol=2e-6,atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='NVIDIA CUDA required for real Warp parity')
def test_real_warp_native_actuation_and_poststep_safety_parity():
    torch.set_num_threads(1)
    robot = K1Model()
    robot.substeps=1
    robot.control_dt=.002
    config=settings(robot,.001)
    native=CpuParallelPhysics(robot,2,'cpu',config,workers=2)
    warp=WarpPhysics(robot,2,'cuda:0',action_settings=config,epa_horizon=96)
    reference=reset_reference(robot)
    try:
        native.reset(torch.arange(2),reference)
        warp.reset(torch.arange(2,device='cuda:0'),{k:v.cuda() for k,v in reference.items()})
        for tick in range(5):
            target=reference['joint_position']+.1*torch.cos(torch.arange(44).reshape(2,22)+tick)
            native.step(target)
            warp.step(target.cuda())
            for key in ('q','dq','position','velocity'):
                torch.testing.assert_close(warp.state()[key].cpu(),native.state()[key],rtol=3e-4,atol=2e-5)
            for key in SAFETY_FIELDS:
                torch.testing.assert_close(warp.safety_per_env[key].cpu(),native.safety_per_env[key],rtol=5e-4,atol=5e-4)
            measured=warp.state()['dq'].abs()
            assert measured.max()>.001
            torch.testing.assert_close(warp.safety_per_env['operating_speed_max_ratio'],
                                       (measured/.001).max(-1).values,rtol=2e-6,atol=2e-6)
        assert warp.contract['actuator']['profile']==ACTUATOR_PROFILE
        ids=torch.tensor([1],device='cuda:0')
        warp.reset(ids,{k:v[1:].cuda() for k,v in reference.items()})
        for value in warp.safety_per_env.values():
            assert value[1]==0
        warp.validate()
    finally:
        native.close()
        warp.close()
