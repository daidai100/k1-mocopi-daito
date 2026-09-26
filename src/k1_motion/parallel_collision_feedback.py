"""Batched equivalent of ArmCollisionFeedback; private geometry per CPU worker."""
import ctypes as ct

import numpy as np
import torch

from .cpu_physics import native_library, pointer


class ParallelArmCollisionFeedback:
    def __init__(self, robot, clearance, workers):
        self.lib, self.build = native_library()
        self.lib.k1_arm_create.argtypes = [ct.c_void_p, ct.c_int, ct.c_double, ct.c_void_p]
        self.lib.k1_arm_create.restype = ct.c_void_p
        self.lib.k1_arm_project.argtypes = [ct.c_void_p, ct.c_int, *([ct.c_void_p]*4)]
        self.lib.k1_arm_project.restype = ct.c_int
        self.lib.k1_arm_destroy.argtypes = [ct.c_void_p]
        self.lib.k1_arm_destroy.restype = None
        neutral = np.ascontiguousarray(robot.neutral_qpos, dtype=np.float64)
        self.handle = self.lib.k1_arm_create(robot.model._address, workers, clearance, pointer(neutral))
        if not self.handle:
            raise RuntimeError(self.lib.k1_error().decode())

    def project_batch(self, joint_position, joint_velocity, target):
        if not self.handle:
            raise RuntimeError("Arm feedback is closed")
        arrays = [np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
                  for x in (joint_position, joint_velocity, target)]
        if any(a.shape != (len(target), 22) for a in arrays):
            raise ValueError("Arm feedback requires canonical batched joint vectors")
        output = np.empty_like(arrays[2])
        if self.lib.k1_arm_project(self.handle, len(target), *map(pointer, arrays), pointer(output)):
            raise RuntimeError(self.lib.k1_error().decode())
        return torch.tensor(output, dtype=target.dtype, device=target.device)

    def close(self):
        if self.handle:
            self.lib.k1_arm_destroy(self.handle)
            self.handle = None
