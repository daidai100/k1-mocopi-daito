"""Compiled, shared-model CPU MuJoCo with batched policy-device transfers.

The C++ hot loop retains the exact K1 PD feedback/contact checks at every physics
substep. Independent mjData instances are distributed over an OpenMP thread team.
The native library is built once in a locked, content-addressed local cache.
"""
import ctypes as ct
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import mujoco
import numpy as np
import torch

from .math3d import rotation
from .robot import ROOT
from .servo import gains
from .actuators import SAFETY_FIELDS, actuator_parameters, actuator_contract, effective_model

STATE_WIDTH = 179 + len(SAFETY_FIELDS)


def native_library():
    source = Path(__file__).parent / "native/cpu_physics.cpp"
    installed = Path(mujoco.__file__).parent
    library = installed / f"libmujoco.so.{mujoco.__version__}"
    compiler = os.environ.get("CXX", "c++")
    compiler_version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    flags = ["-O3", "-std=c++17", "-shared", "-fPIC", "-fopenmp", "-ffp-contract=off"]
    identity = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "mujoco": mujoco.__version__, "library": str(library), "compiler": compiler_version,
                "flags": flags, "machine": platform.machine()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = ROOT / "artifacts/native-build" / digest
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "cpu_physics.so"
    with (directory / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            temporary = directory / "cpu_physics.partial.so"
            command = [compiler, *flags, f"-I{installed / 'include'}", str(source), str(library),
                       f"-Wl,-rpath,{installed}", "-o", str(temporary)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            temporary.replace(destination)
            (directory / "build.json").write_text(json.dumps({**identity, "command": command}, indent=2)+"\n")
    lib = ct.CDLL(str(destination))  # CDLL releases the GIL during each batch operation.
    ptr = ct.c_void_p
    lib.k1_error.restype = ct.c_char_p
    lib.k1_mujoco_version.restype = ct.c_int
    if lib.k1_mujoco_version() != mujoco.mj_version():
        raise RuntimeError("Native MuJoCo ABI mismatch")
    lib.k1_create.argtypes = [ptr, ct.c_int, ct.c_int, ct.c_int, ct.c_int, *([ptr]*9), ct.c_int, ct.c_int]
    lib.k1_create.restype = ptr
    lib.k1_reset.argtypes = [ptr, ptr, ct.c_int, ptr, ptr]
    lib.k1_reset.restype = ct.c_int
    lib.k1_step.argtypes = [ptr, ptr, ptr, ptr]
    lib.k1_step.restype = ct.c_int
    lib.k1_step_projected.argtypes = [ptr]*8
    lib.k1_step_projected.restype = ct.c_int
    lib.k1_destroy.argtypes = [ptr]
    lib.k1_destroy.restype = None
    lib.k1_workers.argtypes = [ptr]
    lib.k1_workers.restype = ct.c_int
    lib.k1_foot_support.argtypes = [ptr, ptr]
    lib.k1_foot_support.restype = ct.c_int
    return lib, {**identity, "build_key": digest}


def pointer(array):
    return array.ctypes.data_as(ct.c_void_p)


class CpuParallelPhysics:
    def __init__(self, spec, num_envs, device, action_settings=None, workers=32, chunk_size=4,
                 fuse_arm_feedback=True):
        if min(num_envs, workers) < 1 or chunk_size < 0:
            raise ValueError("Positive world/thread counts and nonnegative chunk required")
        if workers > len(os.sched_getaffinity(0)):
            raise ValueError("Requested CPU workers exceed the process CPU affinity")
        # A Python MuJoCo callback is unsafe when invoked by native OpenMP threads.
        for name in ("control", "passive", "sensor", "contactfilter", "act_dyn", "act_gain", "act_bias", "time"):
            if getattr(mujoco, f"get_mjcb_{name}")() is not None:
                raise ValueError("CPU parallel backend requires no process-global MuJoCo callbacks")
        self.spec, self.num_envs, self.device = spec, num_envs, torch.device(device)
        self.fuse_arm_feedback = fuse_arm_feedback
        self.handle = None
        self.lib, build = native_library()
        self.host = np.zeros((num_envs, STATE_WIDTH), np.float32)
        self.host_projected = np.empty((num_envs, 22), np.float32)
        self.projection_limits = np.ascontiguousarray(spec.limits, dtype=np.float32)
        from .actuation import action_settings as resolve_settings, command_velocity_limits
        settings = resolve_settings(spec, action_settings)
        parameters = actuator_parameters(spec, settings)
        operating = command_velocity_limits(spec, settings)
        dynamic = 'actuator_profile' in settings
        model = effective_model(spec, settings)
        actuator = np.ascontiguousarray([parameters['velocity'], parameters['knee'], operating], dtype=np.float64)
        self.projection_cap = np.ascontiguousarray(
            command_velocity_limits(spec, resolve_settings(spec, action_settings)) * spec.control_dt,
            dtype=np.float32)
        self.projection_timing = np.zeros(2, np.float64)
        kp, kd = gains(spec, action_settings)
        inputs = [np.ascontiguousarray(a, dtype=np.float64) for a in
                  (kp, kd, parameters['effort'], operating if dynamic else spec.velocity_limit, spec.limits)]
        sites = np.ascontiguousarray(spec.site_ids, dtype=np.int32)
        neutral = np.ascontiguousarray(spec.neutral_qpos, dtype=np.float64)
        self.handle = self.lib.k1_create(model._address, num_envs, workers, spec.substeps,
                                         chunk_size, *map(pointer, inputs), pointer(sites),
                                         pointer(neutral), pointer(self.host), pointer(actuator), int(dynamic),
                                         int(settings.get('operating_speed_guard', False)))
        if not self.handle:
            raise RuntimeError(self.lib.k1_error().decode())
        actual_workers = self.lib.k1_workers(self.handle)
        if actual_workers != workers:
            self.close()
            raise RuntimeError(f"OpenMP created {actual_workers} workers, requested {workers}")
        self.contract = {"backend": "native-mujoco-cpp-openmp", "workers": workers,
                         "actual_workers": actual_workers,
                         "chunk_size": chunk_size, "substeps": spec.substeps,
                         "control_dt": spec.control_dt, "physics_dt": float(spec.model.opt.timestep),
                         "precision": "native_mujoco_float64", "build": build,
                         "servo": "k1_motion.servo.step_pd-equivalent", "batched_transfers": True,
                         "fuse_arm_feedback": fuse_arm_feedback,
                         "arm_geometry": "kinematics-comPos-collision",
                         "actuator": actuator_contract(spec, settings)}
        self.timing = {"steps": 0, "action_transfer_seconds": 0., "native_step_seconds": 0.,
                       "state_transfer_seconds": 0., "reset_seconds": 0., "arm_projection_seconds": 0.}
        self._publish()

    def _check(self, result):
        if result:
            raise RuntimeError(self.lib.k1_error().decode())

    def _open(self):
        if not self.handle:
            raise RuntimeError("CPU parallel physics is closed")

    def _publish(self, projected=None):
        # One transfer instead of one transfer per field. Own the tensor memory;
        # state() values held by the caller must survive later steps/resets.
        host = self.host if projected is None else np.concatenate((self.host, projected), axis=1)
        values = torch.tensor(host, device=self.device)
        self.cached = {"q": values[:, :22], "dq": values[:, 22:44],
                       "orientation": values[:, 44:48], "omega": values[:, 48:51],
                       "position": values[:, 51:54], "velocity": values[:, 54:57],
                       "landmarks": values[:, 57:108].reshape(self.num_envs, 17, 3),
                       "body_orientation": values[:, 108:176].reshape(self.num_envs, 17, 4)}
        self.effort_per_env = values[:, 176]
        self.self_collision_per_env = values[:, 178]
        self.effort = float(self.host[:, 176].mean())
        self.saturation = float(self.host[:, 177].mean())
        self.safety_per_env = {key: values[:,179+index] for index,key in enumerate(SAFETY_FIELDS)}
        return values[:, STATE_WIDTH:] if projected is not None else None

    def reset(self, ids, ref):
        self._open()
        start = time.perf_counter()
        indices = ids.detach().cpu().numpy()
        if len(indices) == 0:
            return
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or len(np.unique(indices)) != len(indices) or min(indices) < 0 or max(indices) >= self.num_envs):
            raise ValueError("Reset IDs must be unique valid integers")
        indices = np.ascontiguousarray(indices, dtype=np.int32)
        keys = ("root_position", "root_orientation", "joint_position", "root_velocity", "joint_velocity")
        host = {k: ref[k].detach().cpu().numpy().astype(np.float64) for k in keys}
        for key, width in zip(keys, (3, 4, 22, 6, 22)):
            if host[key].shape != (len(indices), width) or not np.isfinite(host[key]).all():
                raise ValueError("Invalid reset reference")
        root = host["root_velocity"].copy()
        root[:, 3:] = rotation(host["root_orientation"]).inv().apply(root[:, 3:])
        states = np.ascontiguousarray(np.concatenate(
            [host["root_position"], host["root_orientation"], host["joint_position"], root,
             host["joint_velocity"]], axis=1))
        self._check(self.lib.k1_reset(self.handle, pointer(indices), len(indices), pointer(states),
                                      pointer(self.host)))
        self._publish()
        self.timing["reset_seconds"] += time.perf_counter()-start

    def step(self, target, velocity=None):
        self._open()
        start = time.perf_counter()
        velocity = torch.zeros_like(target) if velocity is None else velocity
        if target.shape != (self.num_envs, 22) or velocity.shape != target.shape:
            raise ValueError("Invalid batched K1 command shape")
        commands = torch.stack((target.detach(), velocity.detach())).cpu().numpy().astype(np.float64)
        if not np.isfinite(commands).all():
            raise ValueError("Nonfinite K1 command")
        transferred = time.perf_counter()
        self._check(self.lib.k1_step(self.handle, pointer(commands[0]), pointer(commands[1]),
                                     pointer(self.host)))
        stepped = time.perf_counter()
        self._publish()
        self.timing["steps"] += 1
        self.timing["action_transfer_seconds"] += transferred-start
        self.timing["native_step_seconds"] += stepped-transferred
        self.timing["state_transfer_seconds"] += time.perf_counter()-stepped

    def step_projected(self, target, velocity, previous_target, feedback):
        """Project and step on host, returning state and applied target together."""
        self._open()
        if not feedback.handle:
            raise RuntimeError("Arm feedback is closed")
        if any(x.shape != (self.num_envs, 22) for x in (target, velocity, previous_target)):
            raise ValueError("Invalid batched projected command shape")
        start = time.perf_counter()
        commands = torch.stack((target.detach(), velocity.detach(), previous_target.detach())).cpu().numpy()
        commands = np.ascontiguousarray(commands, dtype=np.float32)
        if not np.isfinite(commands).all():
            raise ValueError("Nonfinite K1 projected command")
        transferred = time.perf_counter()
        self._check(self.lib.k1_step_projected(
            self.handle, feedback.handle, pointer(commands), pointer(self.projection_limits),
            pointer(self.projection_cap), pointer(self.host_projected), pointer(self.host),
            pointer(self.projection_timing)))
        stepped = time.perf_counter()
        applied = self._publish(self.host_projected)
        self.timing["steps"] += 1
        self.timing["action_transfer_seconds"] += transferred-start
        self.timing["arm_projection_seconds"] += self.projection_timing[0]
        self.timing["native_step_seconds"] += self.projection_timing[1]
        self.timing["state_transfer_seconds"] += time.perf_counter()-stepped
        return applied

    def state(self):
        self._open()
        return self.cached

    def foot_support(self):
        self._open()
        output = np.empty((self.num_envs, 6), dtype=np.float32)
        self._check(self.lib.k1_foot_support(self.handle, pointer(output)))
        values = torch.tensor(output, device=self.device)
        from .spatial_rewards import SUPPORT_CONTRACT
        force = values[:, :2]
        return dict(contact=force > SUPPORT_CONTRACT['normal_force_threshold_n'],
                    slip_speed=values[:, 2:4], normal_force=force)

    def validate(self):
        self._open()
        if not np.isfinite(self.host).all():
            raise FloatingPointError("Nonfinite native CPU state")

    def close(self):
        if self.handle:
            self.lib.k1_destroy(self.handle)
            self.handle = None

    def __del__(self):
        if getattr(self, "handle", None):
            self.close()
