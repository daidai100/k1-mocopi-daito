"""Benchmark-only persistent CPU workers; native MuJoCo and the exact K1 servo.

The learner/actor and task tensors can remain on one GPU. Worker processes never
initialize a GPU or import Torch. This does not replace the production backend.
"""
import copy
import multiprocessing as mp
import traceback

import numpy as np


def _worker(connection, count, settings):
    import mujoco
    from k1_motion.math3d import rotation
    from k1_motion.robot import K1Model
    from k1_motion.servo import step_pd

    try:
        spec = K1Model()
        robots = []
        for _ in range(count):
            robot = copy.copy(spec)
            robot.data = mujoco.MjData(spec.model)
            robot.reset()
            robots.append(robot)
        effort, saturation, collision = np.zeros((3, count), np.float32)

        def respond():
            # Match the production state timestamp: mj_step's cached site/body
            # transforms precede its final qpos integration.
            for robot in robots:
                mujoco.mj_kinematics(robot.model, robot.data)
            fields = {
                "q": [r.data.qpos[7:] for r in robots],
                "dq": [r.data.qvel[6:] for r in robots],
                "orientation": [r.data.qpos[3:7] for r in robots],
                "omega": [r.data.qvel[3:6] for r in robots],
                "position": [r.data.qpos[:3] for r in robots],
                "velocity": [r.data.qvel[:3] for r in robots],
                "landmarks": [r.landmarks() for r in robots],
                "body_orientation": [r.data.xquat[r.model.site_bodyid[r.site_ids]] for r in robots],
            }
            connection.send({"state": {k: np.asarray(v, np.float32) for k, v in fields.items()},
                             "effort": effort, "saturation": saturation, "collision": collision,
                             "model_signature": spec.signature})

        respond()
        while True:
            operation, payload = connection.recv()
            if operation == "close":
                break
            if operation == "step":
                targets, velocities = payload
                for i, robot in enumerate(robots):
                    value = step_pd(robot, targets[i], None if velocities is None else velocities[i],
                                    settings=settings)
                    effort[i] = value["effort"]
                    saturation[i] = value["effort_saturation"]
                    collision[i] = float(value["self_collision"])
            elif operation == "reset":
                indices, ref = payload
                for i, index in enumerate(indices):
                    robot = robots[index]
                    mujoco.mj_resetData(robot.model, robot.data)
                    robot.data.qpos[:] = np.concatenate([ref[k][i] for k in
                        ("root_position", "root_orientation", "joint_position")])
                    velocity = ref["root_velocity"][i]
                    robot.data.qvel[:3] = velocity[:3]
                    robot.data.qvel[3:6] = rotation(robot.data.qpos[3:7]).inv().apply(velocity[3:])
                    robot.data.qvel[6:] = ref["joint_velocity"][i]
                    mujoco.mj_forward(robot.model, robot.data)
            else:
                raise ValueError(f"Unknown worker operation: {operation}")
            respond()
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


class ParallelCpuPhysics:
    def __init__(self, spec, num_envs, device, workers=16, action_settings=None):
        import torch
        if workers < 1 or num_envs < 1:
            raise ValueError("Positive worker/world counts required")
        self.spec, self.num_envs, self.device = spec, num_envs, torch.device(device)
        self.action_settings = action_settings
        self.groups = np.array_split(np.arange(num_envs), min(workers, num_envs))
        self.connections, self.processes, self.results = [], [], [None] * len(self.groups)
        self.closed = False
        self.contract = {"backend": "native-mujoco-persistent-processes", "workers": len(self.groups),
                         "control_dt": spec.control_dt, "physics_dt": float(spec.model.opt.timestep),
                         "substeps": spec.substeps, "servo": "k1_motion.servo.step_pd",
                         "precision": "native_mujoco_float64", "benchmark_only": True}
        context = mp.get_context("spawn")
        try:
            for indices in self.groups:
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, len(indices), action_settings), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self._receive(range(len(self.groups)))
        except BaseException:
            self.close()
            raise

    def _receive(self, workers):
        import torch
        for index in workers:
            connection = self.connections[index]
            if not connection.poll(60):
                raise TimeoutError(f"CPU physics worker {index} did not respond")
            result = connection.recv()
            if "error" in result:
                raise RuntimeError(result["error"])
            if result["model_signature"] != self.spec.signature:
                raise ValueError("Worker model differs")
            self.results[index] = result
        # State is cached until physics changes, avoiding repeated transfers for
        # reward/observation access to the same state. New tensors preserve callers'
        # previous-state references across steps and partial resets.
        self.cached = {key: torch.as_tensor(np.concatenate([r["state"][key] for r in self.results]),
                                            device=self.device) for key in self.results[0]["state"]}
        effort = np.concatenate([r["effort"] for r in self.results])
        self.effort_per_env = torch.as_tensor(effort, device=self.device)
        self.self_collision_per_env = torch.as_tensor(
            np.concatenate([r["collision"] for r in self.results]), device=self.device)
        self.effort = float(effort.mean())
        self.saturation = float(np.concatenate([r["saturation"] for r in self.results]).mean())

    def reset(self, ids, ref):
        ids = ids.cpu().numpy()
        if len(ids) == 0:
            return
        fields = ("root_position", "root_orientation", "joint_position", "root_velocity", "joint_velocity")
        host = {k: ref[k].detach().cpu().numpy() for k in fields}
        active = []
        for worker, indices in enumerate(self.groups):
            positions = np.flatnonzero((ids >= indices[0]) & (ids <= indices[-1]))
            if len(positions):
                self.connections[worker].send(("reset", (ids[positions] - indices[0],
                    {k: v[positions] for k, v in host.items()})))
                active.append(worker)
        self._receive(active)

    def step(self, target, velocity=None):
        target = target.detach().cpu().numpy()
        velocity = None if velocity is None else velocity.detach().cpu().numpy()
        for connection, indices in zip(self.connections, self.groups):
            connection.send(("step", (target[indices], None if velocity is None else velocity[indices])))
        self._receive(range(len(self.groups)))

    def state(self):
        return self.cached

    def validate(self):
        import torch
        if any(not torch.isfinite(value).all() for value in self.cached.values()):
            raise FloatingPointError("Nonfinite CPU state")
        if any(not process.is_alive() for process in self.processes):
            raise RuntimeError("CPU worker exited unexpectedly")

    def close(self):
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        for connection in self.connections:
            connection.close()
