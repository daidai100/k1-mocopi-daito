#!/usr/bin/env python3
"""Compare GPU MuJoCo with CPU dynamics, including velocities and partial resets."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from k1_motion.robot import K1Model  # noqa: E402
from k1_motion.tracking_env import MujocoPhysics  # noqa: E402
from k1_motion.warp_physics import WarpPhysics  # noqa: E402
from k1_motion.observations import targets_tensor  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--no-conditional-graphs", action="store_true")
    parser.add_argument("--epa-horizon", type=int)
    parser.add_argument("--action-settings")
    parser.add_argument("--target-velocity", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = K1Model()
    settings = json.loads(Path(args.action_settings).read_text()) if args.action_settings else None
    started = time.monotonic()
    gpu = WarpPhysics(
        spec, 2, "cuda:0", conditional_graphs=not args.no_conditional_graphs, epa_horizon=args.epa_horizon,
        action_settings=settings,
    )
    cpu = MujocoPhysics(spec, 2, torch.device("cpu"), action_settings=settings)
    r = spec.neutral_reference()
    ref = {
        k: torch.tensor(np.tile(getattr(r, k), (2, 1)), dtype=torch.float32, device="cuda:0")
        for k in ("root_position", "root_orientation", "root_velocity", "joint_position", "joint_velocity")
    }
    # A yawed, moving root checks world linear velocity versus body gyro order.
    ref["root_orientation"][1] = torch.tensor([np.cos(0.4), 0, 0, np.sin(0.4)], device="cuda:0")
    ref["root_velocity"][1] = torch.tensor([0.03, -0.01, 0.0, 0.1, -0.05, 0.2], device="cuda:0")
    ref["joint_velocity"][1, 3] = 0.2
    ids = torch.arange(2, device="cuda:0")
    gpu.reset(ids, ref)
    cpu.reset(ids, ref)
    initial = {k: float((gpu.state()[k].cpu() - cpu.state()[k]).abs().max()) for k in gpu.state()}
    assert max(initial.values()) < 1e-5, initial
    maximum = {k: 0.0 for k in ("q", "dq", "position", "orientation", "landmarks")}
    locations, cpu_trace, gpu_trace, contacts = {}, [], [], []
    efforts, saturations = [], []
    prior = {**{k: v.cpu() for k, v in ref.items()}, "contacts": torch.ones((2, 2))}
    for tick in range(args.steps):
        state = cpu.state()
        # Use the shared standing stabilizer so parity remains in its declared
        # contact envelope. Open-loop neutral PD eventually falls into an impact.
        targets = targets_tensor(
            torch.zeros((2, 22)),
            prior,
            state["orientation"],
            state["omega"],
            torch.tensor(spec.limits, dtype=torch.float32),
        ).to("cuda:0")
        targets[:, 2] += 0.04 * np.sin(tick * spec.control_dt * 2)
        velocity = torch.zeros_like(targets)
        if args.target_velocity:
            velocity[:, 2] = 0.08 * np.cos(tick * spec.control_dt * 2)
        gpu.step(targets, velocity)
        cpu.step(targets, velocity)
        # Both backends are compared at the integrated state, not a stale FK cache.
        for robot in cpu.robots:
            mujoco.mj_forward(robot.model, robot.data)
        a, b = gpu.state(), cpu.state()
        cpu_trace.append(np.concatenate([b["q"].numpy(), b["dq"].numpy()], -1))
        gpu_trace.append(np.concatenate([a["q"].cpu().numpy(), a["dq"].cpu().numpy()], -1))
        contacts.append(
            [
                [[int(c.geom1), int(c.geom2), float(c.dist)] for c in robot.data.contact[: robot.data.ncon]]
                for robot in cpu.robots
            ]
        )
        for key in maximum:
            error = (a[key].cpu() - b[key]).abs()
            if float(error.max()) > maximum[key]:
                maximum[key] = float(error.max())
                locations[key] = {
                    "tick": tick,
                    "index": list(map(int, np.unravel_index(int(error.argmax()), error.shape))),
                }
        efforts.append(abs(float(gpu.effort) - cpu.effort))
        saturations.append(abs(float(gpu.saturation) - cpu.saturation))
    gpu.validate()
    limits = {"q": 0.02, "dq": 0.15, "position": 0.01, "orientation": 0.01, "landmarks": 0.01}
    dynamics_passed = all(maximum[k] <= limits[k] for k in maximum)
    before = {k: v[1].clone() for k, v in gpu.state().items()}
    first = {k: v[:1] for k, v in ref.items()}
    gpu.reset(ids[:1], first)
    after = gpu.state()
    partial_reset_passed = all(torch.equal(before[k], after[k][1]) for k in before)
    assert partial_reset_passed, "Reset changed a different world"
    contact_ref = {k: v.clone() for k, v in ref.items()}
    contact_ref["root_velocity"].zero_()
    contact_ref["joint_velocity"].zero_()
    contact_ref["root_position"][1, 2] += 1.0
    contact_ref["joint_position"][1, 7] = 1.629  # Right arm/trunk penetration, above ground.
    gpu.reset(ids, contact_ref)
    cpu.reset(ids, contact_ref)
    gpu.step(contact_ref["joint_position"])
    cpu.step(contact_ref["joint_position"])
    expected_contact = torch.tensor([0.0, 1.0])
    torch.testing.assert_close(cpu.self_collision_per_env, expected_contact)
    torch.testing.assert_close(gpu.self_collision_per_env.cpu(), expected_contact)
    gpu.validate()
    # An overflow cannot disappear through an episode reset before the learner checks it.
    gpu.overflow[0] = 1
    gpu.reset(ids[:1], first)
    try:
        gpu.validate()
    except RuntimeError as error:
        assert "overflow" in str(error)
        overflow_latched = True
    else:
        raise AssertionError("Reset hid a physics overflow")
    gpu.overflow_latch.zero_()  # Clear only this deliberately injected preflight error.
    gpu.close()
    report = {
        "backend": "warp",
        "device": torch.cuda.get_device_name(),
        "steps": args.steps,
        "target_velocity_tested": args.target_velocity,
        "contract": gpu.contract,
        "initial_max_abs_error": initial,
        "trajectory_max_abs_error": maximum,
        "max_error_locations": locations,
        "limits": limits,
        "max_effort_difference": max(efforts),
        "max_saturation_difference": max(saturations),
        "partial_reset_passed": partial_reset_passed,
        "self_contact_detection_passed": True,
        "overflow_latched": overflow_latched,
        "dynamics_passed": dynamics_passed,
        "elapsed_seconds": time.monotonic() - started,
        "scope": "Bounded CPU/GPU dynamics parity; not learned-policy acceptance",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(path.with_suffix(".npz"), cpu=cpu_trace, gpu=gpu_trace)
    path.with_suffix(".contacts.json").write_text(json.dumps(contacts) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not dynamics_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
