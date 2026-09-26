#!/usr/bin/env python3
"""Read-only full-reference initializer replay, command tracing and reward calibration."""
# ruff: noqa: E402 - resolve the checkout before importing its package
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import mujoco

ROOT = Path(__file__).resolve().parents[1]
os.environ['K1_MOTION_ROOT'] = str(ROOT)
sys.path.insert(0, str(ROOT/'src'))
from k1_motion import actuators, runtime, control_validation
from k1_motion.contracts import MotionClip
from k1_motion.export import export_checkpoint
from k1_motion.learning import Policy
from k1_motion.robot import K1Model
from k1_motion.world_objective import WorldBodyTracking


def main():
    torch.set_num_threads(1)
    base = ROOT/'artifacts/easy-walk-campaign-20260923'
    bundle, output = base/'bundle', base/'initializer-audit'
    output.mkdir(exist_ok=True)
    export_checkpoint(bundle/'initialize.pt', output/'actor.pt')
    robot = K1Model()
    policy = Policy(output/'actor.pt', robot.signature)
    rewards = {p: WorldBodyTracking(p) for p in ('world-body-v1', 'world-decomposed-v1')}
    originals = (runtime.targets_tensor, runtime.position_guard_tensor,
                 actuators.safety_sample, control_validation.step_command)
    rows = [json.loads(s) for s in (bundle/'library/index.jsonl').read_text().splitlines()]
    reports = []
    for row in rows:
        clip = MotionClip.load(bundle/'library'/row['reference_path'])
        commands, guards, substeps, components = [], [], [], []

        def targets(*args, **kwargs):
            value = originals[0](*args, **kwargs)
            action, ref = args[:2]
            commands.append(dict(action=action[0].numpy().copy(),
                reference_joint=ref['joint_position'][0].numpy().copy(),
                target_after_joint_clamp=value[0].numpy().copy()))
            return value

        def guard(*args, **kwargs):
            value = originals[1](*args, **kwargs)
            guards.append(dict(before=args[0][0].numpy().copy(), after=value[0][0].numpy().copy(),
                velocity_before=args[1][0].numpy().copy(), velocity_after=value[1][0].numpy().copy()))
            return value

        def safety(*args, **kwargs):
            value = originals[2](*args, **kwargs)
            q, dq, requested, available = args[:4]
            collision = any(c.dist <= 0 and 0 not in robot.model.geom_bodyid[[c.geom1, c.geom2]]
                            for c in robot.data.contact[:robot.data.ncon])
            substeps.append(dict(time_s=(len(substeps)+1)*.002, q=q.copy(), dq=dq.copy(),
                requested=requested.copy(), applied=robot.data.ctrl.copy(), available=available.copy(),
                safety=value.copy(), collision=collision))
            return value

        def step(*args, **kwargs):
            result = originals[3](*args, **kwargs)
            # Match replay's post-step forward pass before measuring landmarks.
            mujoco.mj_forward(robot.model, robot.data)
            index = len(components)+1
            state = dict(q=robot.data.qpos[7:], position=robot.data.qpos[:3],
                orientation=robot.data.qpos[3:7], omega=robot.data.qvel[3:6],
                velocity=robot.data.qvel[:3], landmarks=robot.landmarks())
            state = {k: torch.tensor(v.copy(), dtype=torch.float32)[None] for k, v in state.items()}
            ref = {k: torch.tensor(clip.values[k][index].copy(), dtype=torch.float32)[None]
                   for k in ('joint_position', 'root_position', 'root_orientation', 'root_velocity', 'landmarks')}
            parts = {}
            for name, reward in rewards.items():
                _, values = reward.step(state, ref, None)
                parts[name] = {k: float(v[0]) for k, v in values.items()}
            components.append(parts)
            return result

        runtime.targets_tensor, runtime.position_guard_tensor = targets, guard
        actuators.safety_sample, control_validation.step_command = safety, step
        try:
            result = control_validation.replay_clip(robot, policy, clip, output/f"{row['id']}.npz")
        finally:
            runtime.targets_tensor, runtime.position_guard_tensor, actuators.safety_sample, control_validation.step_command = originals
        array = {f'command_{k}': np.asarray([r[k] for r in commands]) for k in commands[0]}
        array.update({f'guard_{k}': np.asarray([r[k] for r in guards]) for k in guards[0]})
        array.update({f'substep_{k}': np.asarray([r[k] for r in substeps]) for k in substeps[0]})
        np.savez_compressed(output/f"{row['id']}-authority.npz", **array)
        sample = array['substep_safety']
        bad = (sample[:, 0] > 0) | (sample[:, 4] > 0) | array['substep_collision']
        first = int(np.flatnonzero(bad)[0]) if bad.any() else None
        means = {name: {k: float(np.mean([r[name][k] for r in components])) for k in components[0][name]}
                 for name in rewards}
        saturated = np.abs(array['substep_requested']) > array['substep_available']
        settings = policy.metadata['action_settings']
        ideal_servo_offset = robot.kd/robot.kp*(1-settings['target_velocity_scale'])*clip.values['joint_velocity']
        feet_velocity = np.diff(clip.values['landmarks'][:, [11, 15]], axis=0)/.02
        stance = clip.values['contacts'][1:] > .8
        report = dict(id=row['id'], family=row['family'], replay=result,
            first_safety_event_s=None if first is None else float(array['substep_time_s'][first]),
            per_joint_saturation_seconds=(saturated.sum(0)*.002).tolist(),
            reward_component_means=means,
            max_ideal_servo_offset_lower_rad=float(np.max(np.abs(ideal_servo_offset[:, 10:]))),
            ideal_servo_offset_exceeds_residual_lower_fraction=float(np.mean(np.abs(ideal_servo_offset[:, 10:]) > .25)),
            stance_landmark_speed_p95_m_s=float(np.quantile(np.linalg.norm(feet_velocity, axis=-1)[stance], .95)) if stance.any() else None,
            command_trace='actor, hard-limit target, post-arm/slew pre-margin and post-margin targets; requested/applied torque and safety at 500Hz',
            authority_equation='tau_req = kp*(target-q) + kd*(0.25*reference_dq-dq); lower residual +/-0.25rad; upper residual zero',
            authority_boundary='ideal servo offset excludes inertial/gravity/contact loads; no gain or authority change justified by this alone',
            early_termination_audit='negative tracking share recorded; large-lag negative-tail risk remains a promotion blocker')
        (output/f"{row['id']}.json").write_text(json.dumps(report, indent=2)+'\n')
        reports.append(report)
        print(row['id'], result['completed'], result['clean_success'], result['absolute_motion_v1']['clean'], flush=True)
    (output/'report.json').write_text(json.dumps(dict(scope='training-only original-timing initializer diagnostics',
        runs=reports, confirmation_panel_consumed=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
