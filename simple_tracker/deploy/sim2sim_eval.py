"""Headless MuJoCo sim2sim: success rate / tracking error per clip, optional mp4 with reference ghost.

Uses booster_deploy's MujocoController physics (K1_22dof.xml, 500 Hz PD, 50 Hz policy)
but drives it without the GLFW viewer, which does not open on this machine.

    cd ~/ws/beyondmimic/booster_deploy && MUJOCO_GL=egl .venv/bin/python \
        ~/ws/k1-mocopi/simple_tracker/deploy/sim2sim_eval.py --checkpoint <policy.pt> \
        --clips 'normal_00[1-3]$' --video out.mp4 --json out.json
"""

import argparse
import json
import re

import numpy as np

from deploy import add_args, build_cfg

FALL_HEIGHT = 0.30


def run_clip(cfg, renderer=None, frames=None):
    import mujoco
    import torch
    from booster_deploy.controllers.mujoco_controller import MujocoController

    ctrl = MujocoController(cfg)
    ctrl.update_state()
    ctrl.start()
    policy = ctrl.policy
    ref = policy.reference
    joint_err, height_err, fell = [], [], False
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 2.2, -15, 135
    opt = mujoco.MjvOption()
    while ctrl.is_running and not ref.done:
        ctrl.update_state()
        targets = ctrl.policy_step()
        ctrl.ctrl_step(targets)
        s2r = ctrl.robot.data.sim2real_joint_indexes
        q = torch.from_numpy(ctrl.mj_data.qpos[7:].astype(np.float32))
        joint_err.append(float((q - policy.ref_joint_pos[s2r].cpu()).abs().mean()))
        height_err.append(abs(float(ctrl.mj_data.qpos[2]) - float(policy.ref_root_pos[2])))
        if ctrl.mj_data.qpos[2] < FALL_HEIGHT:
            fell = True
            break
        if renderer is not None and ctrl._step_count % 2 == 0:
            # draw the reference at the robot's xy/heading: the policy tracks neither, by design
            from booster_deploy.utils.isaaclab import math as lab_math

            robot_q = torch.from_numpy(ctrl.mj_data.qpos[3:7].astype(np.float32))
            ref_q = policy.ref_root_quat.cpu()
            q = lab_math.quat_mul(lab_math.quat_mul(lab_math.yaw_quat(robot_q), lab_math.quat_inv(lab_math.yaw_quat(ref_q))), ref_q)
            pos = torch.tensor([*ctrl.mj_data.qpos[:2], float(policy.ref_root_pos[2])])
            pos[1] += 0.5  # side by side
            ctrl.set_reference_qpos(torch.cat([pos, q, policy.ref_joint_pos[s2r].cpu()]))
            cam.lookat[:] = ctrl.mj_data.qpos[:3]
            cam.lookat[1] += 0.25
            renderer.update_scene(ctrl.mj_data, cam)
            n0 = renderer.scene.ngeom
            mujoco.mjv_addGeoms(
                ctrl.mj_model, ctrl._ghost_mj_data, opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC,
                renderer.scene,
            )
            for i in range(n0, renderer.scene.ngeom):
                renderer.scene.geoms[i].rgba[:] = (0.2, 0.8, 0.2, 0.3)
            frames.append(renderer.render())
    stopped_early = not ctrl.is_running and not ref.done
    return {
        "fell": bool(fell or stopped_early),
        "seconds": ctrl._step_count * 0.02,
        "joint_err_mean": float(np.mean(joint_err)) if joint_err else None,
        "height_err_mean": float(np.mean(height_err)) if height_err else None,
    }


def main():
    p = argparse.ArgumentParser()
    add_args(p)
    p.add_argument("--video", default=None)
    p.add_argument("--json", default=None)
    a = p.parse_args()

    names = np.load(a.library)["clip_names"].tolist()
    clips = [n for n in names if re.search(a.clips, n)]
    if a.max_clips:
        clips = clips[: a.max_clips]
    renderer, frames = None, []
    results = {}
    for name in clips:
        a.clips = "^" + re.escape(name) + "$"
        cfg = build_cfg(a)
        if a.video and renderer is None:
            import mujoco
            from booster_deploy.controllers.mujoco_controller import MujocoController

            renderer = mujoco.Renderer(MujocoController(cfg).mj_model, 480, 640)
        results[name] = run_clip(cfg, renderer, frames)
        print(name, results[name], flush=True)
    n = len(results)
    summary = {
        "clips": n,
        "success_rate": sum(not r["fell"] for r in results.values()) / max(n, 1),
        "joint_err_mean": float(np.mean([r["joint_err_mean"] for r in results.values() if r["joint_err_mean"]])),
    }
    print("SUMMARY", summary)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": summary, "clips": results}, f, indent=1)
    if a.video and frames:
        import imageio

        imageio.mimsave(a.video, frames, fps=25)
        print("wrote", a.video)


if __name__ == "__main__":
    main()
