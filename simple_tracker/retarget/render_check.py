"""Offscreen render of retargeted K1 clips next to the scaled human skeleton (no GUI needed).

    MUJOCO_GL=egl <GMR venv python> simple_tracker/retarget/render_check.py \
        --clips ~/ws/k1-mocopi-data/k1_bandai/dataset-2_walk_normal_001.npz --bandai ~/ws/k1-mocopi-data/bandai/dataset \
        --out check.mp4
"""

import argparse
from pathlib import Path

import imageio
import mujoco as mj
import numpy as np

from bandai_to_k1 import K1_HIP_TO_ANKLE, leg_length, load_bandai, params

EDGES_COLOR = np.array([1.0, 0.3, 0.2, 0.8], dtype=np.float32)


def add_sphere(scn, pos, rgba, size=0.02):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]), pos, np.eye(3).flatten(), rgba)
    scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", type=Path, required=True)
    ap.add_argument("--bandai", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    a = ap.parse_args()

    model = mj.MjModel.from_xml_path(str(params.ROBOT_XML_DICT["booster_k1"]))
    data = mj.MjData(model)
    renderer = mj.Renderer(model, 480, 640)
    cam = mj.MjvCamera()
    cam.distance, cam.elevation = 2.0, -15
    frames = []
    for clip in a.clips:
        motion = np.load(clip)
        bvh = next(a.bandai.glob(f"*/data/{clip.stem}.bvh"))
        hpos, _, bones, hfps = load_bandai(str(bvh))
        scale = K1_HIP_TO_ANKLE / leg_length(hpos, bones)
        hip = bones.index("Hips")
        n = min(len(motion["dof_pos"]), int(a.seconds * 50))
        for t in range(0, n, 2):
            data.qpos[:3] = motion["root_pos"][t]
            data.qpos[3:7] = motion["root_quat"][t]
            data.qpos[7:] = motion["dof_pos"][t]
            mj.mj_kinematics(model, data)
            cam.lookat[:] = data.qpos[:3]
            cam.azimuth = 150
            renderer.update_scene(data, cam)
            # human skeleton, scaled about the hips and placed 0.6 m to the side
            th = min(int(t / 50 * hfps), len(hpos) - 1)
            root = hpos[th, hip]
            for b in range(len(bones)):
                p = (hpos[th, b] - root) * scale + data.qpos[:3] + np.array([0.0, 0.6, 0.0])
                add_sphere(renderer.scene, p, EDGES_COLOR)
            frames.append(renderer.render())
    imageio.mimsave(a.out, frames, fps=25)
    print("wrote", a.out, len(frames))


if __name__ == "__main__":
    main()
