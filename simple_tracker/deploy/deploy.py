"""Run the simple tracker through booster_deploy (MuJoCo viewer or the real K1).

    cd ~/ws/beyondmimic/booster_deploy && .venv/bin/python ~/ws/k1-mocopi/simple_tracker/deploy/deploy.py \
        --checkpoint <exported policy.pt> --clips 'dataset-2_walk_normal_001$' [--mujoco]

    # live mocopi (start live/mocopi_retarget.py udp first; stand still for the first second)
    ... deploy.py --checkpoint <policy.pt> --source udp

Without ``--mujoco`` this commands the real robot via booster_deploy's
BoosterRobotPortal, exactly like ``booster_deploy/scripts/deploy.py``.
"""

import argparse
import os
import sys

BOOSTER_DEPLOY = os.environ.get("BOOSTER_DEPLOY", os.path.expanduser("~/ws/beyondmimic/booster_deploy"))
sys.path.insert(0, BOOSTER_DEPLOY)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_LIBRARY = os.path.expanduser("~/ws/k1-mocopi-data/k1_bandai_library.npz")


def build_cfg(a):
    from simple_tracker_policy import K1SimpleTrackerControllerCfg

    cfg = K1SimpleTrackerControllerCfg()
    cfg.policy.checkpoint_path = os.path.abspath(a.checkpoint)
    cfg.policy.reference_source = getattr(a, "source", "clip")
    cfg.policy.udp_port = getattr(a, "udp_port", 12400)
    cfg.policy.library_path = os.path.abspath(a.library)
    cfg.policy.clip_regex = a.clips
    cfg.policy.max_clips = a.max_clips
    cfg.policy.hold_s = a.hold
    cfg.policy.device = a.device
    if getattr(a, "exit_mode", None):
        cfg.booster.exit_mode = a.exit_mode
    return cfg


def add_args(p):
    p.add_argument("--checkpoint", required=True, help="exported TorchScript policy (.pt) or .onnx")
    p.add_argument("--library", default=DEFAULT_LIBRARY)
    p.add_argument("--clips", default=r"dataset-2_walk_normal_001$", help="regex over clip names, played in order")
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--hold", type=float, default=2.0, help="seconds to hold the first/last frame")
    p.add_argument("--device", default="cpu")


def main():
    p = argparse.ArgumentParser()
    add_args(p)
    p.add_argument("--mujoco", action="store_true")
    p.add_argument("--source", choices=("clip", "udp"), default="clip", help="udp: live/mocopi_retarget.py stream")
    p.add_argument("--udp-port", type=int, default=12400)
    p.add_argument("--exit-mode", choices=("walking", "damping"), default=None)
    a = p.parse_args()
    cfg = build_cfg(a)
    if a.mujoco:
        from booster_deploy.controllers.mujoco_controller import MujocoController

        MujocoController(cfg).run()
    else:
        from booster_deploy.controllers.booster_robot_controller import BoosterRobotPortal

        with BoosterRobotPortal(cfg) as portal:
            portal.run()


if __name__ == "__main__":
    main()
