#!/usr/bin/env python3
"""Shared PPO launcher using compiled parallel CPU MuJoCo and a CPU/GPU learner."""
import sys

from train_warp import main

if __name__ == "__main__":
    if "--backend" in sys.argv:
        raise SystemExit("train_cpu.py selects mujoco_cpp; use train_warp.py for backend selection")
    sys.argv.extend(["--backend", "mujoco_cpp"])
    main()
