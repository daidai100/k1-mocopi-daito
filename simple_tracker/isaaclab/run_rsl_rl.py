"""Run booster_train's RSL-RL train.py / play.py with the simple-tracker tasks registered.

    cd ~/ws/k1-mocopi && OMNI_KIT_ACCEPT_EULA=Y ~/ws/beyondmimic/.venv-isaaclab/bin/python \
        simple_tracker/isaaclab/run_rsl_rl.py train --task K1-Simple-Tracker-v0 --headless

Logs go to ./logs/rsl_rl/k1_simple_tracker/<timestamp> (relative to the cwd).
Using booster_train's scripts unchanged keeps checkpoint/export formats identical to
the earlier single-clip runs, so booster_deploy loads them the same way.
"""

import os
import runpy
import sys

BOOSTER_TRAIN = os.environ.get("BOOSTER_TRAIN", os.path.expanduser("~/ws/beyondmimic/booster_train"))
SCRIPTS = os.path.join(BOOSTER_TRAIN, "scripts", "rsl_rl")

if len(sys.argv) < 2 or sys.argv[1] not in ("train", "play"):
    sys.exit("usage: run_rsl_rl.py {train|play} [booster_train script args...]")
script = os.path.join(SCRIPTS, sys.argv[1] + ".py")
sys.argv = [script] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)

import k1_simple_tracker  # noqa: E402,F401  (registers the gym ids; env cfg imports lazily)

runpy.run_path(script, run_name="__main__")
