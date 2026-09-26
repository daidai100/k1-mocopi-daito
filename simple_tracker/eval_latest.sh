#!/usr/bin/env bash
# Export the newest (or given) checkpoint and run headless MuJoCo sim2sim on a clip set.
#   simple_tracker/eval_latest.sh [model_N.pt] [clip regex]
# Outputs (next to the checkpoint): exported/policy_N.{pt,onnx}, sim2sim_N.json, sim2sim_N.mp4,
# plus sim2sim_mocopi_N_<clip>.mp4 for the causal mocopi test clips.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN=${RUN:-$(ls -td "$REPO"/logs/rsl_rl/k1_simple_tracker/*/ | head -1)}
CKPT=${1:-$(ls -v "$RUN"/model_*.pt | tail -1)}
CLIPS=${2:-'_normal_001$'}
N=$(basename "$CKPT" .pt | sed 's/model_//')
DATA=${DATA:-$HOME/ws/k1-mocopi-data}
"$HOME"/ws/beyondmimic/.venv-isaaclab/bin/python "$REPO"/simple_tracker/isaaclab/export_policy.py "$CKPT"
POLICY="$RUN/exported/policy_$N.pt"
cd "$HOME"/ws/beyondmimic/booster_deploy
export MUJOCO_GL=egl
.venv/bin/python "$REPO"/simple_tracker/deploy/sim2sim_eval.py --checkpoint "$POLICY" --clips "$CLIPS" \
  --json "$RUN/sim2sim_$N.json" --video "$RUN/sim2sim_$N.mp4" 2>&1 | grep -E "^dataset|SUMMARY"
for f in "$DATA"/mocopi_live/*.npz; do
  .venv/bin/python "$REPO"/simple_tracker/deploy/sim2sim_eval.py --checkpoint "$POLICY" --library "$f" --clips . \
    --video "$RUN/sim2sim_mocopi_${N}_$(basename "$f" .npz).mp4" 2>&1 | grep -E "SUMMARY"
done
