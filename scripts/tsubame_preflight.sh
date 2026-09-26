#!/usr/bin/env bash
# Run inside an existing iqrsh allocation; never starts work on a login node.
set -euo pipefail

: "${JOB_ID:?Enter a TSUBAME allocation with iqrsh first}"
: "${T4TMPDIR:?Allocated local scratch is required}"
: "${K1_ISAAC_PYTHON:?Set this to the qualified Isaac Lab Python executable}"
test -d "$T4TMPDIR"
test -x "$K1_ISAAC_PYTHON"

k1_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
k1_scratch="${T4TMPDIR}/k1-motion-${JOB_ID}"
mkdir -p "$k1_scratch/reports" "$k1_scratch/logs"
k1_persistent="$k1_project_root/artifacts/tsubame-${JOB_ID}"
retain_results() {
  k1_exit_status=$?
  mkdir -p "$k1_persistent"
  rsync -a "$k1_scratch/reports" "$k1_scratch/logs" "$k1_persistent/" || true
  if test -d "$k1_scratch/artifacts"; then
    rsync -a --exclude='references-v7-contact-root' "$k1_scratch/artifacts/" "$k1_persistent/artifacts/" || true
  fi
  printf '%s\n' "$k1_exit_status" > "$k1_persistent/exit-code.txt"
}
trap retain_results EXIT
# Stage hot code/assets and the small reference library, not the 20-hour raw corpus.
rsync -a --exclude='.git' "$k1_project_root/src" "$k1_project_root/scripts" \
  "$k1_project_root/configs" "$k1_project_root/manifests" "$k1_project_root/third_party" "$k1_scratch/"
mkdir -p "$k1_scratch/artifacts"
rsync -a "$k1_project_root/artifacts/references-v7-contact-root" "$k1_scratch/artifacts/"
export K1_MOTION_ROOT="$k1_scratch"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
cd "$k1_scratch"
nvidia-smi -L > reports/gpu-profile.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > reports/gpu.csv
"$K1_ISAAC_PYTHON" scripts/preflight_isaac.py --output reports/isaac-preflight.json > logs/isaac-preflight.log 2>&1
"$K1_ISAAC_PYTHON" scripts/train_isaac.py --library artifacts/references-v7-contact-root \
  --output artifacts/teacher-preflight --iterations 3 --num-envs 32 --horizon 16 --history 10 --hidden-sizes 512 256 --evaluation-interval 0 > logs/teacher-preflight.log 2>&1
"$K1_ISAAC_PYTHON" scripts/train_isaac.py --library artifacts/references-v7-contact-root \
  --output artifacts/student-preflight --stage student --teacher artifacts/teacher-preflight/checkpoint.pt \
  --iterations 3 --num-envs 32 --horizon 16 --history 10 --hidden-sizes 512 256 --evaluation-interval 0 > logs/student-preflight.log 2>&1
printf 'Allocation %s: retaining physics and learner preflights in %s\n' "$JOB_ID" "$k1_persistent"
