#!/usr/bin/env bash
set -euo pipefail

dataset=/mnt/storage/k1-motion/datasets/bones-seed
derived=/mnt/storage/k1-motion/derived
archive="$dataset/soma_proportional.tar.gz"
extracted="$dataset/.soma-proportional-extracted"
python=/home/vivi/c/k1-motion/.venv/bin/python
converter=/home/vivi/c/k1-motion/scripts/convert_bones_seed.py
canary_workers=8
workers="${K1_CONVERSION_WORKERS:-12}"
shard_count="${K1_CONVERSION_SHARD_COUNT:-1}"
shard_index="${K1_CONVERSION_SHARD_INDEX:-0}"
production_output="${K1_CONVERSION_OUTPUT:-$derived/bones-seed-k1-gmr-v2}"

# Prevent NumPy/MuJoCo libraries from multiplying threads inside each process.
# Clip-level processes are the intended parallelism boundary.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# The Hugging Face command publishes the archive atomically. Polling this exact
# final path avoids reading its .incomplete file or duplicating the transfer.
while [[ ! -f "$archive" ]]; do
    sleep 30
done

if [[ ! -f "$extracted" ]]; then
    available=$(df --output=avail -B1 "$dataset" | tail -n 1)
    if (( available < 500000000000 )); then
        echo "Refusing extraction with less than 500 GB free on the dataset filesystem" >&2
        exit 1
    fi
    tar --extract --gzip --file "$archive" --directory "$dataset" \
        --no-same-owner --no-same-permissions
    expected=$("$python" - <<'PY'
import pandas as pd
p = "/mnt/storage/k1-motion/datasets/bones-seed/metadata/seed_metadata_v004.parquet"
print(pd.read_parquet(p).move_soma_proportional_path.nunique())
PY
)
    actual=$(find "$dataset/soma_proportional/bvh" -type f -name '*.bvh' -printf . | wc -c)
    if [[ "$actual" != "$expected" ]]; then
        echo "Extracted BVH count mismatch: expected $expected, found $actual" >&2
        exit 1
    fi
    touch "$extracted"
fi

# A category-stratified, full-duration canary must satisfy the same clip and
# category gates before the complete 142,220-row campaign can consume hours.
if [[ "${K1_CONVERSION_SKIP_CANARY:-0}" != 1 ]]; then
    set +e
    "$python" "$converter" \
        --dataset-root "$dataset" \
        --output "$derived/bones-seed-k1-gmr-v2-canary" \
        --canary-per-category 5 \
        --workers "$canary_workers"
    canary_status=$?
    set -e
    if (( canary_status != 0 && canary_status != 2 )); then
        echo "Canary failed operationally with status $canary_status" >&2
        exit "$canary_status"
    fi
    if (( canary_status == 2 )); then
        echo "Canary admission gates failed; continuing conversion with rejects retained as training-ineligible" >&2
    fi
fi

"$python" "$converter" \
    --dataset-root "$dataset" \
    --output "$production_output" \
    --workers "$workers" \
    --shard-count "$shard_count" \
    --shard-index "$shard_index"
