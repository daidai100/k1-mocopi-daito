#!/usr/bin/env bash
set -euo pipefail

root=/mnt/ssd512/k1-motion
dataset="$root/dataset"
archive="$dataset/soma_proportional.tar.gz"
extracted="$dataset/.soma-proportional-extracted"
python="$root/source/.venv/bin/python"
decoder="$root/source/.venv/bin/rapidgzip"
converter="$root/source/scripts/convert_bones_seed.py"
output=/mnt/ssd1/k1-motion/derived/bones-seed-k1-gmr-v3-shard1
workers="${K1_CONVERSION_WORKERS:-52}"
decompression_workers="${K1_DECOMPRESSION_WORKERS:-24}"

while [[ ! -f "$archive" ]]; do sleep 10; done
if [[ ! -f "$extracted" ]]; then
    if [[ -x "$decoder" ]]; then
        echo "Extracting BONES-SEED with $decompression_workers decoder workers"
        "$decoder" --decompress --stdout -P "$decompression_workers" "$archive" | \
            tar --extract --file - --directory "$dataset" \
                --no-same-owner --no-same-permissions
    else
        tar --extract --gzip --file "$archive" --directory "$dataset" \
            --no-same-owner --no-same-permissions
    fi
    expected=$("$python" - <<'PY'
import pandas as pd
p = "/mnt/ssd512/k1-motion/dataset/metadata/seed_metadata_v004.parquet"
print(pd.read_parquet(p).move_soma_proportional_path.nunique())
PY
)
    actual=$(find "$dataset/soma_proportional/bvh" -type f -name '*.bvh' -printf . | wc -c)
    [[ "$actual" == "$expected" ]] || {
        echo "Extracted BVH count mismatch: expected $expected, found $actual" >&2
        exit 1
    }
    touch "$extracted"
    echo "BONES-SEED extraction verified: $actual BVH files"
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "Starting BONES-SEED shard 1 of 2 with $workers conversion workers"
"$python" "$converter" \
    --dataset-root "$dataset" \
    --output "$output" \
    --workers "$workers" \
    --shard-count 2 \
    --shard-index 1
