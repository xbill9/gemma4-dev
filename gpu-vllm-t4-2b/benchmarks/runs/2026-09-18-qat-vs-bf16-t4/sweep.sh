#!/usr/bin/env bash
# Sweep one checkpoint already being served at 127.0.0.1:8000.
#
#   bash sweep.sh <label> <served-model-id>
#   bash sweep.sh bf16 google/gemma-4-E2B-it
#   bash sweep.sh qat  google/gemma-4-E2B-it-qat-w4a16-ct
#
# Grid: input {512, 4096} x concurrency {1, 4, 8, 16}, output 128, 3 repeats.
# 512/128 at c=1..16 is the g4dn twin's shape (2026-08-30-first-serve-g4dn), so
# those cells line up with it; 4096 is added to see whether the larger KV pool
# the QAT build leaves ever matters at this rig's serving shape.
#
# EVERY CELL AND REPEAT GETS ITS OWN SEED, and the seed does not depend on the
# label, so both checkpoints see the same prompts cell for cell while no two
# requests within one server lifetime share a prefix. gpu-vllm-mi300x-2b
# measured the prefix cache instead of the model when seeds collided (2.25x at
# 8192 context) — see its -seedcollision run.
set -euo pipefail

label=$1
model=$2
here=$(cd "$(dirname "$0")" && pwd)
out="$here/$label"
mkdir -p "$out"

export PYTHONUSERBASE=/opt1/pyuser TMPDIR=/opt1/tmp
VLLM=/opt1/pyuser/bin/vllm

for rep in 1 2 3; do
  for in_len in 512 4096; do
    for c in 1 4 8 16; do
      cell="c${c}-in${in_len}-out128.rep${rep}"
      seed=$((rep * 100000 + in_len * 10 + c))
      n=$((c * 4 > 8 ? c * 4 : 8))
      echo "== $label $cell seed=$seed prompts=$n"
      "$VLLM" bench serve \
        --backend vllm --base-url http://127.0.0.1:8000 --model "$model" \
        --dataset-name random --random-input-len "$in_len" --random-output-len 128 \
        --num-prompts "$n" --max-concurrency "$c" --ignore-eos --temperature 0 \
        --num-warmups 2 --seed "$seed" \
        --save-result --result-dir "$out" --result-filename "$cell.json" \
        > "$out/$cell.log" 2>&1
      grep -E "Output token throughput|Median TPOT" "$out/$cell.log" | tr -s ' ' | paste -sd' '
    done
  done
done
