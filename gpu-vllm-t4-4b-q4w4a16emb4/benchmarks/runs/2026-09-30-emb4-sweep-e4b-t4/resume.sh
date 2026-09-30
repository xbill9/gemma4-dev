#!/usr/bin/env bash
# Finish an interrupted sweep.sh: the same grid, seeds and flags, skipping every
# cell that already has a result JSON. sweep.sh itself stays verbatim.
#
#   bash resume.sh emb4 xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text-emb4
#
# Written 2026-09-30 when the first pass died at c4-in4096 rep 2 (4/16 requests
# sent; its log is evidence/aborted-c4-in4096-out128.rep2.log). The server was
# restarted before resuming, because the aborted cell had already put seed
# 240964's prompts in the prefix cache and rerunning it on that server would
# have measured cache hits rather than the model.
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
      [ -f "$out/$cell.json" ] && continue
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
