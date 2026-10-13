#!/usr/bin/env bash
# First-light llama-bench cells on the RTX 3050 OEM. Each cell waits until the GPU
# is at or below COOL_C, and records GPU temperature, SM clock and power before
# and after. Usage: bench.sh <out_dir>
set -euo pipefail
OUT=${1:?out dir}; mkdir -p "$OUT"
BIN=/home/xbill/llama.cpp/build/bin/llama-bench
MODEL=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
COOL_C=${COOL_C:-40}
gpu() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader,nounits | tr -d ' '; }
cool() { while (( $(gpu | cut -d, -f1) > COOL_C )); do sleep 5; done; }
# name | flags. Serving config is fa1-f16-t6.
CELLS=(
  "fa1-f16-t6|-fa 1 -ctk f16 -ctv f16 -t 6"
  "fa0-f16-t6|-fa 0 -ctk f16 -ctv f16 -t 6"
  "fa1-q8-t6|-fa 1 -ctk q8_0 -ctv q8_0 -t 6"
  "fa1-f16-t4|-fa 1 -ctk f16 -ctv f16 -t 4"
  "fa1-f16-t6-rep|-fa 1 -ctk f16 -ctv f16 -t 6"
)
for c in "${CELLS[@]}"; do
  name=${c%%|*}; flags=${c#*|}
  cool; g0=$(gpu)
  # shellcheck disable=SC2086
  "$BIN" -m "$MODEL" -p 512 -n 128 -r 5 -ngl 99 $flags -o json > "$OUT/$name.json" 2> "$OUT/$name.log"
  g1=$(gpu)
  echo "$name flags=[$flags] start(C,MHz,W)=$g0 end=$g1" | tee -a "$OUT/cells.txt"
done
