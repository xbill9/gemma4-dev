#!/bin/bash
# Google vs v1 vs v2 on the 1650 Ti, llama-bench, ABBA order (G,v1,v2 / v2,v1,G / v2,v1,G / G,v1,v2),
# card cooled to COOL_C before each pass.
set -eu
BENCH=${BENCH:-/home/xbill/llama.cpp/build/bin/llama-bench}
G=/home/xbill/models/gemma-4-E2B-it-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
V1=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact/gemma-4-E2B-it-q4_0-exact.gguf
V2=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
COOL_C=${COOL_C:-50}
temp() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le "$COOL_C" ]; do sleep 5; done; }
pass() { cool; echo "gpu before pass $1 (order $2): $(temp)"; shift 2
  args=(); for m in "$@"; do args+=(-m "$m"); done
  "$BENCH" "${args[@]}" -ngl 99 -fa 1 -t 6 -p 512 -n 128 -r 5 -o md; echo "gpu after: $(temp)"; echo; }
pass 1 G,v1,v2 "$G" "$V1" "$V2"
pass 2 v2,v1,G "$V2" "$V1" "$G"
pass 3 v2,v1,G "$V2" "$V1" "$G"
pass 4 G,v1,v2 "$G" "$V1" "$V2"
