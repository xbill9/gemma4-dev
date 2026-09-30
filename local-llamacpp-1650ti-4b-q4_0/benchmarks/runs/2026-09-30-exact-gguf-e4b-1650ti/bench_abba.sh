#!/bin/bash
# E4B: Google's GGUF vs the exact Q4_0 rebuild on the 1650 Ti, llama-bench, ABBA order
# (G,X / X,G / X,G / G,X), card cooled to COOL_C before each pass. Same flags as the
# E2B rig's 2026-09-29-exact-gguf-v2-1650ti/bench_abba.sh.
set -eu
BENCH=${BENCH:-/home/xbill/llama.cpp/build/bin/llama-bench}
G=/home/xbill/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-gguf/snapshots/4b4a2c1d584be7264f87aac328a1bc739ce81b6c/gemma-4-E4B_q4_0-it.gguf
X=/home/xbill/models/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-q4_0-exact.gguf
COOL_C=${COOL_C:-50}
temp() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le "$COOL_C" ]; do sleep 5; done; }
pass() { cool; echo "gpu before pass $1 (order $2): $(temp)"; shift 2
  args=(); for m in "$@"; do args+=(-m "$m"); done
  "$BENCH" "${args[@]}" -ngl 99 -fa 1 -t 6 -p 512 -n 128 -r 5 -o md; echo "gpu after: $(temp)"; echo; }
pass 1 G,X "$G" "$X"
pass 2 X,G "$X" "$G"
pass 3 X,G "$X" "$G"
pass 4 G,X "$G" "$X"
