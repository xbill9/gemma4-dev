#!/bin/bash
# Google's file vs the exact rebuild on the 1650 Ti, llama-bench, ABBA order.
# Each pass runs both files; passes alternate which goes first. Before every pass the
# card cools to COOL_C so passes start from the same temperature.
set -eu
BENCH=${BENCH:-/home/xbill/llama.cpp/build/bin/llama-bench}
G=/home/xbill/models/gemma-4-E2B-it-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
X=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact/gemma-4-E2B-it-q4_0-exact.gguf
COOL_C=${COOL_C:-50}
temp() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le "$COOL_C" ]; do sleep 5; done; }
pass() {
  cool; echo "gpu before pass $1 ($2 first): $(temp)"
  "$BENCH" -m "$3" -m "$4" -ngl 99 -fa 1 -t 6 -p 512 -n 128 -r 5 -o md
  echo "gpu after pass $1: $(temp)"; echo
}
pass 1 google "$G" "$X"
pass 2 exact  "$X" "$G"
pass 3 exact  "$X" "$G"
pass 4 google "$G" "$X"
