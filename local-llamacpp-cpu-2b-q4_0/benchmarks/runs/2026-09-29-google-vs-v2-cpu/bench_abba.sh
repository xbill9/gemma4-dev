#!/bin/bash
# Google's GGUF vs the v2 exact rebuild on the CPU arm's binary, llama-bench, ABBA
# (G,v2 / v2,G / v2,G / G,v2). -t 6 and -t 12 because the server runs -t 6 -tb 12:
# decode at 6 threads, prefill at 12. Each pass waits for the package to cool to <= COOL_C.
set -eu
BENCH=/home/xbill/llama.cpp/build-cpu/bin/llama-bench
G=/home/xbill/models/gemma-4-E2B-it-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
V2=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
COOL_C=${COOL_C:-50}
pkg() { sensors | awk '/Package id 0/{gsub(/[+°C]/,"",$4); print int($4)}'; }
thr() { cat /sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count; }
cool() { sleep 60; until [ "$(pkg)" -le "$COOL_C" ]; do sleep 10; done; }
pass() { cool; echo "pkg before pass $1 ($2 first): $(pkg) C, throttle $(thr)"; shift 2
  CUDA_VISIBLE_DEVICES= $BENCH -m "$1" -m "$2" -ngl 0 -fa 1 -t 6,12 -p 512 -n 128 -r 5 -o md
  echo "pkg after: $(pkg) C, throttle $(thr)"; echo; }
pass 1 google "$G" "$V2"
pass 2 v2 "$V2" "$G"
pass 3 v2 "$V2" "$G"
pass 4 google "$G" "$V2"
