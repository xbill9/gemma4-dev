#!/bin/bash
# per_layer_token_embd on the host (default: TENSOR_READ_LAZY, GET_ROWS out of the mmap)
# vs forced onto the card with -ot. v2 exact GGUF, rig binary, same flags as `make serve`
# otherwise. Step 1 proves where the tensor landed from llama-server's allocation log;
# step 2 is llama-bench in ABBA order (host, card, card, host), card cooled first.
set -eu
cd "$(dirname "$0")"
BIN=/home/xbill/llama.cpp/build/bin
M=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
OT=(-ot 'per_layer_token_embd\.weight=CUDA0')
COOL_C=${COOL_C:-50}
temp() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le "$COOL_C" ]; do sleep 5; done; }

alloc() {  # name, extra args...
  local n=$1; shift
  CUDA_VISIBLE_DEVICES=0 $BIN/llama-server -lv 4 -m $M --host 127.0.0.1 --port 8099 -ngl 99 -c 8192 \
    -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 --parallel 1 "$@" > alloc-$n.log 2>&1 &
  local pid=$!
  until curl -fs localhost:8099/health >/dev/null; do sleep 1; kill -0 $pid 2>/dev/null || { echo "$n: server died"; return 1; }; done
  echo "== $n: nvidia-smi $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
  grep -E 'buffer size|per_layer_token_embd' alloc-$n.log | sed 's/^[0-9.]* //'
  kill $pid; wait $pid 2>/dev/null || true
}
alloc host
alloc card "${OT[@]}"
echo

pass() {  # n, label, extra args...
  local n=$1 l=$2; shift 2
  cool; echo "gpu before pass $n ($l): $(temp)"
  CUDA_VISIBLE_DEVICES=0 $BIN/llama-bench -m $M -ngl 99 -fa 1 -t 6 -p 512,2048 -n 128 -r 5 -o md "$@"
  echo "gpu after: $(temp)"; echo
}
pass 1 host
pass 2 card "${OT[@]}"
pass 3 card "${OT[@]}"
pass 4 host
