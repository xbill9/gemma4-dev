#!/bin/bash
# First light for E4B on the RTX 3050 OEM, same binary as the E2B sister rig.
# 1) allocation at -c 131072: per-layer table on host vs on card, then card + drafter.
# 2) llama-bench ABBA (host, card, card, host), pp512/pp2048/tg128, r=5, cooled to <=40 C.
# 3) llama-bench by depth with the table on the card, r=3.
set -eu
cd "$(dirname "$0")"
BIN=/home/xbill/llama.cpp/build/bin
M=/home/xbill/models/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-q4_0-exact.gguf
DR=/home/xbill/models/gemma-4-E4B-it-qat-assistant/gemma-4-E4B-it-qat-assistant-q8_0.gguf
OT=(-ot 'per_layer_token_embd\.weight=CUDA0')
temp() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 40 ]; do sleep 5; done; }
alloc() {  # name, extra args...
  local n=$1; shift
  $BIN/llama-server -lv 4 -m $M --host 127.0.0.1 --port 8099 -ngl 99 -c 131072 -ctk f16 -ctv f16 -fa 1 \
    -t 6 -tb 12 --parallel 1 "$@" > alloc-$n.log 2>&1 &
  local pid=$!
  until curl -fs localhost:8099/health >/dev/null; do sleep 1; kill -0 $pid 2>/dev/null || { echo "$n: server died"; return 1; }; done
  echo "== $n: nvidia-smi $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
  grep -E 'buffer size|n_ctx_train' alloc-$n.log | sed 's/^[0-9.]* //'
  kill $pid; wait $pid 2>/dev/null || true
}
alloc host
alloc card "${OT[@]}"
alloc card-draft "${OT[@]}" -md $DR --spec-type draft-mtp -ngld all --spec-draft-n-max 2
echo
pass() { local n=$1 l=$2; shift 2; cool; echo "gpu before pass $n ($l): $(temp)"
  $BIN/llama-bench -m $M -ngl 99 -fa 1 -t 6 -p 512,2048 -n 128 -r 5 -o md "$@"; echo "gpu after: $(temp)"; echo; }
pass 1 host
pass 2 card "${OT[@]}"
pass 3 card "${OT[@]}"
pass 4 host
cool; echo "depth sweep, gpu before: $(temp)"
$BIN/llama-bench -m $M -ngl 99 -fa 1 -t 6 "${OT[@]}" -p 512 -n 128 -d 0,8192,32768,65536,120000 -r 3 -o md
echo "gpu after: $(temp)"
