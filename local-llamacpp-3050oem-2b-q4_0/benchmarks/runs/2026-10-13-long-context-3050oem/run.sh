#!/bin/bash
# Single-user context length on the RTX 3050 OEM. Step 1: allocation at -c 8192 and
# -c 131072 (the trained maximum) with the serving flags. Step 2: llama-bench at
# increasing depth (tokens already in the KV cache), r=3, GPU cooled to <=40 C first.
set -eu
cd "$(dirname "$0")"
BIN=/home/xbill/llama.cpp/build/bin
M=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
OT=(-ot 'per_layer_token_embd\.weight=CUDA0')
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 40 ]; do sleep 5; done; }
for c in 8192 32768 131072; do
  $BIN/llama-server -lv 4 -m $M --host 127.0.0.1 --port 8099 -ngl 99 -c $c -ctk f16 -ctv f16 -fa 1 \
    -t 6 -tb 12 --parallel 1 "${OT[@]}" > alloc-c$c.log 2>&1 &
  pid=$!
  until curl -fs localhost:8099/health >/dev/null; do sleep 1; kill -0 $pid 2>/dev/null || { echo "c=$c: server died"; exit 1; }; done
  echo "== -c $c: nvidia-smi $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
  grep -E 'CUDA0 (KV|compute|model) buffer' alloc-c$c.log | sed 's/^[0-9.]* //'
  kill $pid; wait $pid 2>/dev/null || true
done
echo
cool
$BIN/llama-bench -m $M -ngl 99 -fa 1 -t 6 "${OT[@]}" -p 512 -n 128 -d 0,8192,32768,65536,120000 -r 3 -o md
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader
