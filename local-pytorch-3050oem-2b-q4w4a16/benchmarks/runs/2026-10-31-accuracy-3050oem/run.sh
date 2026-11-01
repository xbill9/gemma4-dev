#!/bin/bash
# GSM8K (1,319, 2,048-token limit) and BFCL simple (400) on the PyTorch int4 engine through jev-tpu-v5e1/gen_eval.py, one
# request at a time. Same harness, sets and limit as the 1650 Ti run. Starts its own server.
set -u
cd "$(dirname "$0")"
HERE=$PWD
F=../2026-10-31-first-light-3050oem
M=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
~/llama.cpp/build/bin/llama-bench -m $M -ngl 99 -fa 1 -t 6 -c 131072 -n 64 -p 0 -d 61,4081,32761,130981 -r 3 -o json > $F/llamacpp-depth.json 2> $F/llamacpp-depth.log
echo "$(date +%T) depth rc=$?" >> driver.log
python3 ../../../openai_server.py --port 8100 --max-ctx 8192 > server.log 2>&1 &
S=$!
until curl -sf localhost:8100/health >/dev/null; do kill -0 $S 2>/dev/null || { echo "$(date +%T) server died" >> driver.log; exit 1; }; sleep 5; done
J=$HOME/gemma4-dev/jev-tpu-v5e1
for t in bfcl_simple gsm8k; do
  echo "$(date +%T) START $t gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader)" >> driver.log
  (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8100 google/gemma-4-E2B-it $t "$HERE/$t-pytorch.jsonl" --concurrency 1) > $t-pytorch.log 2>&1
  echo "$(date +%T) END $t rc=$? $(tail -1 $t-pytorch.log)" >> driver.log
done
kill $S
echo DONE >> driver.log
