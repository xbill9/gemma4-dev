#!/bin/bash
# Final-engine speed, llama.cpp 7182ef7 at the same depths, then GSM8K (1,319) and BFCL simple (400) through openai_server.py,
# one request at a time. Same harness, sets and token limit as local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-accuracy-1650ti.
set -u
cd "$(dirname "$0")"; HERE=$PWD; RIG=$HERE/../../..; J=$HOME/gemma4-dev/jev-tpu-v5e1
FL=$HERE/../2026-10-31-first-light-1650ti
M=/home/xbill/models/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-q4_0-exact.gguf
log() { echo "$(date +%T) $* gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader)" | tee -a driver.log; }
log START first-light; python3 $FL/first_light.py 40960 > $FL/run-final.jsonl 2> $FL/run-final.err; log END first-light rc=$?
log START llama-depth
for c in 8192 16640 33024 40960; do
  d=$((c - 640)); [ $c = 8192 ] && d=4096
  ~/llama.cpp/build/bin/llama-bench -m $M -ngl 99 -fa 1 -t 6 -c $c -p 512 -n 64 -d $d -r 2 -o json > llamacpp-depth-$d.json 2> llamacpp-depth-$d.log; echo "c=$c d=$d rc=$?" | tee -a driver.log
done
log END llama-depth
(cd $RIG && exec python3 openai_server.py --max-ctx 8192) > server.log 2>&1 &
S=$!
until curl -sf 127.0.0.1:8100/health >/dev/null; do kill -0 $S 2>/dev/null || { log server died; exit 1; }; sleep 5; done
for t in bfcl_simple gsm8k; do
  log START $t
  (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8100 google/gemma-4-E4B-it $t "$HERE/$t-pytorch.jsonl" --concurrency 1) > $t-pytorch.log 2>&1
  log END $t rc=$? "$(tail -1 $t-pytorch.log | cut -c1-160)"
done
kill $S; log DONE
