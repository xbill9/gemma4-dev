#!/bin/bash
# Reruns the one arm piece the 2026-10-15 01:4x memory-pressure stop killed (BFCL, E4B, drafter at
# 2 tokens), with run.sh's flags, then writes DONE so ../2026-10-14-drafter-gaps-1650ti/run.sh starts.
set -u
cd "$(dirname "$0")"
OUT=$PWD/accuracy; LOG=$OUT/driver.log; J=$HOME/gemma4-dev/jev-tpu-v5e1; M=$HOME/models
BIN=$HOME/llama.cpp-77c4540ce/build/bin/llama-server
state() { echo "$(date +%T) gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" >> "$LOG"; }
pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done
state "RESUME after the 01:4x stop: bfcl_simple e4b-q4x-n2 again from the start"
setsid $BIN -m $M/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-q4_0-exact.gguf --host 127.0.0.1 --port 8099 -ngl 99 \
  -c 32768 --parallel 8 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 --reasoning off --metrics \
  -md $M/gemma-4-E4B-it-qat-assistant/gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf --spec-type draft-mtp -ngld all \
  --spec-draft-n-max 2 > "$OUT/server-e4b-q4x-n2-bfcl.log" 2>&1 < /dev/null &
for i in $(seq 180); do curl -fsS http://127.0.0.1:8099/health >/dev/null 2>&1 && break; sleep 1; done
state "server e4b-q4x-n2 up: $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
curl -fsS http://127.0.0.1:8099/metrics > "$OUT/metrics-before-e4b-q4x-n2-bfcl.prom"
state "START bfcl_simple e4b-q4x-n2"
(cd $J && python3 gen_eval.py run http://127.0.0.1:8099 google/gemma-4-E4B-it bfcl_simple "$OUT/bfcl_simple-e4b-q4x-n2.jsonl" --concurrency 8) > "$OUT/bfcl_simple-e4b-q4x-n2.log" 2>&1
state "END bfcl_simple e4b-q4x-n2 rc=$? $(tail -1 $OUT/bfcl_simple-e4b-q4x-n2.log)"
curl -fsS http://127.0.0.1:8099/metrics > "$OUT/metrics-after-e4b-q4x-n2-bfcl.prom"
pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done
state "DONE"
