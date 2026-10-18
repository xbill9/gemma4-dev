#!/bin/bash
# 12B on the RTX 3050 OEM: which KV cache type and drafter setting gives the most context and
# speed. 1) kv_sweep.py: fit and generation speed with 0 to 128k tokens in context, f16 / q8_0 /
# q4_0 cache, with and without the exact Q4_0 drafter. 2) GSM8K + BFCL simple one request at a
# time with the q8_0 cache at its largest -c, with the drafter (49,152) and without (65,536):
# the method of ../2026-10-16-accuracy-12b-3050oem, whose f16 scores are the reference.
# Stops the rig's live server first and restores it (make serve) at the end.
set -u
cd "$(dirname "$0")"; HERE=$PWD
ROOT=/home/xbill/gemma4-dev
RIG=$ROOT/local-llamacpp-3050oem-12b-q4_0
J=$ROOT/jev-tpu-v5e1
BIN=/home/xbill/llama.cpp/build/bin/llama-server
X=/home/xbill/models/gemma-4-12B-it-qat-q4_0-exact/gemma-4-12B-it-q4_0-exact.gguf
DR=/home/xbill/models/gemma-4-12B-it-qat-assistant/gemma-4-12B-it-qat-assistant-q4_0-exact.gguf
LOG=$HERE/driver.log
state() { echo "$(date '+%F %T') $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$LOG"; }
stopall() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 45 ]; do sleep 5; done; }
serve() {  # LOG CTX [DRAFTER]
  local log=$1 ctx=$2 d=${3:-}
  stopall; cool
  local extra=()
  [ -n "$d" ] && extra=(-md "$d" --spec-type draft-mtp -ngld all --spec-draft-n-max 2)
  setsid $BIN -m $X --host 127.0.0.1 --port 8080 -ngl 99 -c $ctx --parallel 1 -ctk q8_0 -ctv q8_0 \
    -fa 1 -t 6 -tb 12 --reasoning off --metrics "${extra[@]}" > "$log" 2>&1 < /dev/null &
  local pid=$!
  for i in $(seq 300); do
    curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && { state "server up $(basename $log): $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | tr '\n' ' ')"; return 0; }
    kill -0 $pid 2>/dev/null || { state "server FAILED $(basename $log)"; return 1; }
    sleep 1
  done
  state "server TIMEOUT $(basename $log)"; return 1
}
state "BEGIN"
stopall
state "START kv_sweep"
python3 kv_sweep.py $RIG $HERE > $HERE/kv_sweep.out 2>&1
state "END kv_sweep rc=$?"
for spec in "q8x 49152 $DR" "q8none 65536 -"; do
  set -- $spec; arm=$1 ctx=$2 d=$3; [ "$d" = - ] && d=""
  serve $HERE/server-$arm.log $ctx "$d" || { state "SKIP $arm"; continue; }
  for t in bfcl_simple gsm8k; do
    state "START $t $arm"
    (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8080 google/gemma-4-12B-it $t $HERE/$t-$arm.jsonl --concurrency 1) > $HERE/$t-$arm.log 2>&1
    state "END $t $arm rc=$? $(tail -1 $HERE/$t-$arm.log)"
  done
  curl -fsS http://127.0.0.1:8080/metrics > $HERE/metrics-after-$arm.prom
done
stopall
(cd $RIG && setsid nohup make serve > run/llama-server.log 2>&1 < /dev/null &)
for i in $(seq 180); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
state "12B live server restored: $(curl -fsS http://127.0.0.1:8080/health)"
state "DONE"
