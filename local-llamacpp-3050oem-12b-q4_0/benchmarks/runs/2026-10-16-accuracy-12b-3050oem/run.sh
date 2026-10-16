#!/bin/bash
# Answer quality of the 12B exact Q4_0 as the rig serves (-c 24576 --parallel 1, one request at a
# time, prompt cache at the server default), with the exact Q4_0 12B drafter at 2 tokens and
# without it. GSM8K (1,319, 2048 tokens) and BFCL simple (400) via jev-tpu-v5e1/gen_eval.py.
# The method of local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-fullctx-3050oem,
# at this rig's context. Waits for the CPU KL-divergence run to finish so timings are clean, and
# restores the rig's live server (make serve) at the end.
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
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 45 ]; do sleep 5; done; }
serve() {  # LOG [DRAFTER]
  local log=$1 d=${2:-}
  stop8080; cool
  local extra=()
  [ -n "$d" ] && extra=(-md "$d" --spec-type draft-mtp -ngld all --spec-draft-n-max 2)
  setsid $BIN -m $X --host 127.0.0.1 --port 8080 -ngl 99 -c 24576 --parallel 1 -ctk f16 -ctv f16 \
    -fa 1 -t 6 -tb 12 --reasoning off --metrics "${extra[@]}" > "$log" 2>&1 < /dev/null &
  local pid=$!
  for i in $(seq 300); do
    curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && { state "server up $(basename $log): $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | tr '\n' ' ')"; return 0; }
    kill -0 $pid 2>/dev/null || { state "server FAILED $(basename $log)"; return 1; }
    sleep 1
  done
  state "server TIMEOUT $(basename $log)"; return 1
}
while pgrep -x llama-perplexity >/dev/null || pgrep -f run_kld.sh >/dev/null; do sleep 30; done
state "BEGIN"
for spec in "q4x $DR" "none -"; do
  set -- $spec; arm=$1 d=$2; [ "$d" = - ] && d=""
  serve $HERE/server-$arm.log "$d" || { state "SKIP $arm"; continue; }
  for t in bfcl_simple gsm8k; do
    state "START $t $arm"
    (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8080 google/gemma-4-12B-it $t $HERE/$t-$arm.jsonl --concurrency 1) > $HERE/$t-$arm.log 2>&1
    state "END $t $arm rc=$? $(tail -1 $HERE/$t-$arm.log)"
  done
  curl -fsS http://127.0.0.1:8080/metrics > $HERE/metrics-after-$arm.prom
done
stop8080
(cd $RIG && setsid nohup make serve > run/llama-server.log 2>&1 < /dev/null &)
for i in $(seq 180); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
state "12B live server restored: $(curl -fsS http://127.0.0.1:8080/health)"
state "DONE"
