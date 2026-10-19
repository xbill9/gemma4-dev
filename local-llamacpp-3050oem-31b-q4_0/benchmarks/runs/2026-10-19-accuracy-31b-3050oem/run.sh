#!/bin/bash
# Answer quality of the 31B (Google's QAT Q4_0 GGUF) as the rig serves it (-ngl 20, q4_0 cache,
# -c 32768, -t 8 -tb 12, --parallel 1, one request at a time, prompt cache at the server default),
# with the exact Q4_0 31B drafter at 1 token and without it. GSM8K (1,319, 2048 tokens) and BFCL
# simple (400), scored by jev-tpu-v5e1/gen_eval.py through resumable_eval.py. The 26B sister's
# method (local-llamacpp-3050oem-26b-q4_0/benchmarks/runs/2026-10-18-accuracy-26b-3050oem/run.sh).
# Order: BFCL both arms, then GSM8K with the drafter, then GSM8K without. Rerunning resumes:
# finished tasks are skipped, a cut-short task continues from its last record. Restores the 12B
# rig's live server (make serve) at the end.
set -u
cd "$(dirname "$0")"; HERE=$PWD
ROOT=/home/xbill/gemma4-dev
LIVE=$ROOT/local-llamacpp-3050oem-12b-q4_0
BIN=/home/xbill/llama.cpp/build/bin/llama-server
X=/home/xbill/models/gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf
DR=/home/xbill/models/gemma-4-31B-it-qat-assistant/gemma-4-31B-it-qat-assistant-q4_0-exact.gguf
LOG=$HERE/driver.log
state() { echo "$(date '+%F %T') $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$LOG"; }
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 45 ]; do sleep 5; done; }
CUR=""
serve() {  # ARM
  local arm=$1 extra=()
  [ "$CUR" = "$arm" ] && return 0
  stop8080; cool; CUR=""
  [ "$arm" = q4x ] && extra=(-md "$DR" --spec-type draft-mtp -ngld all --spec-draft-n-max 1)
  local log=$HERE/server-$arm.log
  setsid $BIN -m $X --host 127.0.0.1 --port 8080 -ngl 20 -c 32768 --parallel 1 \
    -ctk q4_0 -ctv q4_0 -fa 1 -t 8 -tb 12 --reasoning off --metrics "${extra[@]}" >> "$log" 2>&1 < /dev/null &
  local pid=$!
  for i in $(seq 600); do
    curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && { CUR=$arm; state "server up $arm: $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | tr '\n' ' ')"; return 0; }
    kill -0 $pid 2>/dev/null || { state "server FAILED $arm"; return 1; }
    sleep 1
  done
  state "server TIMEOUT $arm"; return 1
}
state "BEGIN"
for job in "bfcl_simple q4x" "bfcl_simple none" "gsm8k q4x" "gsm8k none"; do
  set -- $job; t=$1 arm=$2
  if tail -1 $HERE/$t-$arm.log 2>/dev/null | grep -q '"wall_s"'; then state "HAVE $t $arm"; continue; fi
  serve $arm || { state "SKIP $t $arm"; continue; }
  state "START $t $arm"
  JEV_GEN_MAX_TOKENS=2048 python3 $HERE/resumable_eval.py http://127.0.0.1:8080 google/gemma-4-31B-it $t $HERE/$t-$arm.jsonl >> $HERE/$t-$arm.log 2>&1
  state "END $t $arm rc=$? $(tail -1 $HERE/$t-$arm.log)"
  curl -fsS http://127.0.0.1:8080/metrics > $HERE/metrics-after-$t-$arm.prom
done
stop8080
(cd $LIVE && setsid nohup make serve > run/llama-server.log 2>&1 < /dev/null &)
for i in $(seq 180); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
state "12B live server restored: $(curl -fsS http://127.0.0.1:8080/health)"
state "DONE"
