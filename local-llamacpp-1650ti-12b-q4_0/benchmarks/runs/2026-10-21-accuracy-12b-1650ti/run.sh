#!/bin/bash
# Answer quality of the exact Q4_0 12B as this rig serves it on the GTX 1650 Ti (layers split
# between card and CPU), one request at a time, prompt cache at the server default, scored by
# jev-tpu-v5e1/gen_eval.py through resumable_eval.py: BFCL simple (all 400) and the first
# GSM8K_N GSM8K questions (2048 tokens). Paired afterwards with the 3050 12B sister's records for
# the same questions at the same cache type (compare.py). From the 3050 31B sister's run.sh
# (local-llamacpp-3050oem-31b-q4_0/benchmarks/runs/2026-10-19-accuracy-31b-3050oem).
# Rerunning resumes. Settings come from the environment so the serving choice is recorded here:
#   NGL CTX KV DRAFT_N (0 = no drafter) ARM GSM8K_N
set -u
cd "$(dirname "$0")"; HERE=$PWD
BIN=/home/xbill/llama.cpp/build/bin/llama-server
X=/home/xbill/models/gemma-4-12B-it-qat-q4_0-exact/gemma-4-12B-it-q4_0-exact.gguf
DR=/home/xbill/models/gemma-4-12B-it-qat-assistant/gemma-4-12B-it-qat-assistant-q4_0-exact.gguf
: "${NGL:?}" "${CTX:?}" "${KV:?}" "${DRAFT_N:?}" "${ARM:?}" "${GSM8K_N:=200}"
LOG=$HERE/driver.log
state() { echo "$(date '+%F %T') load=$(cut -d' ' -f1 /proc/loadavg) $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$LOG"; }
extra=()
[ "$DRAFT_N" != 0 ] && extra=(-md "$DR" --spec-type draft-mtp -ngld all --spec-draft-n-max "$DRAFT_N")
state "BEGIN arm=$ARM ngl=$NGL ctx=$CTX kv=$KV draft_n=$DRAFT_N gsm8k_n=$GSM8K_N"
setsid $BIN -m $X --host 127.0.0.1 --port 8080 -ngl $NGL -c $CTX --parallel 1 \
  -ctk $KV -ctv $KV -fa 1 -t 6 -tb 12 --reasoning off --metrics "${extra[@]}" >> "$HERE/server-$ARM.log" 2>&1 < /dev/null &
PID=$!
for i in $(seq 600); do
  curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { state "server FAILED"; exit 1; }
  sleep 1
done
state "server up: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
for job in "bfcl_simple 400" "gsm8k $GSM8K_N"; do
  set -- $job; t=$1 n=$2
  if tail -1 $HERE/$t-$ARM.log 2>/dev/null | grep -q '"wall_s"'; then state "HAVE $t"; continue; fi
  state "START $t ($n)"
  JEV_GEN_MAX_TOKENS=2048 python3 $HERE/resumable_eval.py http://127.0.0.1:8080 google/gemma-4-12B-it $t $HERE/$t-$ARM.jsonl $n >> $HERE/$t-$ARM.log 2>&1
  state "END $t rc=$? $(tail -1 $HERE/$t-$ARM.log)"
  curl -fsS http://127.0.0.1:8080/metrics > $HERE/metrics-after-$t-$ARM.prom
done
kill $PID; wait $PID 2>/dev/null
state "DONE"
