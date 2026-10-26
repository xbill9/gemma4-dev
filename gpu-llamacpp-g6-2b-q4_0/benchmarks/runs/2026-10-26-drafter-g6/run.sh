#!/bin/bash
# Drafter on the L4: speed.py, longctx.py, then GSM8K and BFCL simple with and without the drafter
# (jev-tpu-v5e1/gen_eval.py; the 3050 quality runs' flags: -c 32768 --parallel 8, concurrency 8,
# GSM8K at 2,048 tokens). Run from /opt/bench-ll on the instance.
set -u
cd "$(dirname "$0")"
. ./envsh
state() { echo "$(date +%T) $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader) $*" >> driver.log; }
state BEGIN
pkill -x llama-server; sleep 2
sha256sum $MODEL_PATH $SPEC_DRAFT_MODEL > models.sha256
DOC=/usr/share/doc/NVIDIA_GLX-1.0/README.txt; sha256sum $DOC > doc.sha256
mkdir -p speed longctx accuracy
state "START speed"; $PY speed.py tpu.env speed > speed/run.log 2>&1; state "END speed rc=$?"
state "START longctx"; $PY longctx.py tpu.env longctx $DOC > longctx/run.log 2>&1; state "END longctx rc=$?"
for arm in none q4x-n2; do
  pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done
  extra=""; [ $arm = q4x-n2 ] && extra="-md $SPEC_DRAFT_MODEL --spec-type draft-mtp -ngld all --spec-draft-n-max 2"
  setsid $LLAMA_SERVER_BIN -m $MODEL_PATH --host 127.0.0.1 --port 8080 -ngl 99 -c 32768 --parallel 8 -ctk f16 -ctv f16 \
    -fa 1 -t $THREADS -tb $THREADS_BATCH --reasoning off --metrics -ot 'per_layer_token_embd\.weight=CUDA0' $extra \
    > accuracy/server-$arm.log 2>&1 < /dev/null &
  for i in $(seq 180); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  state "server $arm up"
  curl -fsS http://127.0.0.1:8080/metrics > accuracy/metrics-before-$arm.prom
  for t in gsm8k bfcl_simple; do
    state "START $t $arm"
    (cd jev && JEV_GEN_MAX_TOKENS=2048 $PY gen_eval.py run http://127.0.0.1:8080 google/gemma-4-E2B-it $t ../accuracy/$t-$arm.jsonl --concurrency 8) > accuracy/$t-$arm.log 2>&1
    state "END $t $arm rc=$? $(tail -1 accuracy/$t-$arm.log)"
  done
  curl -fsS http://127.0.0.1:8080/metrics > accuracy/metrics-after-$arm.prom
done
pkill -x llama-server
state DONE
touch DONE
