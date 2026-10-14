#!/bin/bash
# Answer quality with each drafter serving, E2B and E4B exact Q4_0 on the RTX 3050 OEM.
# Server flags are the 2026-10-13 no-drafter quality run's (-c 32768 --parallel 8, ...) plus
# -md DRAFTER --spec-type draft-mtp -ngld all --spec-draft-n-max 2, so each arm pairs
# record-for-record with that run's gsm8k-*.jsonl / bfcl-*.jsonl.
#   GSM8K (1,319) via jev-tpu-v5e1/gen_eval.py, JEV_GEN_MAX_TOKENS=2048, concurrency 8.
#   BFCL simple (400) via gen_eval.py, concurrency 8.
# The suite is not rerun: it reads next-token probabilities of one prompt, where no draft is made.
# The E4B rig's live server is restored at the end with `make serve`.
set -u
ROOT=/home/xbill/gemma4-dev
J=$ROOT/jev-tpu-v5e1
OUT=$ROOT/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-q4exact-3050oem/accuracy
mkdir -p $OUT
LOG=$OUT/driver.log
BIN=/home/xbill/llama.cpp/build/bin/llama-server
M=/home/xbill/models
state() { echo "$(date +%T) gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$LOG"; }
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
arm() {  # name gguf drafter hf-id
  stop8080
  setsid $BIN -m "$2" --host 127.0.0.1 --port 8080 -ngl 99 -c 32768 --parallel 8 -ctk f16 -ctv f16 -fa 1 \
    -t 6 -tb 12 --reasoning off --metrics -ot 'per_layer_token_embd\.weight=CUDA0' \
    -md "$3" --spec-type draft-mtp -ngld all --spec-draft-n-max 2 > "$OUT/server-$1.log" 2>&1 < /dev/null &
  for i in $(seq 120); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  state "server $1 up: $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
  curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-before-$1.prom"
  for t in gsm8k bfcl_simple; do
    state "START $t $1"
    (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8080 "$4" $t "$OUT/$t-$1.jsonl" --concurrency 8) > "$OUT/$t-$1.log" 2>&1
    state "END $t $1 rc=$? $(tail -1 $OUT/$t-$1.log)"
  done
  curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-after-$1.prom"
}
state "BEGIN"
E2B=$M/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
E4B=$M/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-q4_0-exact.gguf
arm e2b-q8 $E2B $M/gemma-4-E2B-it-qat-assistant/gemma-4-E2B-it-qat-assistant-q8_0.gguf google/gemma-4-E2B-it
arm e2b-q4x $E2B $M/gemma-4-E2B-it-qat-assistant/gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf google/gemma-4-E2B-it
arm e4b-q8 $E4B $M/gemma-4-E4B-it-qat-assistant/gemma-4-E4B-it-qat-assistant-q8_0.gguf google/gemma-4-E4B-it
arm e4b-q4x $E4B $M/gemma-4-E4B-it-qat-assistant/gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf google/gemma-4-E4B-it
stop8080
(cd $ROOT/local-llamacpp-3050oem-4b-q4_0 && setsid nohup make serve > run/llama-server.log 2>&1 < /dev/null &)
for i in $(seq 120); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
state "E4B live server restored: $(curl -fsS http://127.0.0.1:8080/health)"
state "DONE"
