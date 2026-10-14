#!/bin/bash
# Same-day control for run_accuracy.sh: E2B and E4B with NO drafter, server flags otherwise
# identical (-c 32768 --parallel 8 ...), GSM8K and BFCL through gen_eval.py as there.
# Pairs with the drafted arms run the same day, and checks that the 2026-10-13 no-drafter
# baseline reproduces. Waits for run_accuracy.sh to finish, then restores the E4B live server.
set -u
ROOT=/home/xbill/gemma4-dev
J=$ROOT/jev-tpu-v5e1
OUT=$ROOT/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-q4exact-3050oem/accuracy
LOG=$OUT/driver.log
BIN=/home/xbill/llama.cpp/build/bin/llama-server
M=/home/xbill/models
until grep -q DONE "$LOG"; do sleep 30; done
state() { echo "$(date +%T) gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$LOG"; }
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
arm() {  # name gguf hf-id
  stop8080
  setsid $BIN -m "$2" --host 127.0.0.1 --port 8080 -ngl 99 -c 32768 --parallel 8 -ctk f16 -ctv f16 -fa 1 \
    -t 6 -tb 12 --reasoning off --metrics -ot 'per_layer_token_embd\.weight=CUDA0' > "$OUT/server-$1.log" 2>&1 < /dev/null &
  for i in $(seq 120); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  state "server $1 up: $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
  curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-before-$1.prom"
  for t in gsm8k bfcl_simple; do
    state "START $t $1"
    (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8080 "$3" $t "$OUT/$t-$1.jsonl" --concurrency 8) > "$OUT/$t-$1.log" 2>&1
    state "END $t $1 rc=$? $(tail -1 $OUT/$t-$1.log)"
  done
  curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-after-$1.prom"
}
state "BEGIN control"
arm e2b-none $M/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf google/gemma-4-E2B-it
arm e4b-none $M/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-q4_0-exact.gguf google/gemma-4-E4B-it
stop8080
(cd $ROOT/local-llamacpp-3050oem-4b-q4_0 && setsid nohup make serve > run/llama-server.log 2>&1 < /dev/null &)
for i in $(seq 120); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
state "E4B live server restored: $(curl -fsS http://127.0.0.1:8080/health)"
state "CONTROL DONE"
