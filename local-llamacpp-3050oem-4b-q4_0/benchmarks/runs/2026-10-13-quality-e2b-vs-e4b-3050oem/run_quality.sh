#!/bin/bash
# Answer quality, E2B vs E4B, exact Q4_0 GGUFs on the RTX 3050 OEM, llama.cpp 77c4540ce.
# Per model, one llama-server with identical flags and NO drafter (so the scores are the
# models', not the speculator's): -c 32768 --parallel 8, everything else as the rigs serve.
#   1. Nimble 13-subset suite (3,880 records) via jev-tpu-v5e1/nimble_suite/run_suite.py
#      through vllm_shim.py on :8090, JEV_TOPK=32, concurrency 8.
#   2. GSM8K (1,319) via jev-tpu-v5e1/gen_eval.py, JEV_GEN_MAX_TOKENS=2048, concurrency 8.
#   3. BFCL simple (400) via gen_eval.py, concurrency 8.
# The E4B rig's live server is restored at the end with `make serve`.
set -u
ROOT=/home/xbill/gemma4-dev
J=$ROOT/jev-tpu-v5e1
OUT=$ROOT/local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-10-13-quality-e2b-vs-e4b-3050oem
LOG=$OUT/driver.log
DATA=/home/xbill/data/nimble-suite/public
BIN=/home/xbill/llama.cpp/build/bin/llama-server
state() { echo "$(date +%T) gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$LOG"; }
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
serve() {  # name gguf
  stop8080
  setsid $BIN -m "$2" --host 127.0.0.1 --port 8080 -ngl 99 -c 32768 --parallel 8 -ctk f16 -ctv f16 -fa 1 \
    -t 6 -tb 12 --reasoning off --metrics -ot 'per_layer_token_embd\.weight=CUDA0' > "$OUT/server-$1.log" 2>&1 < /dev/null &
  for i in $(seq 120); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  state "server $1 up: $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
}
model() {  # name gguf hf-id
  serve "$1" "$2"
  curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-before-$1.prom"
  python3 $OUT/vllm_shim.py --listen 8090 --upstream http://127.0.0.1:8080 > "$OUT/shim-$1.log" 2>&1 & SH=$!
  sleep 1
  state "START suite $1"
  (cd $J && JEV_TOPK=32 python3 nimble_suite/run_suite.py --arm autoregressive --upstream http://127.0.0.1:8090 \
     --model "$3" --records $DATA --run 2026-10-13-suite-3050oem-$1-q4exact --concurrency 8) > "$OUT/suite-$1.log" 2>&1
  state "END suite $1 rc=$?"
  kill $SH
  state "START gsm8k $1"
  (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8080 "$3" gsm8k "$OUT/gsm8k-$1.jsonl" --concurrency 8) > "$OUT/gsm8k-$1.log" 2>&1
  state "END gsm8k $1 rc=$? $(tail -1 $OUT/gsm8k-$1.log)"
  state "START bfcl $1"
  (cd $J && python3 gen_eval.py run http://127.0.0.1:8080 "$3" bfcl_simple "$OUT/bfcl-$1.jsonl" --concurrency 8) > "$OUT/bfcl-$1.log" 2>&1
  state "END bfcl $1 rc=$? $(tail -1 $OUT/bfcl-$1.log)"
  curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-after-$1.prom"
}
state "BEGIN"
model e2b /home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf google/gemma-4-E2B-it
model e4b /home/xbill/models/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-q4_0-exact.gguf google/gemma-4-E4B-it
stop8080
(cd $ROOT/local-llamacpp-3050oem-4b-q4_0 && setsid nohup make serve > run/llama-server.log 2>&1 < /dev/null &)
for i in $(seq 120); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
state "E4B live server restored: $(curl -fsS http://127.0.0.1:8080/health)"
state "DONE"
