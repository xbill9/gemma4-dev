#!/bin/bash
# The 1650 Ti on the RTX 3050's llama.cpp build (77c4540ce), then answer quality with the drafter.
#  1. Speed: spec_ab.py with LLAMA_SERVER_BIN pointed at 77c4540ce, E2B and E4B, no drafter and the
#     exact Q4_0 drafter at 1 and 2 tokens, no drafter again last. Compares with
#     2026-10-14-drafter-1650ti (f95b0d9) and tests whether E2B's greedy text at 1 token matches
#     no drafter, as it does on the 3050.
#  2. Accuracy: GSM8K (1,319, 2,048-token limit) and BFCL simple (400) through
#     jev-tpu-v5e1/gen_eval.py, 8 at a time, on 77c4540ce with the 3050 accuracy run's server flags
#     (-c 32768 --parallel 8) and this rig's own (per_layer_token_embd stays on the host).
#     Arms: no drafter, then each rig's serving drafter (E2B 1 token, E4B 2 tokens).
# Runs on the 1650 Ti laptop; nothing else may hold the card.
set -u
cd "$(dirname "$0")"
HERE=$PWD
ROOT=$HOME/gemma4-dev
J=$ROOT/jev-tpu-v5e1
M=$HOME/models
export LLAMA_SERVER_BIN=$HOME/llama.cpp-77c4540ce/build/bin/llama-server
if pgrep -x llama-server >/dev/null; then echo "a llama-server is running; stop it first" >&2; exit 1; fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee host.txt
$LLAMA_SERVER_BIN --version 2>&1 | tail -2 | tee -a host.txt
sha256sum -c $ROOT/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-q4exact-3050oem/sha256.txt | tee sha256.log

python3 spec_ab.py $ROOT/local-llamacpp-1650ti-2b-q4_0 speed-e2b none,q4x-n1,q4x-n2,none-again > speed-e2b.log 2>&1
python3 summarize.py speed-e2b > speed-e2b/summary.txt
python3 spec_ab.py $ROOT/local-llamacpp-1650ti-4b-q4_0 speed-e4b none,q4x-n1,q4x-n2,none-again > speed-e4b.log 2>&1
python3 summarize.py speed-e4b > speed-e4b/summary.txt

OUT=$HERE/accuracy
mkdir -p $OUT
LOG=$OUT/driver.log
state() { echo "$(date +%T) gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$LOG"; }
stop() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
arm() {  # name gguf hf-id [draft n]
  stop
  extra=()
  [ -n "${4:-}" ] && extra=(-md "$4" --spec-type draft-mtp -ngld all --spec-draft-n-max "$5")
  setsid $LLAMA_SERVER_BIN -m "$2" --host 127.0.0.1 --port 8099 -ngl 99 -c 32768 --parallel 8 -ctk f16 -ctv f16 \
    -fa 1 -t 6 -tb 12 --reasoning off --metrics "${extra[@]}" > "$OUT/server-$1.log" 2>&1 < /dev/null &
  for i in $(seq 180); do curl -fsS http://127.0.0.1:8099/health >/dev/null 2>&1 && break; sleep 1; done
  if ! curl -fsS http://127.0.0.1:8099/health >/dev/null 2>&1; then state "server $1 FAILED to start"; return 1; fi
  state "server $1 up: $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
  curl -fsS http://127.0.0.1:8099/metrics > "$OUT/metrics-before-$1.prom"
  for t in gsm8k bfcl_simple; do
    state "START $t $1"
    (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8099 "$3" $t "$OUT/$t-$1.jsonl" --concurrency 8) > "$OUT/$t-$1.log" 2>&1
    state "END $t $1 rc=$? $(tail -1 $OUT/$t-$1.log)"
  done
  curl -fsS http://127.0.0.1:8099/metrics > "$OUT/metrics-after-$1.prom"
}
E2B=$M/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
E4B=$M/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-q4_0-exact.gguf
D2=$M/gemma-4-E2B-it-qat-assistant/gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf
D4=$M/gemma-4-E4B-it-qat-assistant/gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf
state "BEGIN accuracy"
arm e2b-none $E2B google/gemma-4-E2B-it
arm e2b-q4x-n1 $E2B google/gemma-4-E2B-it $D2 1
arm e4b-none $E4B google/gemma-4-E4B-it
arm e4b-q4x-n2 $E4B google/gemma-4-E4B-it $D4 2
stop
state "DONE"
