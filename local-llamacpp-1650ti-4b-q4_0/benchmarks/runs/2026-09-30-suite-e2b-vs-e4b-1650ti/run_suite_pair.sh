#!/bin/bash
# Nimble 13-subset suite (3,880 records) on E2B then E4B, both exact Q4_0 GGUFs on the GTX 1650 Ti,
# through jev-tpu-v5e1/nimble_suite/run_suite.py unchanged and vllm_shim.py (llama-server /completion
# with n_probs 32, JEV_TOPK=32 as on the TPU runs). E2B is scored on the E2B rig's running demo server;
# E4B on its own rig's `make serve` flags; the E2B server is restored at the end with its original argv.
set -u
ROOT=/home/xbill/gemma4-dev
OUT=$ROOT/local-llamacpp-1650ti-4b-q4_0/benchmarks/runs/2026-09-30-suite-e2b-vs-e4b-1650ti
LOG=$OUT/driver.log
DATA=/home/xbill/data/nimble-suite/public
state() { echo "$(date +%T) gpu=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)C $*" | tee -a "$LOG"; }
suite() {  # tokenizer-model run-name
  state "START suite $2 on $(curl -fsS http://127.0.0.1:8080/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')"
  (cd $ROOT/jev-tpu-v5e1 && JEV_TOPK=32 python3 nimble_suite/run_suite.py --arm autoregressive \
     --upstream http://127.0.0.1:8090 --model $1 --records $DATA --run $2 --concurrency 1) > $OUT/$2.log 2>&1
  state "END suite $2 rc=$?"
}
stop8080() { kill $(pgrep -x llama-server) 2>/dev/null; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
wait8080() { for i in $(seq 90); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

state "BEGIN"
suite google/gemma-4-E2B-it 2026-09-30-suite-1650ti-e2b-q4exact
stop8080
(cd $ROOT/local-llamacpp-1650ti-4b-q4_0 && setsid nohup make serve > $OUT/e4b-server.log 2>&1 < /dev/null &)
wait8080 || state "E4B server FAILED"
suite google/gemma-4-E4B-it 2026-09-30-suite-1650ti-e4b-q4exact
stop8080
(cd $ROOT/local-llamacpp-1650ti-2b-q4_0 && setsid nohup $(cat $ROOT/local-llamacpp-1650ti-4b-q4_0/benchmarks/runs/2026-09-30-exact-gguf-e4b-1650ti/serve/e2b-server-cmdline-before.txt) > run/llama-server.log 2>&1 < /dev/null &)
wait8080; state "E2B server restored: $(curl -fsS http://127.0.0.1:8080/health)"
state "DONE"
