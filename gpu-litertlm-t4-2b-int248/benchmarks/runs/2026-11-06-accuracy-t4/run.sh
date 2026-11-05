#!/bin/bash
# GSM8K (1,319, 2,048-token limit) and BFCL simple (400, 512) through jev-tpu-v5e1/gen_eval.py, greedy, one
# request at a time, LiteRT-LM 0.19.0 on the Tesla T4, Google's int248 file, GPU backend, drafter off.
# As the 3050 paper table did: GSM8K on the stock server, BFCL with its constrained decoding off
# (serve_noconstraint.py), both behind proxy.py (max_tokens -> max_completion_tokens). Both copied unchanged
# from local-litertlm-3050oem-2b-int248/benchmarks/runs/2026-11-03-tools-constraint-3050oem.
set -u
cd "$(dirname "$0")"
HERE=$PWD
J=$HOME/gemma4-dev/jev-tpu-v5e1
export PYTHONUSERBASE=/opt1/pyuser-engines LITERT_LM_DIR=/opt1/cache/gpu-litertlm-t4-2b-int248
PY=/usr/bin/python3.13
id=gemma-4-e2b
printf '{"models": {"%s": {"backend": "gpu", "cache": "disk", "speculative_decoding": "off"}}}\n' "$id" > config.json
arm() {  # name task server-command...
  local name=$1 t=$2; shift 2
  "$@" > server-$name.log 2>&1 &
  local S=$!
  $PY proxy.py 9398 http://127.0.0.1:9399 "$id" > proxy-$name.log 2>&1 &
  local P=$!
  until curl -sf localhost:9398/v1/models >/dev/null; do kill -0 $S 2>/dev/null || { echo "$(date +%F' '%T) $name server died" >> driver.log; return 1; }; sleep 2; done
  echo "$(date +%F' '%T) START $name $t gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader)" >> driver.log
  (cd $J && PYTHONUSERBASE= JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:9398 google/gemma-4-E2B-it $t "$HERE/$t-$name.jsonl" --concurrency 1) > $t-$name.log 2>&1
  echo "$(date +%F' '%T) END $name $t rc=$? $(tail -1 $t-$name.log)" >> driver.log
  kill $P $S; wait $S 2>/dev/null
}
arm int248-off bfcl_simple $PY serve_noconstraint.py serve --config config.json --host 127.0.0.1 --port 9399
arm int248 gsm8k /opt1/pyuser-engines/bin/litert-lm serve --config config.json --host 127.0.0.1 --port 9399
echo DONE >> driver.log
