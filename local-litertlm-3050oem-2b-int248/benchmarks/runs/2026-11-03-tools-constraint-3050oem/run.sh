#!/bin/bash
# BFCL simple (400), greedy, one request at a time, Google's gemma-4-E2B-it.litertlm, drafter off,
# with litert-lm serve's constrained decoding off. Baseline with it on (the stock server):
# local-litertlm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-02-accuracy-3050oem/bfcl_simple-int248.jsonl
# Usage: run.sh [int248-off|q4exact-off]. q4exact = local-litertlm-3050oem-2b-q4w4a16emb4's exact QAT build.
# Ports 9401/9402 so it can run beside that rig's accuracy run on 9398/9399.
set -u
cd "$(dirname "$0")"
HERE=$PWD
J=$HOME/gemma4-dev/jev-tpu-v5e1
name=${1:-int248-off}
case $name in
  int248-off) id=gemma-4-e2b dir=$HOME/.cache/local-litertlm-3050oem-2b-int248 ;;
  q4exact-off) id=gemma-4-e2b-q4exact dir=$HOME/.cache/local-litertlm-3050oem-2b-q4w4a16emb4 ;;
esac
printf '{"models": {"%s": {"backend": "gpu", "cache": "disk", "speculative_decoding": "off"}}}\n' "$id" > config-$name.json
LITERT_LM_DIR=$dir python3 serve_noconstraint.py serve --config config-$name.json --port 9402 > server-$name.log 2>&1 &
S=$!
python3 proxy.py 9401 http://127.0.0.1:9402 "$id" &
P=$!
until curl -sf localhost:9401/v1/models >/dev/null; do kill -0 $S 2>/dev/null || { echo "$(date +%F' '%T) server died" >> driver.log; exit 1; }; sleep 2; done
t=bfcl_simple
echo "$(date +%F' '%T) START $name $t gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader)" >> driver.log
(cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:9401 google/gemma-4-E2B-it $t "$HERE/$t-$name.jsonl" --concurrency 1) > $t-$name.log 2>&1
echo "$(date +%F' '%T) END $name $t rc=$? $(tail -1 $t-$name.log)" >> driver.log
kill $P $S; wait $S 2>/dev/null
