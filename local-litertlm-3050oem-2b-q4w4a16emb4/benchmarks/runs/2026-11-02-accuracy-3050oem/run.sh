#!/bin/bash
# GSM8K (1,319, 2,048-token limit) and BFCL simple (400) through jev-tpu-v5e1/gen_eval.py, greedy, one
# request at a time: the same harness, sets and limits as the llama.cpp and PyTorch accuracy runs on
# this card. Two arms, drafter off in both so only the weights differ:
#   int248   Google's gemma-4-E2B-it.litertlm (local-litertlm-3050oem-2b-int248)
#   q4exact  this rig's exact QAT Q4_0 build
# Each arm gets its own litert-lm server on 9399 and proxy.py on 9398.
set -u
cd "$(dirname "$0")"
HERE=$PWD
J=$HOME/gemma4-dev/jev-tpu-v5e1
arm() {  # name litert_lm_dir model_id
  local name=$1 dir=$2 id=$3
  printf '{"models": {"%s": {"backend": "gpu", "cache": "disk", "speculative_decoding": "off"}}}\n' "$id" > config-$name.json
  LITERT_LM_DIR=$dir litert-lm serve --config config-$name.json --port 9399 > server-$name.log 2>&1 &
  local S=$!
  python3 proxy.py 9398 http://127.0.0.1:9399 "$id" &
  local P=$!
  until curl -sf localhost:9398/v1/models >/dev/null; do kill -0 $S 2>/dev/null || { echo "$(date +%T) $name server died" >> driver.log; return 1; }; sleep 2; done
  for t in bfcl_simple gsm8k; do
    echo "$(date +%T) START $name $t gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader)" >> driver.log
    (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:9398 google/gemma-4-E2B-it $t "$HERE/$t-$name.jsonl" --concurrency 1) > $t-$name.log 2>&1
    echo "$(date +%T) END $name $t rc=$? $(tail -1 $t-$name.log)" >> driver.log
  done
  kill $P $S; wait $S 2>/dev/null
}
arm int248 $HOME/.cache/local-litertlm-3050oem-2b-int248 gemma-4-e2b
arm q4exact $HOME/.cache/local-litertlm-3050oem-2b-q4w4a16emb4 gemma-4-e2b-q4exact
echo DONE >> driver.log
