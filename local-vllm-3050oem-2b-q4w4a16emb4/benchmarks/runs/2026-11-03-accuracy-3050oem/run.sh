#!/bin/bash
# GSM8K (1,319, 2,048-token limit) and BFCL simple (400, 512-token limit) through
# jev-tpu-v5e1/gen_eval.py, greedy, one request at a time: the harness, sets and limits of the
# llama.cpp, PyTorch and LiteRT accuracy runs on this card. Server from tpu.env (no drafter),
# with vLLM's gemma4 tool-call and reasoning parsers. Run from anywhere.
set -u
cd "$(dirname "$0")"
HERE=$PWD
RIG=$(cd ../../.. && pwd)
J=$HOME/gemma4-dev/jev-tpu-v5e1
set -a; . "$RIG/tpu.env"; set +a
python3 "$RIG/serve.py" --print > command.txt
python3 "$RIG/serve.py" > server.log 2>&1 &
S=$!
until curl -sf "$ENDPOINT/health" >/dev/null; do kill -0 $S 2>/dev/null || { echo "$(date +%F' '%T) server died" >> driver.log; exit 1; }; sleep 2; done
for t in bfcl_simple gsm8k; do
  echo "$(date +%F' '%T) START $t gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader)" >> driver.log
  (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run "$ENDPOINT" "$MODEL_NAME" $t "$HERE/$t-vllm.jsonl" --concurrency 1) > $t-vllm.log 2>&1
  echo "$(date +%F' '%T) END $t rc=$? $(tail -1 $t-vllm.log)" >> driver.log
done
kill $S; wait $S 2>/dev/null
echo DONE >> driver.log
