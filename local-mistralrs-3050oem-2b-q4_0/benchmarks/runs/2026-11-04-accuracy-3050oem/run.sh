#!/bin/bash
# GSM8K (1,319, 2,048-token limit) and BFCL simple (400, 512-token limit) through
# jev-tpu-v5e1/gen_eval.py, greedy, one request at a time: the harness, sets and limits of the
# llama.cpp, PyTorch, LiteRT and vLLM accuracy runs on this card. Server from tpu.env (no drafter:
# every drafter length is slower than none on this engine, ../2026-11-04-engines-3050oem). Thinking off
# per request (JEV_ENABLE_THINKING=0): mistral.rs thinks by default on Gemma 4, the other engines did not.
set -u
cd "$(dirname "$0")"
HERE=$PWD
RIG=$(cd ../../.. && pwd)
J=$HOME/gemma4-dev/jev-tpu-v5e1
set -a; . "$RIG/tpu.env"; set +a
export MTP_MODEL=
python3 "$RIG/serve.py" --print > command.txt
python3 "$RIG/serve.py" > server.log 2>&1 &
S=$!
until curl -fsS "$ENDPOINT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"ok"}],"max_tokens":2}' >/dev/null 2>&1; do
  kill -0 $S 2>/dev/null || { echo "$(date +%F' '%T) server died" >> driver.log; exit 1; }; sleep 2; done
for t in bfcl_simple gsm8k; do
  echo "$(date +%F' '%T) START $t gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader)" >> driver.log
  (cd $J && JEV_ENABLE_THINKING=0 JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run "$ENDPOINT" default $t "$HERE/$t-mistralrs.jsonl" --concurrency 1) > $t-mistralrs.log 2>&1
  echo "$(date +%F' '%T) END $t rc=$? $(tail -1 $t-mistralrs.log)" >> driver.log
done
kill $S; wait $S 2>/dev/null
echo DONE >> driver.log
