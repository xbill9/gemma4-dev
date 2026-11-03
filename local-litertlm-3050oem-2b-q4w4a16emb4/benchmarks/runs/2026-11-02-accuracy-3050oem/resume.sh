#!/bin/bash
# Re-runs the one arm the first pass lost: q4exact GSM8K was killed at 01:11 on 2026-11-03 when the
# login session that started run.sh closed, before gen_eval.py wrote its output. Same server config,
# proxy, harness and limits as run.sh; launched under setsid nohup.
set -u
cd "$(dirname "$0")"
HERE=$PWD
J=$HOME/gemma4-dev/jev-tpu-v5e1
id=gemma-4-e2b-q4exact name=q4exact
LITERT_LM_DIR=$HOME/.cache/local-litertlm-3050oem-2b-q4w4a16emb4 litert-lm serve --config config-$name.json --port 9399 > server-$name-resume.log 2>&1 &
S=$!
python3 proxy.py 9398 http://127.0.0.1:9399 "$id" &
P=$!
until curl -sf localhost:9398/v1/models >/dev/null; do kill -0 $S 2>/dev/null || { echo "$(date +%T) $name server died" >> driver.log; exit 1; }; sleep 2; done
t=gsm8k
echo "$(date +%F' '%T) START $name $t (resume) gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader)" >> driver.log
(cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:9398 google/gemma-4-E2B-it $t "$HERE/$t-$name.jsonl" --concurrency 1) > $t-$name.log 2>&1
echo "$(date +%F' '%T) END $name $t rc=$? $(tail -1 $t-$name.log)" >> driver.log
kill $P $S; wait $S 2>/dev/null
echo DONE >> driver.log
