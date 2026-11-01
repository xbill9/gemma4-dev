#!/bin/bash
cd "$(dirname "$0")"
MAX_CTX=8192 python3 depth_bench.py short8k.jsonl 0,4096,8000 float16 > short8k.log 2>&1; echo "short8k rc=$?" >> driver.log
python3 profile_decode.py profile.json > profile.log 2>&1; echo "profile rc=$?" >> driver.log
cd ../2026-10-31-accuracy-3050oem; H=$PWD
python3 ../../../openai_server.py --port 8100 --max-ctx 4096 > server.log 2>&1 &
S=$!
until curl -sf localhost:8100/health >/dev/null; do kill -0 $S 2>/dev/null || { echo "server died" >> driver.log; exit 1; }; sleep 5; done
(cd ~/gemma4-dev/jev-tpu-v5e1 && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8100 google/gemma-4-12B-it bfcl_simple "$H/bfcl_simple-pytorch.jsonl" --concurrency 1) > bfcl_simple-pytorch.log 2>&1
echo "bfcl rc=$? $(tail -1 bfcl_simple-pytorch.log)" >> driver.log
kill $S; echo DONE >> driver.log
