#!/bin/bash
# The whole 1650 Ti drafter suite: fetch and verify the drafters, then E2B and E4B, each with no
# drafter and the exact Q4_0 drafter at 1/2/3 tokens, then the tables. Run on the 1650 Ti laptop
# from anywhere; nothing else may hold the card (both rigs serve on :8080 and E4B needs most of it).
set -eu
cd "$(dirname "$0")"
R=$HOME/gemma4-dev
M=$HOME/models
if pgrep -x llama-server >/dev/null; then echo "a llama-server is running; stop it first" >&2; exit 1; fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee host.txt
$(grep ^LLAMA_SERVER_BIN= $R/local-llamacpp-1650ti-2b-q4_0/tpu.env | cut -d= -f2) --version 2>&1 | tail -2 | tee -a host.txt
./fetch_drafters.sh | tee fetch.log
python3 spec_ab.py $R/local-llamacpp-1650ti-2b-q4_0 $M/gemma-4-E2B-it-qat-assistant/gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf e2b > e2b.log 2>&1
python3 summarize.py e2b | tee e2b/summary.txt
python3 spec_ab.py $R/local-llamacpp-1650ti-4b-q4_0 $M/gemma-4-E4B-it-qat-assistant/gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf e4b > e4b.log 2>&1
python3 summarize.py e4b | tee e4b/summary.txt
echo DONE
