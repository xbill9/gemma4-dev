#!/bin/bash
# The QAT drafters repacked to compressed-tensors W4A16 (repack/repack_drafter.py, output in
# /opt1/models), at 1-4 drafted tokens, with same-day controls: no drafter and the bf16 QAT drafter
# at the serving 3 tokens, each first and again last. Runs after run.sh; same rules for the card.
set -eu
cd "$(dirname "$0")"
R=$HOME/gemma4-dev
for m in E2B E4B; do
  d=/opt1/models/gemma-4-$m-it-qat-q4_0-w4a16-ct-assistant
  echo "$m $(sha256sum $d/model.safetensors | cut -c1-64) $(stat -c %s $d/model.safetensors) B"
done | tee q4ct-sha256.txt
ARMS=none,qat-n3,q4ct-n1,q4ct-n2,q4ct-n3,q4ct-n4,qat-n3-again,none-again
python3 spec_ab.py $R/gpu-vllm-t4-2b-w4a16 e2b-q4ct $ARMS > e2b-q4ct.log 2>&1
python3 summarize.py e2b-q4ct | tee e2b-q4ct/summary.txt
python3 spec_ab.py $R/gpu-vllm-t4-4b-q4w4a16emb4 e4b-q4ct $ARMS > e4b-q4ct.log 2>&1
python3 summarize.py e4b-q4ct | tee e4b-q4ct/summary.txt
echo DONE
