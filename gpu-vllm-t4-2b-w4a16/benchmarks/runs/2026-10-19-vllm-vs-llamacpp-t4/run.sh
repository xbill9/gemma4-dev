#!/bin/bash
# vLLM on the Tesla T4, the same day and card as the llama.cpp runs in
# ~/qwen-dev/gpu-llamacpp-t4-qwen352b-q4_0/benchmarks/runs/2026-10-19-*-t4: Gemma 4 E2B QAT
# (xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4) with and without Google's QAT assistant drafter.
#   1. spec_ab.py (the 2026-10-15 drafter sweep's, unchanged): six prompts, greedy x3 + one sampled pass,
#      arms none, QAT drafter at 1-4 tokens, none again.
#   2. quality.py (the 2026-10-15 overnight run's, suite run name dated): GSM8K + BFCL with no drafter and
#      at the serving setting (QAT, 3 tokens); the Nimble suite on the no-drafter arm.
# Run by the llama.cpp chain after its last llama-server has stopped. Nothing else may hold port 8000.
set -u
cd "$(dirname "$0")"
E2B=$HOME/gemma4-dev/gpu-vllm-t4-2b-w4a16
say() { echo "$(date +%F\ %T) $*" | tee -a run.log; }
if curl -fs -o /dev/null --max-time 3 http://127.0.0.1:8000/health; then say "port 8000 is serving; stop it first"; exit 1; fi
[ -n "$(awk 'NR>1' /proc/swaps)" ] || sudo swapon /opt1/swapfile
make -s -C $E2B verify-patch | grep -q 'Patched' || { say "Turing clamp not confirmed"; exit 1; }
{ nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  PYTHONUSERBASE=/opt1/pyuser /usr/bin/python3.13 -c "import vllm,torch,transformers;print('vllm',vllm.__version__,'torch',torch.__version__,'transformers',transformers.__version__)" 2>/dev/null | tail -1
  for d in $HOME/.cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-q4_0-unquantized-assistant \
           $HOME/.cache/huggingface/hub/models--xbill9--gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4; do
    echo "$(basename $d | sed 's/^models--//; s/--/\//') $(ls $d/snapshots)"; done; } > host.txt
say "START spec_ab"
python3 spec_ab.py $E2B e2b none,qat-n1,qat-n2,qat-n3,qat-n4,none-again > e2b.log 2>&1
say "END spec_ab rc=$?"
python3 summarize.py e2b > e2b/summary.txt 2>&1
say "START quality"
TASKS=gsm8k,bfcl python3 quality.py $E2B quality e2b none,qat-n3 > quality.out 2>&1
say "END quality rc=$?"
say "DONE"
