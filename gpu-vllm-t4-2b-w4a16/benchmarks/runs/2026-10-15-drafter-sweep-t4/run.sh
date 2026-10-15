#!/bin/bash
# The T4 drafter sweep under vLLM: E2B (gpu-vllm-t4-2b-w4a16) then E4B (gpu-vllm-t4-4b-q4w4a16emb4),
# each with no drafter, Google's QAT assistant (the rigs' SPECULATIVE_MODEL) and its non-QAT
# assistant at 1-4 drafted tokens, then the serving setting (QAT, 3 tokens) and no drafter again.
# Nothing else may hold the card or port 8000; the swapfile must be on (vLLM is OOM-killed loading
# weights without it on this 7.8 GB host) and the Turing clamp confirmed.
set -eu
cd "$(dirname "$0")"
R=$HOME/gemma4-dev
E2B=$R/gpu-vllm-t4-2b-w4a16
E4B=$R/gpu-vllm-t4-4b-q4w4a16emb4
if curl -fs -o /dev/null --max-time 3 http://127.0.0.1:8000/health; then echo "port 8000 is serving; stop it first" >&2; exit 1; fi
[ -n "$(awk 'NR>1' /proc/swaps)" ] || sudo swapon /opt1/swapfile
cat /proc/swaps
make -s -C $E2B verify-patch | grep -q 'Patched' || { echo "Turing clamp not confirmed" >&2; exit 1; }
{ nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  PYTHONUSERBASE=/opt1/pyuser /usr/bin/python3.13 -c "import vllm,torch,transformers;print('vllm',vllm.__version__,'torch',torch.__version__,'transformers',transformers.__version__)" 2>/dev/null | tail -1
  for d in $HOME/.cache/huggingface/hub/models--google--gemma-4-E?B-it*assistant \
           $HOME/.cache/huggingface/hub/models--xbill9--gemma-4-E?B-it-qat-q4_0-w4a16-ct-text-emb4; do
    echo "$(basename $d | sed 's/^models--//; s/--/\//') $(ls $d/snapshots)"; done; } | tee host.txt
ARMS=none,qat-n1,qat-n2,qat-n3,qat-n4,nq-n1,nq-n2,nq-n3,nq-n4,qat-n3-again,none-again
python3 spec_ab.py $E2B e2b $ARMS > e2b.log 2>&1
python3 summarize.py e2b | tee e2b/summary.txt
python3 spec_ab.py $E4B e4b $ARMS > e4b.log 2>&1
python3 summarize.py e4b | tee e4b/summary.txt
echo DONE
