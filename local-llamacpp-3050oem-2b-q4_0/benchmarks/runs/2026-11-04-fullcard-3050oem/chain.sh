#!/bin/bash
# After the SGLang drafter runs (chain6): the llama.cpp full-card probe.
until grep -q "accuracy-drafter exit" /home/xbill/gemma4-dev/local-sglang-3050oem-2b-q4w4a16emb4/benchmarks/runs/chain6.out 2>/dev/null; do sleep 20; done
# and no Android build on the host: it takes the CPU and 46 GB of RAM (2026-11-04 16:00)
while pgrep -x soong_build >/dev/null || pgrep -x siso >/dev/null; do sleep 30; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100 ]; do sleep 2; done
cd /home/xbill/gemma4-dev/local-llamacpp-3050oem-2b-q4_0
bash benchmarks/runs/2026-11-04-fullcard-3050oem/probe.sh > benchmarks/runs/2026-11-04-fullcard-3050oem/probe.log 2>&1
echo "probe exit $?" >> benchmarks/runs/2026-11-04-fullcard-3050oem/probe.log
