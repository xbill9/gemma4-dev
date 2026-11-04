#!/bin/bash
# SGLang drafters under load, then GSM8K/BFCL with the drafter at 4 tokens.
cd /home/xbill/gemma4-dev/local-sglang-3050oem-2b-q4w4a16emb4
R=benchmarks/runs
bash $R/2026-11-04-concurrency-drafters-3050oem/run.sh > $R/2026-11-04-concurrency-drafters-3050oem/run.log 2>&1
echo "concurrency-drafters exit $?" >> $R/chain6.out
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100 ]; do sleep 2; done
bash $R/2026-11-04-accuracy-drafter-3050oem/run.sh
echo "accuracy-drafter exit $?" >> $R/chain6.out
