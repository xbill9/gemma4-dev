#!/bin/bash
# After chain4 (concurrency, accuracy): the drafter arms again, with the int4 lm_head fix for the drafter.
E=/home/xbill/gemma4-dev/local-sglang-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-04-engines-3050oem
until grep -q "sglang accuracy exit" $E/run.out; do sleep 15; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100 ]; do sleep 2; done
cd /home/xbill/gemma4-dev/local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-engines-3050oem
python3 engines.py $E sglang-d1,sglang-d2,sglang-d3,sglang-d4 >> $E/run.out 2>&1
echo "sglang drafters exit $?" >> $E/run.out
