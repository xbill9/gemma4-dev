#!/bin/bash
# After the SGLang engines arms (int4 tables patch): concurrency sweep, then GSM8K/BFCL.
E=/home/xbill/gemma4-dev/local-sglang-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-04-engines-3050oem
until grep -q "sglang engines int4-emb exit" $E/run.out; do sleep 15; done
cd /home/xbill/gemma4-dev/local-sglang-3050oem-2b-q4w4a16emb4
bash benchmarks/runs/2026-11-04-concurrency-3050oem/run.sh > benchmarks/runs/2026-11-04-concurrency-3050oem/run.log 2>&1
echo "sglang concurrency exit $?" >> $E/run.out
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 100 ]; do sleep 2; done
bash benchmarks/runs/2026-11-04-accuracy-3050oem/run.sh
echo "sglang accuracy exit $?" >> $E/run.out
