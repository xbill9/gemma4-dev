#!/bin/bash
# After the mistral.rs accuracy rerun: SGLang arms with the full text-config patch (gemma4_text shim).
R=/home/xbill/gemma4-dev/local-sglang-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-04-engines-3050oem
until grep -q "mistralrs accuracy exit" $R/run.out; do sleep 15; done
while pgrep -x mistralrs >/dev/null; do sleep 2; done
cd /home/xbill/gemma4-dev/local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-engines-3050oem
python3 engines.py $R sglang,sglang-d1,sglang-d2,sglang-d3,sglang-d4 >> $R/run.out 2>&1
echo "sglang engines rerun exit $?" >> $R/run.out
