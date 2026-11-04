#!/bin/bash
# After the first engines chain (its vLLM arms): SGLang arms with the text-config patch, then the
# mistral.rs GSM8K/BFCL rerun with thinking off.
R=/home/xbill/gemma4-dev/local-sglang-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-04-engines-3050oem
until grep -q "engines exit" $R/run.out; do sleep 15; done
cd /home/xbill/gemma4-dev/local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-engines-3050oem
python3 engines.py $R sglang,sglang-d1,sglang-d2,sglang-d3,sglang-d4 >> $R/run.out 2>&1
echo "sglang engines exit $?" >> $R/run.out
while pgrep -x mistralrs >/dev/null; do sleep 2; done
bash /home/xbill/gemma4-dev/local-mistralrs-3050oem-2b-q4_0/benchmarks/runs/2026-11-04-accuracy-3050oem/run.sh
echo "mistralrs accuracy exit $?" >> $R/run.out
