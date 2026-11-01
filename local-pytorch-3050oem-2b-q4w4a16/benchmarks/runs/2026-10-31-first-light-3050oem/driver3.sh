#!/bin/bash
# Tuned tilings: short depth twice; llama.cpp on the same card and night, same method as the 1650 Ti close-gap run.
cd "$(dirname "$0")"
python3 depth_bench.py short-tuned.jsonl 0,4096,32768 float16 > short-tuned.log 2>&1; echo "tuned rc=$?" >> driver.log
python3 depth_bench.py short-tuned2.jsonl 0,4096 float16 > short-tuned2.log 2>&1; echo "tuned2 rc=$?" >> driver.log
M=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
~/llama.cpp/build/bin/llama-bench -m $M -ngl 99 -fa 1 -t 6 -n 128 -p 0 -d 0,128 -r 5 -o json > llamacpp-bench.json 2> llamacpp-bench.log; echo "llama rc=$?" >> driver.log
echo DONE3 >> driver.log
