#!/bin/bash
# After the autotune: baseline profile, deep depth and fp32 prompt reads on the 1650 Ti tilings.
cd "$(dirname "$0")"
while pgrep -f tune_gemv.py >/dev/null; do sleep 10; done
python3 profile_decode.py profile.json > profile.log 2>&1; echo "profile rc=$?" >> driver.log
python3 depth_bench.py deep.jsonl 32768,131000 float16 > deep.log 2>&1; echo "deep rc=$?" >> driver.log
python3 depth_bench.py short-fp32.jsonl 0,4096,32768 float32 > short-fp32.log 2>&1; echo "fp32 rc=$?" >> driver.log
echo DONE >> driver.log
