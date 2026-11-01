#!/bin/bash
# Sequential: profile, deep depth, fp32 short, then full autotune.
cd "$(dirname "$0")"
/usr/bin/time -f 'profile %e s' python3 profile_decode.py profile.json > profile.log 2>&1
/usr/bin/time -f 'deep %e s' python3 depth_bench.py deep.jsonl 32768,131000 float16 > deep.log 2>&1
/usr/bin/time -f 'fp32 %e s' python3 depth_bench.py short-fp32.jsonl 0,4096,32768 float32 > short-fp32.log 2>&1
rm -f tune.jsonl
python3 tune_gemv.py tune.jsonl > tune.log 2>&1
