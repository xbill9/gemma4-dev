#!/bin/bash
# Job C. Does the prompt cache change scores? The 2026-10-14 accuracy arms ran with llama.cpp's
# default cache_prompt (on). This reruns them with cache_prompt false (gen_eval.py JEV_CACHE_PROMPT=0),
# same flags as 2026-10-14-drafter-q4exact-3050oem/accuracy (-c 32768 --parallel 8, concurrency 8,
# GSM8K 2048 tokens, BFCL simple), E4B then E2B, no drafter and exact Q4_0 drafter at 2 tokens,
# so each arm pairs record-for-record with its cache-on twin there.
set -u
cd "$(dirname "$0")"; HERE=$PWD
. ../2026-10-14-overnight-3050oem/lib.sh
J=$ROOT/jev-tpu-v5e1
for spec in "e4b-none $E4B - google/gemma-4-E4B-it" "e4b-q4x $E4B $D4 google/gemma-4-E4B-it" \
            "e2b-none $E2B - google/gemma-4-E2B-it" "e2b-q4x $E2B $D2 google/gemma-4-E2B-it"; do
  set -- $spec; arm=$1 gguf=$2 d=$3 name=$4; [ "$d" = - ] && d=""
  serve $HERE/server-$arm.log 32768 8 $gguf "$d" 2 || { state "SKIP C $arm"; continue; }
  for t in gsm8k bfcl_simple; do
    state "START C $t $arm cache off"
    (cd $J && JEV_CACHE_PROMPT=0 JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8080 $name $t $HERE/$t-$arm-nocache.jsonl --concurrency 8) > $HERE/$t-$arm-nocache.log 2>&1
    state "END C $t $arm rc=$? $(tail -1 $HERE/$t-$arm-nocache.log)"
  done
  curl -fsS http://127.0.0.1:8080/metrics > $HERE/metrics-after-$arm.prom
done
stop8080
python3 compare.py > COMPARE.txt 2>&1
