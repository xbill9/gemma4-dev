#!/bin/bash
# Job B. Answer quality as the rigs serve: -c 131072 --parallel 1, one request at a time
# (gen_eval.py --concurrency 1), prompt cache at the server default (on). GSM8K (1,319, 2048
# tokens) and BFCL simple (400). Order: E4B with the exact Q4_0 drafter at 2 tokens (the live
# setting), E4B without, E2B with, E2B without. Each arm is committed when it finishes.
set -u
cd "$(dirname "$0")"; HERE=$PWD
. ../2026-10-14-overnight-3050oem/lib.sh
J=$ROOT/jev-tpu-v5e1
REL=local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-fullctx-3050oem
for spec in "e4b-q4x $E4B $D4 google/gemma-4-E4B-it" "e4b-none $E4B - google/gemma-4-E4B-it" \
            "e2b-q4x $E2B $D2 google/gemma-4-E2B-it" "e2b-none $E2B - google/gemma-4-E2B-it"; do
  set -- $spec; arm=$1 gguf=$2 d=$3 name=$4; [ "$d" = - ] && d=""
  serve $HERE/server-$arm.log 131072 1 $gguf "$d" 2 || { state "SKIP B $arm"; continue; }
  for t in bfcl_simple gsm8k; do
    state "START B $t $arm"
    (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8080 $name $t $HERE/$t-$arm.jsonl --concurrency 1) > $HERE/$t-$arm.log 2>&1
    state "END B $t $arm rc=$? $(tail -1 $HERE/$t-$arm.log)"
  done
  curl -fsS http://127.0.0.1:8080/metrics > $HERE/metrics-after-$arm.prom
  python3 compare.py > COMPARE.txt 2>&1
  commit_push $REL "3050oem overnight B: full-context one-at-a-time accuracy, arm $arm"
done
stop8080
