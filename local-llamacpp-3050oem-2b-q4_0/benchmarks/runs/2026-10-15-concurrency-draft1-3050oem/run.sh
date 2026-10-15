#!/bin/bash
# One drafted token under concurrency: 2026-10-15-concurrency-mixed-3050oem's sweep (six mixed
# prompts, -c 32768 --parallel 64 -kvu, c=1..64, 5 repeats, greedy and sampled, fresh server per
# mode) with --spec-draft-n-max 1. Compared against that run's no-drafter and 2-token arms.
set -u
cd "$(dirname "$0")"; HERE=$PWD
. ../2026-10-14-overnight-3050oem/lib.sh
DRIVER=$ROOT/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-queue-3050oem/driver.log
BIN=$ROOT/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-queue-3050oem/kvu-server.sh
MIXED=$ROOT/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-concurrency-mixed-3050oem/mixed.py
for spec in "e2b $E2B $D2 xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf" "e4b $E4B $D4 xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf"; do
  set -- $spec; m=$1 gguf=$2 drafter=$3 name=$4
  out=$HERE/$m/q4x-n1; mkdir -p $out; rm -f $out/results.jsonl
  for mode in greedy sampled; do
    serve $out/llama-server-$mode.log 32768 64 $gguf "$drafter" 1 || { state "SKIP J3 $m $mode"; continue; }
    state "START J3 $m $mode"
    python3 $MIXED --model $name --out $out/results.jsonl --mode $mode > $out/$mode.log 2>&1
    state "END J3 $m $mode rc=$?"
  done
done
