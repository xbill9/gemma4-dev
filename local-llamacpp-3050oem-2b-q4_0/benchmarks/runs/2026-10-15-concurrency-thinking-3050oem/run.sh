#!/bin/bash
# Thinking-on arm of 2026-10-15-concurrency-mixed-3050oem: same six mixed prompts, same server
# (-c 32768 --parallel 64 -kvu, rigs' flags), greedy, c=1..64, 5 repeats, but every request sends
# chat_template_kwargs enable_thinking, as sweep.py does. 256 tokens out, so most replies are
# thinking text. Separates the thinking setting from the prompts in the gap between the
# repeated-paragraph sweep and the mixed-prompt sweep.
set -u
cd "$(dirname "$0")"; HERE=$PWD
. ../2026-10-14-overnight-3050oem/lib.sh
DRIVER=$ROOT/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-queue-3050oem/driver.log
BIN=$ROOT/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-queue-3050oem/kvu-server.sh
MIXED=$ROOT/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-concurrency-mixed-3050oem/mixed.py
for spec in "e2b $E2B $D2 xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf" "e4b $E4B $D4 xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf"; do
  set -- $spec; m=$1 gguf=$2 drafter=$3 name=$4
  for arm in none q4x-n2; do
    out=$HERE/$m/$arm; mkdir -p $out; rm -f $out/results.jsonl
    d=""; [ $arm = q4x-n2 ] && d=$drafter
    serve $out/llama-server.log 32768 64 $gguf "$d" 2 || { state "SKIP J2 $m $arm"; continue; }
    state "START J2 $m $arm"
    python3 $MIXED --model $name --out $out/results.jsonl --mode greedy --thinking > $out/greedy.log 2>&1
    state "END J2 $m $arm rc=$?"
  done
done
