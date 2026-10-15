#!/bin/bash
# Concurrency with and without the exact Q4_0 drafter (2 drafted tokens), E2B and E4B, on the
# six mixed chat prompts instead of sweep.py's repeated paragraph. One server per (model, arm)
# at -c 32768 --parallel 64 (512 tokens per slot; prompts are under 40 tokens, replies 256),
# the rigs' flags otherwise. Retried at --parallel 32 and c<=32 if 64 slots do not fit.
# Greedy and sampled (temperature 0.8, one seed per request), c=1..64, 5 repeats.
# One shared KV cache (-kvu) and a fresh server per mode: with llama-server's default per-slot
# caches, decoding two requests at once fell to half speed depending on which slots were picked
# (perslot-kv/, seedtest/).
set -u
cd "$(dirname "$0")"; HERE=$PWD
. ../2026-10-14-overnight-3050oem/lib.sh
DRIVER=$HERE/driver.log
BIN=$HERE/kvu-server.sh
for spec in "e2b $E2B $D2 xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf" "e4b $E4B $D4 xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf"; do
  set -- $spec; m=$1 gguf=$2 drafter=$3 name=$4
  for arm in none q4x-n2; do
    out=$HERE/$m/$arm; mkdir -p $out; rm -f $out/results.jsonl
    d=""; [ $arm = q4x-n2 ] && d=$drafter
    nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu \
      --format=csv,noheader -l 1 > $out/gpu.csv & SMI=$!
    for mode in greedy sampled; do
      levels=1,2,4,8,16,32,64
      if ! serve $out/llama-server-$mode.log 32768 64 $gguf "$d" 2; then
        levels=1,2,4,8,16,32
        serve $out/llama-server-$mode.log 16384 32 $gguf "$d" 2 || { state "SKIP $m $arm $mode"; continue; }
      fi
      echo "$mode levels=$levels" >> $out/levels.txt
      state "START $m $arm $mode"
      python3 mixed.py --model $name --out $out/results.jsonl --levels $levels --mode $mode > $out/$mode.log 2>&1
      state "END $m $arm $mode rc=$?"
    done
    kill $SMI
  done
done
python3 summarize.py > summary.txt 2>&1
restore_live
state DONE
