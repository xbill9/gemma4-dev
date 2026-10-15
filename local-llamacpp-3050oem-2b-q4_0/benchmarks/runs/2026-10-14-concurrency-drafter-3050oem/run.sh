#!/bin/bash
# Job A. Concurrency sweep with and without the exact Q4_0 drafter (2 drafted tokens), E2B and E4B.
# Same sweep as 2026-10-13-concurrency-3050oem (sweep.py, c=1..64, 5 repeats, unique prompts,
# 512 in/128 out and 128 in/512 out), one server per (model, arm) at -c 65536 --parallel 64
# (1024 tokens per slot) with today's serving flags (per_layer_token_embd on the card). If a
# server does not come up at 64 slots it is retried at -c 32768 --parallel 32 and c=1..32.
# Aggregate output is total tokens over wall time; TPOT counts stream gaps, which bunch up when
# drafted tokens are accepted together, so read aggregate and TTFT for the drafted arms.
set -u
cd "$(dirname "$0")"; HERE=$PWD
. ../2026-10-14-overnight-3050oem/lib.sh
SWEEP=$ROOT/local-llamacpp-3050oem-2b-q4_0/sweep.py
for spec in "e2b $E2B $D2 xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf" "e4b $E4B $D4 xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf"; do
  set -- $spec; m=$1 gguf=$2 drafter=$3 name=$4
  for arm in none q4x-n2; do
    out=$HERE/$m/$arm; mkdir -p $out
    d=""; [ $arm = q4x-n2 ] && d=$drafter
    levels=1,2,4,8,16,32,64
    if ! serve $out/llama-server.log 65536 64 $gguf "$d" 2; then
      levels=1,2,4,8,16,32
      serve $out/llama-server.log 32768 32 $gguf "$d" 2 || { state "SKIP A $m $arm"; continue; }
    fi
    echo "levels=$levels" > $out/levels.txt
    nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu \
      --format=csv,noheader -l 1 > $out/gpu.csv & SMI=$!
    for shape in "512 128 in512-out128" "128 512 in128-out512"; do
      set -- $shape
      state "START A $m $arm $3"
      curl -fsS http://127.0.0.1:8080/metrics > $out/metrics-before-$3.prom
      (cd $ROOT/local-llamacpp-3050oem-2b-q4_0 && python3 $SWEEP --base http://127.0.0.1:8080/v1 --model $name \
        --out $out/$3 --concurrency $levels --conc-input $1 --conc-output $2 --repeats 5 --prompt-mode unique \
        --decode-source auto --expect-device gpu) > $out/sweep-$3.log 2>&1
      state "END A $m $arm $3 rc=$?"
      curl -fsS http://127.0.0.1:8080/metrics > $out/metrics-after-$3.prom
    done
    kill $SMI
  done
done
stop8080
python3 summarize.py > summary.txt 2>&1
