#!/bin/bash
# Fit and timing check before the overnight run: E4B with the drafter (the largest arm) at
# -c 131072 --parallel 32, per-slot and shared KV, one repeat at c=32.
set -u
cd "$(dirname "$0")"; HERE=$PWD
. ../2026-10-14-overnight-3050oem/lib.sh
DRIVER=$HERE/fit.log
mkdir -p fit
for kv in perslot kvu; do
  [ $kv = kvu ] && BIN=$HERE/kvu-server.sh || BIN=/home/xbill/llama.cpp/build/bin/llama-server
  serve fit/server-$kv.log 131072 32 $E4B $D4 2 || continue
  state "fit $kv: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  t0=$(date +%s)
  python3 longmixed.py --model e4b --out fit/$kv.jsonl --levels 32 --repeats 1 --pass 0
  state "fit $kv: c=32 x1 took $(( $(date +%s) - t0 )) s, peak $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
done
restore_live
