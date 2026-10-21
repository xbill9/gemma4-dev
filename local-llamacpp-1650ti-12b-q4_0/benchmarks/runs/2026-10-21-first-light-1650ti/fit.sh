#!/bin/bash
# fit.sh NGL CTX KVTYPE [extra...]: does llama-server load? prints VRAM or the failed allocation.
# The 3050 31B sister's fit.sh with this rig's model, -t 6 and the 1650 Ti. TAG names the variant
# (e.g. TAG=-draft with the drafter flags passed as extra).
n=$1; c=$2; kv=$3; shift 3
log=fit-ngl$n-c$c-$kv${TAG}.log
M=/home/xbill/models/gemma-4-12B-it-qat-q4_0-exact/gemma-4-12B-it-q4_0-exact.gguf
~/llama.cpp/build/bin/llama-server -m $M --host 127.0.0.1 --port 8090 -ngl $n -c $c -ctk $kv -ctv $kv -fa 1 -t 6 -tb 12 --parallel 1 --reasoning off "$@" > $log 2>&1 &
pid=$!
for i in $(seq 1 240); do
  if curl -sf localhost:8090/health >/dev/null; then echo "ngl=$n c=$c kv=$kv$TAG OK $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; kill $pid; wait $pid 2>/dev/null; exit 0; fi
  kill -0 $pid 2>/dev/null || { echo "ngl=$n c=$c kv=$kv$TAG FAIL $(grep -E 'cudaMalloc failed' $log | head -1 | sed 's/.*allocating/alloc/')"; exit 1; }
  sleep 1
done
echo "ngl=$n c=$c kv=$kv$TAG TIMEOUT"; kill $pid; wait $pid 2>/dev/null
