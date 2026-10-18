#!/bin/bash
# fit.sh NGL CTX KVTYPE [extra...]: does llama-server load? prints VRAM or the failed allocation
n=$1; c=$2; kv=$3; shift 3
log=fit-ngl$n-c$c-$kv${TAG}.log
~/llama.cpp/build/bin/llama-server -m ~/models/gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf --host 127.0.0.1 --port 8090 -ngl $n -c $c -ctk $kv -ctv $kv -fa 1 -t 8 -tb 12 --parallel 1 --reasoning off "$@" > $log 2>&1 &
pid=$!
for i in $(seq 1 180); do
  if curl -sf localhost:8090/health >/dev/null; then echo "ngl=$n c=$c kv=$kv$TAG OK $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; kill $pid; wait $pid 2>/dev/null; exit 0; fi
  kill -0 $pid 2>/dev/null || { echo "ngl=$n c=$c kv=$kv$TAG FAIL $(grep -E 'cudaMalloc failed' $log | head -1 | sed 's/.*allocating/alloc/')"; exit 1; }
  sleep 1
done
echo "ngl=$n c=$c kv=$kv$TAG TIMEOUT"; kill $pid
