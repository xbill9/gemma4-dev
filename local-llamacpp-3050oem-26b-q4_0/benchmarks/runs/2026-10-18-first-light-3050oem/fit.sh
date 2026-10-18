#!/bin/bash
# fit.sh NCMOE CTX KVTYPE [extra...] -> prints load result + VRAM
n=$1; c=$2; kv=$3; shift 3
M=~/models/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf
log=fit-n$n-c$c-$kv${TAG}.log
~/llama.cpp/build/bin/llama-server -m $M --host 127.0.0.1 --port 8090 -ngl 99 -ncmoe $n -c $c -ctk $kv -ctv $kv -fa 1 -t 8 -tb 12 --parallel 1 --reasoning off "$@" > $log 2>&1 &
pid=$!
for i in $(seq 1 120); do
  if curl -sf localhost:8090/health >/dev/null; then echo "n=$n c=$c kv=$kv OK $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; kill $pid; wait $pid 2>/dev/null; exit 0; fi
  kill -0 $pid 2>/dev/null || { echo "n=$n c=$c kv=$kv FAIL: $(grep -E "cudaMalloc failed" $log | head -1)"; exit 1; }
  sleep 1
done
echo "n=$n c=$c kv=$kv TIMEOUT"; kill $pid
