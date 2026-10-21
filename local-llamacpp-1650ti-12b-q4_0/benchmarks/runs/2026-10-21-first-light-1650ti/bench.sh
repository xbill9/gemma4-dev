#!/bin/bash
# llama-bench, -fa 1 -t 6, two repeats, pp512 / tg64: layers on the card for the exact Q4_0
# rebuild and Google's file, at the default -ub 512 and at -ub 128 (smaller compute buffer).
# Each -ngl value runs as its own llama-bench so an out-of-memory value does not end the rest.
B=~/llama.cpp/build/bin/llama-bench
X=/home/xbill/models/gemma-4-12B-it-qat-q4_0-exact/gemma-4-12B-it-q4_0-exact.gguf
G=/home/xbill/models/gemma-4-12B-it-qat-q4_0/gemma-4-12b-it-qat-q4_0.gguf
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 50 ]; do sleep 5; done; }
run() {  # NAME MODEL UB NGL
  cool
  echo "== $1 ub=$3 ngl=$4 $(date +%T) $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)C"
  $B -m $2 -ngl $4 -ub $3 -fa 1 -p 512 -n 64 -r 2 -t 6 -o md 2>bench-$1-ub$3-ngl$4.err > bench-$1-ub$3-ngl$4.md \
    && grep "^| gemma" bench-$1-ub$3-ngl$4.md || echo "FAIL $(grep -m1 'cudaMalloc failed' bench-$1-ub$3-ngl$4.err)"
}
for n in 0 12 16 18 19 20 21; do run exact $X 512 $n; done
for n in 20 21 22 23 24 25; do run exact $X 128 $n; done
for n in 16 19; do run google $G 512 $n; done
for n in 22; do run google $G 128 $n; done
