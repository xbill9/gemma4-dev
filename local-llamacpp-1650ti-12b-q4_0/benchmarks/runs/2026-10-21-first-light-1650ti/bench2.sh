#!/bin/bash
# Threads at -ngl 21 (-ub 512, tg64), then prompt batch size: pp2048 with -b 2048 at -ub 512 /
# 1024 / 2048, each at the most layers that load (tried from high to low).
B=~/llama.cpp/build/bin/llama-bench
X=/home/xbill/models/gemma-4-12B-it-qat-q4_0-exact/gemma-4-12B-it-q4_0-exact.gguf
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 50 ]; do sleep 5; done; }
cool; echo "== threads $(date +%T)"
$B -m $X -ngl 21 -fa 1 -p 0 -n 64 -r 2 -t 4,6,8,12 -o md 2>bench-threads.err | tee bench-threads.md | grep "^| gemma"
for ub in 512 1024 2048; do
  for n in 21 18 16 14 12 10 8 6; do
    cool; echo "== ub=$ub ngl=$n $(date +%T)"
    if $B -m $X -ngl $n -b 2048 -ub $ub -fa 1 -p 2048 -n 0 -r 2 -t 6 -o md 2>bench-ub$ub-ngl$n.err > bench-ub$ub-ngl$n.md; then
      grep "^| gemma" bench-ub$ub-ngl$n.md; break
    else echo "FAIL"; rm -f bench-ub$ub-ngl$n.md; fi
  done
done
