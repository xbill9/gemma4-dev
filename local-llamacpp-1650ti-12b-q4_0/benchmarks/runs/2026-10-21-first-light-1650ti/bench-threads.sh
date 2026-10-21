#!/bin/bash
# Threads at -ngl 21 -ub 512, pp512 / tg64 as in bench.sh, one llama-bench per value, each started
# with the CPU package at or below 55 C and the GPU at or below 50 C (generation runs mostly on the
# CPU, so CPU heat is the confounder), in the mirrored order 6 4 8 12 12 8 4 6.
B=~/llama.cpp/build/bin/llama-bench
X=/home/xbill/models/gemma-4-12B-it-qat-q4_0-exact/gemma-4-12B-it-q4_0-exact.gguf
pkg() { sensors | awk '/Package id 0/{gsub(/[+°C]/,"",$4); print int($4)}'; }
cool() { until [ "$(pkg)" -le 55 ] && [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 50 ]; do sleep 5; done; }
i=0
for t in 6 4 8 12 12 8 4 6; do
  i=$((i+1)); cool
  echo "== t=$t pass=$i $(date +%T) pkg=$(pkg)C"
  $B -m $X -ngl 21 -fa 1 -p 512 -n 64 -r 2 -t $t -o md 2>/dev/null | tee bench-threads-$i-t$t.md | grep "tg64"
  echo "   end pkg=$(pkg)C"
done
