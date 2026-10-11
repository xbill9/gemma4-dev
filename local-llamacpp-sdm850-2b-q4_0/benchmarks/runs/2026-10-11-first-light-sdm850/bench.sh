#!/usr/bin/env bash
# First-light llama-bench cells on the Snapdragon 850. Each cell waits until the
# hottest CPU sensor is at or below COOL_C before it starts, and records the
# hottest CPU sensor before and after. Usage: bench.sh <out_dir>
set -euo pipefail
OUT=${1:?out dir}; mkdir -p "$OUT"
BIN=/home/xbill/llama.cpp/build-cpu/bin/llama-bench
MODEL=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
COOL_C=${COOL_C:-55}
hot() { cat /sys/class/thermal/thermal_zone{1,2,3,4,7,8,9,10}/temp | sort -n | tail -1 | awk '{print $1/1000}'; }
cool() { while (( $(hot | cut -d. -f1) > COOL_C )); do sleep 10; done; }
# name | flags. cpus 0-3 Silver, 4-7 Gold: 0xF0 = Gold only, 0x0F = Silver only.
CELLS=(
  "t4-gold|-t 4 -C 0xF0"
  "t8-all|-t 8"
  "t4-unpinned|-t 4"
  "t2-gold|-t 2 -C 0x30"
  "t6-gold4silver2|-t 6 -C 0xF3"
  "t4-silver|-t 4 -C 0x0F"
)
for c in "${CELLS[@]}"; do
  name=${c%%|*}; flags=${c#*|}
  cool; t0=$(hot)
  # shellcheck disable=SC2086
  "$BIN" -m "$MODEL" -p 512 -n 128 -r 3 -fa 1 -ngl 0 $flags -o json > "$OUT/$name.json" 2> "$OUT/$name.log"
  t1=$(hot)
  echo "$name flags=[$flags] start_C=$t0 end_C=$t1" | tee -a "$OUT/cells.txt"
done
