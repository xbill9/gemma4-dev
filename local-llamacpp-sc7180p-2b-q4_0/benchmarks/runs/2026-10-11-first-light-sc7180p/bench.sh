#!/usr/bin/env bash
# First-light llama-bench cells on the Snapdragon 7c Gen 2 (SC7180P). Each cell
# waits until the hottest CPU sensor is at or below COOL_C before it starts, and
# records the hottest CPU sensor before and after, the swap-ins during the cell
# (pswpin from /proc/vmstat) and llama-bench's peak RSS. Usage: bench.sh <out_dir>
set -euo pipefail
OUT=${1:?out dir}; mkdir -p "$OUT"
BIN=/home/xbill/llama.cpp/build-cpu/bin/llama-bench
MODEL=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
COOL_C=${COOL_C:-45}
# cpu0..cpu9 thermal sensors (zones 1-6, 9-12); cpuss and GPU zones excluded.
hot() { cat /sys/class/thermal/thermal_zone{1,2,3,4,5,6,9,10,11,12}/temp | sort -n | tail -1 | awk '{print $1/1000}'; }
cool() { while (( $(hot | cut -d. -f1) > COOL_C )); do sleep 10; done; }
pswpin() { awk '$1=="pswpin"{print $2}' /proc/vmstat; }
# name | flags. cpus 0-5 Silver, 6-7 Gold: 0xC0 = Gold only, 0x3F = Silver only.
CELLS=(
  "t2-gold|-t 2 -C 0xC0"
  "t8-all|-t 8"
  "t2-unpinned|-t 2"
  "t1-gold|-t 1 -C 0x40"
  "t4-gold2silver2|-t 4 -C 0xCC"
  "t6-silver|-t 6 -C 0x3F"
)
for c in "${CELLS[@]}"; do
  name=${c%%|*}; flags=${c#*|}
  cool; t0=$(hot); s0=$(pswpin)
  # shellcheck disable=SC2086
  /usr/bin/time -f "maxrss_kb=%M majflt=%F" -o "$OUT/$name.time" \
    "$BIN" -m "$MODEL" -p 512 -n 128 -r 3 -fa 1 -ngl 0 $flags -o json > "$OUT/$name.json" 2> "$OUT/$name.log"
  t1=$(hot); s1=$(pswpin)
  echo "$name flags=[$flags] start_C=$t0 end_C=$t1 pswpin=$((s1-s0)) $(cat "$OUT/$name.time")" | tee -a "$OUT/cells.txt"
done
