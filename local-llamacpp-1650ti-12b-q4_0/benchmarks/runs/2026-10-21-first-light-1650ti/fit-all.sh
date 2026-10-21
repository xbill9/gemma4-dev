#!/bin/bash
# The most layers on the card that load, for each context size, cache type and drafter on/off,
# at -ub 512 and --parallel 1 (fit.sh per try, descending -ngl until one loads).
D=/home/xbill/models/gemma-4-12B-it-qat-assistant/gemma-4-12B-it-qat-assistant-q4_0-exact.gguf
try() {  # KV CTX DRAFT(0/1)
  local kv=$1 c=$2 dr=$3 extra=() tag=""
  [ $dr = 1 ] && { extra=(-md $D --spec-type draft-mtp -ngld all --spec-draft-n-max 2); tag=-draft; }
  for n in 25 24 23 22 21 20 19 18 17 16 15 14 13 12 10 8 6 4 2 0; do
    TAG=$tag ./fit.sh $n $c $kv "${extra[@]}" && return 0
  done
}
for dr in 1 0; do
  for c in 8192 32768 131072; do try q4_0 $c $dr; done
  for c in 8192 32768; do try f16 $c $dr; done
done
