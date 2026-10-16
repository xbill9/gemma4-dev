#!/bin/bash
# Long, varied prompts at 1-32 requests at once: model x drafter x KV layout, greedy.
# Two passes, the second in reverse order; a fresh server per (pass, arm). Each pass is committed
# and pushed as it finishes; live E4B server restored at the end.
set -u
cd "$(dirname "$0")"; HERE=$PWD
. ../2026-10-14-overnight-3050oem/lib.sh
DRIVER=$HERE/driver.log
REL=local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-concurrency-longmixed-3050oem
ARMS=()
for m in e2b e4b; do for kv in perslot kvu; do for d in none q4x-n2; do ARMS+=("$m $kv $d"); done; done; done
run_arm() {
  local pas=$1 m=$2 kv=$3 d=$4 g dr
  [ $m = e2b ] && { g=$E2B; dr=$D2; } || { g=$E4B; dr=$D4; }
  [ $kv = kvu ] && BIN=$HERE/kvu-server.sh || BIN=/home/xbill/llama.cpp/build/bin/llama-server
  mkdir -p $m/$kv-$d
  if [ $d = none ]; then serve $m/$kv-$d/server-pass$pas.log 131072 32 $g || return
  else serve $m/$kv-$d/server-pass$pas.log 131072 32 $g $dr 2 || return; fi
  state "START pass $pas $m $kv $d"
  python3 longmixed.py --model $m --out $m/$kv-$d/results.jsonl --pass $pas > $m/$kv-$d/pass$pas.out 2>&1
  state "END pass $pas $m $kv $d rc=$? peak $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}
state "RUN START"
for a in "${ARMS[@]}"; do run_arm 1 $a; done
commit_push $REL "3050oem: long mixed-prompt concurrency, pass 1 (raw results)"
# lib.sh serve() uses a global i; this loop must not.
for ((k=${#ARMS[@]}-1; k>=0; k--)); do run_arm 2 ${ARMS[$k]}; done
restore_live
state "RUN DONE"
commit_push $REL "3050oem: long mixed-prompt concurrency, pass 2 in reverse order (raw results)"
