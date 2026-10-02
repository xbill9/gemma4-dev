#!/bin/bash
# measure.sh <max_model_len> <out dir> <input lens...>: on the VM. loadlong at 1 and 16 requests for each
# input length, then load at 16 and 64 requests.
L=$1; O=$2; shift 2
MODEL=xbill9/gemma-4-12B-it-qat-w8a8-int8-emb4
mkdir -p $O; cd /tmp
for n in "$@"; do for c in 1 16; do
  JEV_LOAD_CONCURRENCY=$c JEV_LOAD_INPUT_TOKENS=$n timeout 5400 python3 w4a16_client.py loadlong $MODEL $O/loadlong.in$n.c$c.json > $O/loadlong.in$n.c$c.out 2>&1
  echo "$(date -u +%T) in$n c$c rc=$?" >> $O/progress.txt
done; done
for c in 16 64; do
  JEV_LOAD_CONCURRENCY=$c timeout 3600 python3 w4a16_client.py load $MODEL $O/load.c$c.json > $O/load.c$c.out 2>&1
  echo "$(date -u +%T) load c$c rc=$?" >> $O/progress.txt
done
echo DONE >> $O/progress.txt
