#!/bin/bash
# tpvr.sh wait|loads <label> <urls> — helpers for the 2026-10-02 tp-vs-replicas run on tpu-vllm-v5e4-2b-w8a8
set -u
RIG=tpu-vllm-v5e4-2b-w8a8; ROOT=/home/xbill/gemma4-dev
OUT=$ROOT/$RIG/benchmarks/runs/2026-10-02-tp-vs-replicas-v5e4
NODE=$RIG-node; Z=us-west4-a; P=aisprint-491218
MODEL=$(grep '^MODEL_NAME=' $ROOT/$RIG/tpu.env | cut -d= -f2)
S=$(dirname "$0")
mkdir -p $OUT
ssh_() { gcloud compute tpus tpu-vm ssh $NODE --zone=$Z --project=$P --worker=0 --quiet --command="$1" 2>/dev/null; }
log() { echo "$(date -u +%T) $*"; }
red() { grep -v '"GET \|"POST ' | grep -vE "hf_[A-Za-z0-9]{10,}" | sed -E 's/[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}/x.x.x.x/g'; }
case $1 in
wait)
  for i in $(seq 1 120); do
    st=$(gcloud alpha compute tpus queued-resources describe $RIG --zone=$Z --project=$P --format='value(state.state)' 2>/dev/null)
    [ "$st" = ACTIVE ] && ssh_ true && break
    case "$st" in FAILED|SUSPENDED) log "QR state $st"; exit 1;; esac
    sleep 30
  done
  log "ACTIVE, ssh ok"
  for i in $(seq 1 60); do
    s=$(ssh_ "curl -s -m 5 localhost:8000/v1/models >/dev/null && echo READY; sudo docker ps -a --filter name=vllm-gemma4 --format '{{.Status}}'")
    echo "$s" | grep -q READY && { log READY; break; }
    echo "$s" | grep -qE "BOOTFAIL|Exited" && { log "FAILED $s"; break; }
    sleep 45
  done
  ssh_ "sudo cat /var/log/vllm-startup.log" | red | grep -v HF_TOKEN > $OUT/startup.log
  ssh_ "sudo docker logs vllm-gemma4 2>&1" | red > $OUT/vllm-tp4.log
  gcloud alpha compute tpus queued-resources describe $RIG --zone=$Z --project=$P --format=json > $OUT/queued-resource.json 2>/dev/null
  ssh_ "curl -s localhost:8000/v1/models" > $OUT/verify-models.json
  ssh_ "curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"$MODEL\",\"temperature\":0,\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of Australia? Answer in one sentence.\"}]}'" > $OUT/verify-chat-tp4.json
  gcloud compute tpus tpu-vm scp $S/multi_client.py $NODE:/tmp/multi_client.py --zone=$Z --project=$P --worker=0 --quiet >/dev/null 2>&1
  log "wait done"
  ;;
loads)
  L=$2; URLS=$3; shift 3; CS=${*:-"1 4 16 64"}
  for c in $CS; do
    ssh_ "cd /tmp && JEV_LOAD_URLS=$URLS JEV_LOAD_CONCURRENCY=$c python3 multi_client.py load $MODEL /tmp/$L.load.c$c.json >/dev/null 2>/tmp/$L.err; cat /tmp/$L.load.c$c.json" > $OUT/$L.load.c$c.json
    log "$L load c$c $(python3 -c "import json;print(json.load(open('$OUT/$L.load.c$c.json'))['output_tok_per_s'])" 2>&1 | tail -1)"
  done
  ssh_ "cd /tmp && JEV_LOAD_URLS=$URLS JEV_LOAD_CONCURRENCY=16 JEV_LOAD_INPUT_TOKENS=2048 python3 multi_client.py loadlong $MODEL /tmp/$L.loadlong.c16.json >/dev/null 2>/tmp/$L.err; cat /tmp/$L.loadlong.c16.json" > $OUT/$L.loadlong.c16.json
  log "$L loadlong c16 $(python3 -c "import json;d=json.load(open('$OUT/$L.loadlong.c16.json'));print(d['output_tok_per_s'],d['total_tok_per_s'],d['ttft_median_s'])" 2>&1 | tail -1)"
  ;;
esac
