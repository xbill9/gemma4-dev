#!/bin/bash
# Finishes E4B longctx after the 19:56 interruption: the three drafter arms, appended to
# e4b/longctx/results.jsonl beside the completed no-drafter arm. Then restores the live server.
set -u
cd "$(dirname "$0")"
ROOT=/home/xbill/gemma4-dev
state() { echo "$(date +%T) $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" >> driver.log; }
state "INTERRUPTED longctx e4b after arm none (live server restarted on 8080 at 19:57:14 by something outside run.sh)"
pkill -x llama-server; sleep 3
state "START longctx e4b q4x-n1,q4x-n2,nq-n2"
python3 longctx.py $ROOT/local-llamacpp-3050oem-4b-q4_0 e4b/longctx q4x-n1,q4x-n2,nq-n2 >> e4b/longctx/run.log 2>&1
state "END longctx e4b rc=$?"
(cd $ROOT/local-llamacpp-3050oem-4b-q4_0 && setsid nohup make serve > run/llama-server.log 2>&1 < /dev/null &)
for i in $(seq 120); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
state "E4B live server restored: $(curl -fsS http://127.0.0.1:8080/health)"
state DONE
