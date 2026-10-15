#!/bin/bash
# div.py for E2B and E4B, then restores the E4B live server.
set -u
cd "$(dirname "$0")"
ROOT=/home/xbill/gemma4-dev
state() { echo "$(date +%T) $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" >> driver.log; }
sha256sum /usr/share/doc/NVIDIA_GLX-1.0/README.txt > doc.sha256
state BEGIN
pkill -x llama-server; sleep 3
for m in 2b 4b; do
  mkdir -p e$m
  state "START e$m"; python3 div.py $ROOT/local-llamacpp-3050oem-$m-q4_0 e$m > e$m/run.log 2>&1; state "END e$m rc=$?"
  python3 summarize.py e$m/results.jsonl > e$m/summary.txt 2>&1
done
(cd $ROOT/local-llamacpp-3050oem-4b-q4_0 && setsid nohup make serve > run/llama-server.log 2>&1 < /dev/null &)
for i in $(seq 120); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
state "E4B live server restored: $(curl -fsS http://127.0.0.1:8080/health)"
state DONE
