#!/bin/bash
# Drafter gaps on the RTX 3050 OEM: speed.py then longctx.py for E2B and E4B, then restores
# the E4B live server (make serve in local-llamacpp-3050oem-4b-q4_0).
set -u
cd "$(dirname "$0")"
ROOT=/home/xbill/gemma4-dev
sha256sum /usr/share/doc/NVIDIA_GLX-1.0/README.txt > doc.sha256
state() { echo "$(date +%T) $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" >> driver.log; }
state BEGIN
pkill -x llama-server; sleep 3
for m in e2b e4b; do
  case $m in e2b) rig=$ROOT/local-llamacpp-3050oem-2b-q4_0;; e4b) rig=$ROOT/local-llamacpp-3050oem-4b-q4_0;; esac
  mkdir -p $m/speed $m/longctx
  state "START speed $m"; python3 speed.py $rig $m/speed > $m/speed/run.log 2>&1; state "END speed $m rc=$?"
  state "START longctx $m"; python3 longctx.py $rig $m/longctx > $m/longctx/run.log 2>&1; state "END longctx $m rc=$?"
done
(cd $ROOT/local-llamacpp-3050oem-4b-q4_0 && setsid nohup make serve > run/llama-server.log 2>&1 < /dev/null &)
for i in $(seq 120); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
state "E4B live server restored: $(curl -fsS http://127.0.0.1:8080/health)"
state DONE
