#!/bin/bash
# After run.sh and run_q.sh: tune.py (q4_0 fit, -ub, drafted tokens), then recall.py (long-context
# recall by cache type). Stops the rig's live server first and restores it (make serve) at the end.
set -u
cd "$(dirname "$0")"; HERE=$PWD
RIG=/home/xbill/gemma4-dev/local-llamacpp-3050oem-12b-q4_0
LOG=$HERE/driver.log
state() { echo "$(date '+%F %T') $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$LOG"; }
stopall() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
state "BEGIN phase3"
stopall
state "START tune"
python3 tune.py $RIG $HERE > $HERE/tune.out 2>&1
state "END tune rc=$?"
stopall
state "START recall"
python3 recall.py $RIG $HERE \
  f16-none:f16:40960:none:16384,32768 \
  q8-none:q8_0:65536:none:16384,32768,49152,64000 \
  q4-none:q4_0:131072:none:16384,32768,64000,98304,126976 \
  q4-draft:q4_0:81920:draft:32768,64000 > $HERE/recall.out 2>&1
state "END recall rc=$?"
stopall
(cd $RIG && setsid nohup make serve > run/llama-server.log 2>&1 < /dev/null &)
for i in $(seq 180); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
state "12B live server restored: $(curl -fsS http://127.0.0.1:8080/health)"
state "DONE phase3"
