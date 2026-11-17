#!/bin/bash
# Keeps the 3050 busy: waits for wave 1 (E2B) to end, then runs wave 2 (E4B), then wave 3 (12B).
# Each wave's run.sh is resumable, so re-running this after a drop picks up where it stopped.
# On the gpu host:  setsid nohup ./chain.sh > nohup-chain.out 2>&1 < /dev/null &
set -u
R=/home/xbill/gemma4-dev
W1=$R/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem
while pgrep -f '^(/usr)?/bin/bash (\./)?(run|resume)\.sh$' | while read p; do [ "$(readlink /proc/$p/cwd)" = "$W1" ] && echo $p; done | grep -q .; do sleep 60; done
until [ -s $R/models-dl-done ] || grep -q '^ALL DONE' /home/xbill/models/wave23-dl.log 2>/dev/null; do sleep 60; done
grep -q 'SHA MISMATCH' /home/xbill/models/wave23-dl.log && { echo "download SHA mismatch; stopping"; exit 1; }
for w in local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-17-thinking-e4b-3050oem \
         local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-11-17-thinking-12b-3050oem; do
  (cd $R/$w && ./run.sh)
done
