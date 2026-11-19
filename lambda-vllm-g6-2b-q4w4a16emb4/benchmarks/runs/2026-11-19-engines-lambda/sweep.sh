#!/bin/bash
# Lambda serving-engine sweep driver (run from ~/gemma4-dev): one rig at a time.
# Publish; whatever the publish status says, try a request every 60 s for up to 30 min (Lambda can mark a
# publish Failed before a slow engine finishes Init, then the version turns Active on its own); bench x4; down.
set -u
OUT=benchmarks/runs/2026-11-19-lambda
for r in "$@"; do
  cd ~/gemma4-dev/$r || continue
  echo "== $r $(date -u +%H:%M:%S)"
  python3 lmi.py function > deploy-sweep.log 2>&1; echo "  publish exit $? $(date -u +%H:%M:%S): $(grep -v '^+ aws' deploy-sweep.log | tail -1 | cut -c1-140)"
  ok=0
  if grep -q Traceback deploy-sweep.log; then echo "  publish crashed, skipping"; continue; fi
  for t in $(seq 1 30); do
    if python3 lmi.py ask "Name the planets in one line." > ask.log 2>&1; then ok=1; break; fi
    sleep 60
  done
  echo "  first answer $(date -u +%H:%M:%S) ok=$ok"; grep -v '^+ aws' ask.log | tail -3
  if [ $ok = 1 ]; then
    for k in 1 2 3 4; do python3 bench.py $OUT/run$k --levels 1,4 2>&1 | tail -2; done
  fi
  python3 lmi.py down 2>&1 | grep '^scaling' || echo "  down: $(python3 lmi.py down 2>&1 | tail -1)"
done
echo "SWEEP DONE $(date -u +%H:%M:%S)"
