#!/bin/bash
# Waits for the E2B agent run, then runs this E4B run and the 12B run. Stops everything at DEADLINE so the
# card is free on Monday morning; both runs are resumable (re-launch this script, or a run.sh alone).
set -u
ROOT=/home/xbill/gemma4-dev; HERE=$(cd "$(dirname "$0")" && pwd)
E2B=$ROOT/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-26-agent-e2b-3050oem
B12=$ROOT/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-11-27-agent-12b-3050oem
DEADLINE=${DEADLINE:-"2026-11-30 06:00"}; END=$(date -d "$DEADLINE" +%s)
log() { echo "$(date '+%F %T') $*" | tee -a $HERE/chain.log; }
( while [ $(date +%s) -lt $END ]; do sleep 60; done
  log "DEADLINE $DEADLINE: stopping"; pkill -f "$HERE/run.sh|$B12/run.sh"; pkill -f "agent.py $HERE|agent.py $B12"; pkill -x llama-server
  sleep 5; for d in $HERE $B12; do [ -f $d/tasks.jsonl ] || continue
    echo "$(date '+%F %T') stopped at deadline $DEADLINE; resumable" >> $d/driver.log
    ( cd $ROOT && git add "${d#$ROOT/}" && git commit -q -m "3050oem agent: stopped at deadline (${d##*/})" \
        -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "${d#$ROOT/}" && { git push -q || { git pull --no-rebase -q --no-edit && git push -q; }; } ); done
  log "deadline commit done"; kill $$ ) &
WATCH=$!
log "chain start, waiting for E2B run"
idle=0; until grep -q " DONE$" $E2B/driver.log || [ $idle -ge 5 ]; do
  sleep 60; if pgrep -f "agent.py $E2B" >/dev/null; then idle=0; else idle=$((idle+1)); fi; done
sleep 30  # let run.sh push its final commit
log "E2B run finished: $(tail -1 $E2B/driver.log)"
for d in $HERE $B12; do log "start ${d##*/}"; $d/run.sh; log "end ${d##*/}"; done
log "chain done"; kill $WATCH 2>/dev/null
