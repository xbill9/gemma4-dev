#!/bin/bash
# Overnight queue on the RTX 3050 OEM, 2026-10-14/15. Jobs in order, each committed and pushed
# when it finishes (B also after each of its four arms):
#   A ../2026-10-14-concurrency-drafter-3050oem  concurrency sweep, drafter off and on, E2B and E4B
#   C ../2026-10-14-accuracy-cache-3050oem       GSM8K/BFCL with the prompt cache off vs on
#   B ../2026-10-14-accuracy-fullctx-3050oem     GSM8K/BFCL at -c 131072, one request at a time
# The live E4B server (make serve, port 8080) is down throughout and restored at the end.
set -u
cd "$(dirname "$0")"
. ./lib.sh
RUNS=local-llamacpp-3050oem-2b-q4_0/benchmarks/runs
state BEGIN
stop8080
state "JOB A start"; bash ../2026-10-14-concurrency-drafter-3050oem/run.sh; state "JOB A end rc=$?"
commit_push $RUNS/2026-10-14-concurrency-drafter-3050oem "3050oem overnight A: concurrency sweep with and without the drafter (raw results)"
state "JOB C start"; bash ../2026-10-14-accuracy-cache-3050oem/run.sh; state "JOB C end rc=$?"
commit_push $RUNS/2026-10-14-accuracy-cache-3050oem "3050oem overnight C: accuracy with the prompt cache off (raw results)"
state "JOB B start"; bash ../2026-10-14-accuracy-fullctx-3050oem/run.sh; state "JOB B end rc=$?"
restore_live
state DONE
commit_push $RUNS/2026-10-14-overnight-3050oem "3050oem overnight: queue log"
