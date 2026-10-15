#!/bin/bash
# Three 3050 gap jobs back to back, each committed and pushed as it finishes; live E4B server
# restored at the end. Server binary: kvu-server.sh (llama-server ... -kvu).
set -u
cd "$(dirname "$0")"; HERE=$PWD
. ../2026-10-14-overnight-3050oem/lib.sh
DRIVER=$HERE/driver.log
R=$ROOT/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs
state "QUEUE START"
bash $R/2026-10-15-concurrency-drafter-kvu-3050oem/run.sh > $R/2026-10-15-concurrency-drafter-kvu-3050oem/run.out 2>&1
commit_push local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-concurrency-drafter-kvu-3050oem "3050oem: repeated-paragraph concurrency sweep rerun with -kvu (raw results)"
bash $R/2026-10-15-concurrency-thinking-3050oem/run.sh > $R/2026-10-15-concurrency-thinking-3050oem/run.out 2>&1
commit_push local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-concurrency-thinking-3050oem "3050oem: mixed-prompt concurrency with thinking on (raw results)"
bash $R/2026-10-15-concurrency-draft1-3050oem/run.sh > $R/2026-10-15-concurrency-draft1-3050oem/run.out 2>&1
commit_push local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-concurrency-draft1-3050oem "3050oem: mixed-prompt concurrency at 1 drafted token (raw results)"
restore_live
state "QUEUE DONE"
commit_push local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-15-queue-3050oem "3050oem: gap queue log"
