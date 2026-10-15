#!/bin/bash
# Runs on the 3050 host: copies the laptop's results back when its driver.log says DONE, commits, pushes.
set -u
cd "$(dirname "$0")"; HERE=$PWD
. $HOME/gemma4-dev/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-overnight-3050oem/lib.sh
L=xbill@192.168.0.186
RD=overnight/2026-10-14-repeatability-1650ti
until ssh -o BatchMode=yes $L "grep -q DONE $RD/driver.log" 2>/dev/null; do sleep 120; done
scp -q -r "$L:$RD/*" $HERE/
commit_push local-llamacpp-1650ti-4b-q4_0/benchmarks/runs/2026-10-14-repeatability-1650ti "1650ti overnight E: E4B repeatability at temperature 0 (raw results)"
