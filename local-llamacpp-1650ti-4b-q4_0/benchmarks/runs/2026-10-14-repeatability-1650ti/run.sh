#!/bin/bash
# Runs ON THE LAPTOP from ~/overnight/2026-10-14-repeatability-1650ti. Waits for the other laptop
# queue (2026-10-14-drafter-gaps-1650ti/driver.log reaching DONE) so the two never share the card.
set -u
cd "$(dirname "$0")"
G=$HOME/gemma4-dev/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-1650ti/driver.log
state() { echo "$(date '+%F %T') $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" >> driver.log; }
state WAITING
until grep -q DONE "$G" 2>/dev/null; do sleep 60; done
while pgrep -x llama-server >/dev/null; do sleep 10; done
state BEGIN
for b in ~/llama.cpp/build/bin/llama-server ~/llama.cpp-77c4540ce/build/bin/llama-server; do echo "$b: $($b --version 2>&1 | grep -i version)"; done > host.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >> host.txt
python3 repeat.py . > run.log 2>&1
state "END rc=$?"
python3 summarize.py results.jsonl > summary.txt 2>&1
state DONE
