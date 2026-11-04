#!/bin/bash
# One at a time on the card: the concurrency sweep, then GSM8K/BFCL.
cd /home/xbill/gemma4-dev/local-mistralrs-3050oem-2b-q4_0
while pgrep -x mistralrs >/dev/null; do sleep 2; done
bash benchmarks/runs/2026-11-04-concurrency-3050oem/run.sh > benchmarks/runs/2026-11-04-concurrency-3050oem/run.log 2>&1
echo "sweep exit $?"
while pgrep -x mistralrs >/dev/null; do sleep 2; done
bash benchmarks/runs/2026-11-04-accuracy-3050oem/run.sh
echo "accuracy exit $?"
