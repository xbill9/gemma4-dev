#!/bin/bash
# The trimmed T4 run, 2026-10-15 afternoon: the queue.sh plan cut to what the 3050 and 1650 Ti
# runs had not already answered. No cool-down between servers (COOL_C=999): the card is back at
# ~76 C within a minute of load, and each job carries its own no-drafter control.
#   1. W4A16 drafters (repack/repack_drafter.py) against bf16 QAT and no drafter, E2B then E4B
#      (spec_ab.py, the drafter sweep's harness, into ../2026-10-15-drafter-sweep-t4/e?b-q4ct)
#   2. conc.py: no drafter and QAT at 3 tokens, 1 / 4 / 8 requests in flight, one pass, both models
#   3. quality.py: GSM8K with and without QAT at 3 tokens, E2B only
# A failed job is logged and the next one runs. Start detached: setsid nohup ./trimmed.sh &
cd "$(dirname "$0")"
export COOL_C=999
R=$HOME/gemma4-dev
E2B=$R/gpu-vllm-t4-2b-w4a16
E4B=$R/gpu-vllm-t4-4b-q4w4a16emb4
SW=$(cd ../2026-10-15-drafter-sweep-t4 && pwd)
Q=trimmed.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $Q; }
job() { local name=$1; shift; say "START $name"; "$@" > $name.out 2>&1; say "END $name rc=$?"; }
if curl -fs -o /dev/null --max-time 3 http://127.0.0.1:8000/health; then say "port 8000 is serving; abort"; exit 1; fi
[ -n "$(awk 'NR>1' /proc/swaps)" ] || sudo swapon /opt1/swapfile
make -s -C $E2B verify-patch | grep -q 'Patched' || { say "Turing clamp not confirmed; abort"; exit 1; }
for m in E2B E4B; do
  d=/opt1/models/gemma-4-$m-it-qat-q4_0-w4a16-ct-assistant
  echo "$m $(sha256sum $d/model.safetensors | cut -c1-64) $(stat -c %s $d/model.safetensors) B"
done > $SW/q4ct-sha256.txt
Q4ARMS=none,qat-n3,q4ct-n3,q4ct-n4,none-again
job q4ct-e2b python3 $SW/spec_ab.py $E2B $SW/e2b-q4ct $Q4ARMS
python3 $SW/summarize.py $SW/e2b-q4ct > $SW/e2b-q4ct/summary.txt 2>&1
job q4ct-e4b python3 $SW/spec_ab.py $E4B $SW/e4b-q4ct $Q4ARMS
python3 $SW/summarize.py $SW/e4b-q4ct > $SW/e4b-q4ct/summary.txt 2>&1
export CONC=1,4,8 REPEATS=1
job conc-e2b python3 conc.py $E2B conc-e2b none,qat-n3
python3 summarize_conc.py conc-e2b > conc-e2b/summary.txt 2>&1
job conc-e4b python3 conc.py $E4B conc-e4b none,qat-n3
python3 summarize_conc.py conc-e4b > conc-e4b/summary.txt 2>&1
TASKS=gsm8k job quality-e2b python3 quality.py $E2B quality e2b none,qat-n3
say "TRIMMED DONE"
