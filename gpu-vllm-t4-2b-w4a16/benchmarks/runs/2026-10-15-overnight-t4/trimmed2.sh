#!/bin/bash
# Steps 2 and 3 of trimmed.sh rerun WITH the cool-down (COOL_C=50, as the drafter sweep). trimmed.sh
# ran without it, and its W4A16 drafter job showed why that fails on this card: the no-drafter arm
# run first on a 52-63 C card decoded 113.8 tok/s and the same arm run last at 71-81 C 104.8, SM
# clock dipping to 1,230 MHz. trimmed.sh is stopped after its W4A16 drafter job and this takes over.
cd "$(dirname "$0")"
export COOL_C=50
R=$HOME/gemma4-dev
E2B=$R/gpu-vllm-t4-2b-w4a16
E4B=$R/gpu-vllm-t4-4b-q4w4a16emb4
Q=trimmed.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $Q; }
job() { local name=$1; shift; say "START $name"; "$@" > $name.out 2>&1; say "END $name rc=$?"; }
rm -rf conc-e2b conc-e4b quality conc-e2b.out conc-e4b.out
export CONC=1,4,8 REPEATS=1
job conc-e2b python3 conc.py $E2B conc-e2b none,qat-n3
python3 summarize_conc.py conc-e2b > conc-e2b/summary.txt 2>&1
job conc-e4b python3 conc.py $E4B conc-e4b none,qat-n3
python3 summarize_conc.py conc-e4b > conc-e4b/summary.txt 2>&1
TASKS=gsm8k job quality-e2b python3 quality.py $E2B quality e2b none,qat-n3
say "TRIMMED DONE"
