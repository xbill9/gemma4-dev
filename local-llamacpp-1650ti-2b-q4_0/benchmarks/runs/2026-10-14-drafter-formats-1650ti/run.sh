#!/bin/bash
# Drafter precision on the 1650 Ti: Q8_0, exact Q4_0 and bf16 drafters, E2B then E4B, each at the
# draft length its rig serves with (tpu.env SPEC_DRAFT_N_MAX: E2B 1, E4B 2); E2B also at 2 to line
# up with the 3050 run. No drafter and Q8_0 run first and again last. Nothing else may hold the card.
set -eu
cd "$(dirname "$0")"
R=$HOME/gemma4-dev
if pgrep -x llama-server >/dev/null; then echo "a llama-server is running; stop it first" >&2; exit 1; fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee host.txt
$(grep ^LLAMA_SERVER_BIN= $R/local-llamacpp-1650ti-2b-q4_0/tpu.env | cut -d= -f2) --version 2>&1 | tail -2 | tee -a host.txt
sha256sum -c $R/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-q4exact-3050oem/sha256.txt | tee sha256.log
python3 spec_ab.py $R/local-llamacpp-1650ti-2b-q4_0 e2b \
  none,q8-n1,q4x-n1,bf16-n1,q8-n2,q4x-n2,bf16-n2,q8-n1-again,none-again > e2b.log 2>&1
python3 summarize.py e2b | tee e2b/summary.txt
python3 spec_ab.py $R/local-llamacpp-1650ti-4b-q4_0 e4b \
  none,q8-n2,q4x-n2,bf16-n2,q8-n2-again,none-again > e4b.log 2>&1
python3 summarize.py e4b | tee e4b/summary.txt
echo DONE
