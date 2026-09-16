#!/usr/bin/env bash
# 2026-09-16 thread + affinity lever sweep, local-llamacpp-cpu-2b-q4_0.
#
# llama-bench, NOT sweep.py: the thread count and CPU mask are server LAUNCH
# flags, so sweeping them through the HTTP path would need a server restart per
# cell. llama-bench isolates the lever. The cost is that these are engine
# numbers with no HTTP path — never difference them against a serving report.
set -euo pipefail
cd "$(dirname "$0")/../../.."
. ./tpu.env
B="$LLAMA_CPP_DIR/build-cpu/bin/llama-bench"

# CUDA_VISIBLE_DEVICES= is belt-and-braces: this is a GGML_CUDA=OFF build.
common=(-m "$MODEL_PATH" -ngl 0 -ctk f16 -ctv f16 -fa on -p 512 -n 128)

# Pass 1: thread count, unpinned. The lever tpu.env told us to sweep.
CUDA_VISIBLE_DEVICES= "$B" "${common[@]}" -t 4,8,12,16 -r 3 -o md

# Pass 2: affinity. Masks are logical CPUs on an i7-1360P:
#   0x55   = 0,2,4,6      4 distinct P-cores, no SMT sibling doubled
#   0x0F   = 0,1,2,3      2 P-cores with BOTH SMT siblings — CONFOUNDED, see REPORT.md
#   0xFF   = 0-7          all 4 P-cores, SMT doubled  <- clean SMT control vs 0x55
#   0xFF00 = 8-15         the 8 E-cores alone
#   0xFF55 = 0,2,4,6,8-15 4 distinct P-cores + all 8 E-cores
for spec in "4 0x55" "4 0x0F" "8 0xFF00" "12 0xFF55" "8 0xFF"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES= "$B" "${common[@]}" -t "$1" -C "$2" --cpu-strict 1 -r 5 -o md
done
