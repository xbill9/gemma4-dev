#!/bin/bash
# Replaces chain.sh after 12B (2026-10-08 18:10 UTC): the droplet credit runs out about
# 05:45 UTC on 2026-10-09, so 26B-A4B and 31B run three arms each, bf16, fp8 and
# q4w4a16, instead of seven. Waits for the 12B driver chain.sh started to exit.
set -u
cd "$(dirname "$0")/../../.."   # gpu-vllm-mi300x-2b
IMG=vllm/vllm-openai-rocm@sha256:ec62abecc13923172cf1225c0270b8cea7d84b652eacd3c8dffe2dd35dd73e37
DROPLET=debian-gpu-mi300x1-192gb-devcloud-atl1
RID=2026-10-07-dtype-sweep-mi300x
WAIT_PID=${1:?pid of the running 12B driver}
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) done --size 12b (driver $WAIT_PID exited)" >> benchmarks/runs/$RID/chain.log
for s in 26b 31b; do
    d=../gpu-vllm-mi300x-$s/benchmarks/runs/$RID
    mkdir -p $d
    echo "$(date -u +%FT%TZ) start --size $s --only bf16 fp8 q4w4a16" >> benchmarks/runs/$RID/chain.log
    python3 -u dtype_sweep.py --droplet $DROPLET --image $IMG --run-id $RID --size $s --only bf16 fp8 q4w4a16 > $d/driver.log 2>&1
    echo "$(date -u +%FT%TZ) done --size $s exit $?" >> benchmarks/runs/$RID/chain.log
done
