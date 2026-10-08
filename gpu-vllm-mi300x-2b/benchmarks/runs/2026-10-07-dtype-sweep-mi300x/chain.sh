#!/bin/bash
# After the preregistered E2B arms: the two E2B fnuz arms, then E4B, 12B, 26B-A4B and 31B, one size
# at a time, all on the E2B sweep's image digest and run id. Each size's driver log goes beside its
# dtype-summary.json in that size's bf16 rig.
set -u
cd "$(dirname "$0")/../../.."   # gpu-vllm-mi300x-2b
IMG=vllm/vllm-openai-rocm@sha256:ec62abecc13923172cf1225c0270b8cea7d84b652eacd3c8dffe2dd35dd73e37
DROPLET=debian-gpu-mi300x1-192gb-devcloud-atl1
RID=2026-10-07-dtype-sweep-mi300x
until grep -q "^wrote" benchmarks/runs/$RID/driver.log; do sleep 60; done
python3 -u dtype_sweep.py --droplet $DROPLET --image $IMG --run-id $RID --only fp8fnuz fp8fnuzemb4 \
    >> benchmarks/runs/$RID/driver.log 2>&1
for s in 4b 12b 26b 31b; do
    d=../gpu-vllm-mi300x-$s/benchmarks/runs/$RID
    mkdir -p $d
    echo "$(date -u +%FT%TZ) start --size $s" >> benchmarks/runs/$RID/chain.log
    python3 -u dtype_sweep.py --droplet $DROPLET --image $IMG --run-id $RID --size $s > $d/driver.log 2>&1
    echo "$(date -u +%FT%TZ) done --size $s exit $?" >> benchmarks/runs/$RID/chain.log
done
