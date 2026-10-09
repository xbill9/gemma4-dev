#!/bin/bash
# 2026-10-09: VLLM_ROCM_USE_AITER=1 against the stock settings for 12B fp8, same droplet,
# same day, same image digest as the 2026-10-07 dtype sweep.
set -u
cd "$(dirname "$0")/../../.."   # gpu-vllm-mi300x-2b
IMG=vllm/vllm-openai-rocm@sha256:ec62abecc13923172cf1225c0270b8cea7d84b652eacd3c8dffe2dd35dd73e37
DROPLET=debian-gpu-mi300x1-192gb-devcloud-atl1
L=benchmarks/runs/2026-10-09-aiter-mi300x/chain.log
echo "$(date -u +%FT%TZ) start aiter fp8" >> $L
python3 -u dtype_sweep.py --droplet $DROPLET --image $IMG --run-id 2026-10-09-aiter-mi300x --size 12b --only fp8 \
    --engine-env VLLM_ROCM_USE_AITER=1 --seed-base 60000000 > ../gpu-vllm-mi300x-12b/benchmarks/runs/2026-10-09-aiter-mi300x/driver.log 2>&1
echo "$(date -u +%FT%TZ) done aiter fp8 exit $?; start stock fp8" >> $L
python3 -u dtype_sweep.py --droplet $DROPLET --image $IMG --run-id 2026-10-09-stock-mi300x --size 12b --only fp8 \
    --seed-base 61000000 > ../gpu-vllm-mi300x-12b/benchmarks/runs/2026-10-09-stock-mi300x/driver.log 2>&1
echo "$(date -u +%FT%TZ) done stock fp8 exit $?" >> $L
