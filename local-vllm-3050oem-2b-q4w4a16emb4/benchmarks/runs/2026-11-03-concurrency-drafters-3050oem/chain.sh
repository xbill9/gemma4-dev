#!/bin/bash
# vLLM arms, then llama.cpp arms, one engine on the card at a time.
cd /home/xbill/gemma4-dev
(cd local-vllm-3050oem-2b-q4w4a16emb4 && bash benchmarks/runs/2026-11-03-concurrency-drafters-3050oem/run.sh) \
  > local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-concurrency-drafters-3050oem/run.log 2>&1
echo "vllm rc=$?" >> local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-concurrency-drafters-3050oem/chain.log
(cd local-llamacpp-3050oem-2b-q4_0 && bash benchmarks/runs/2026-11-03-concurrency-drafters-3050oem/run.sh) \
  > local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-03-concurrency-drafters-3050oem/run.log 2>&1
echo "llama rc=$?" >> local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-concurrency-drafters-3050oem/chain.log
echo DONE >> local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-concurrency-drafters-3050oem/chain.log
