#!/bin/bash
# KL divergence vs a bf16 reference with Google's metadata, CPU only, wikitext-2 test 16 x 512.
# Same method as local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-09-29-exact-gguf-v2-1650ti/kld.
set -eu
PPL=/home/xbill/llama.cpp-f95b0d9/build/bin/llama-perplexity
S=$(ls -d ~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-unquantized/snapshots/*)
G=$(ls ~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-q4_0-gguf/snapshots/*/gemma-4-E4B_q4_0-it.gguf)
X=/home/xbill/models/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-q4_0-exact.gguf
B=/home/xbill/models/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-bf16-ref.gguf
W=/home/xbill/models/src/wikitext-2-raw/wiki.test.raw
BASE=/tmp/claude-1000/-home-xbill-gemma4-dev/bc514cf6-12cd-470f-992d-11499902efdc/scratchpad/base-bf16-e4b.kld
cd $(dirname $0)
[ -f $B ] || python3 ../gguf_exact.py $S $G $B --bf16 > bf16-build.json
export CUDA_VISIBLE_DEVICES=
$PPL -m $B -f $W -c 512 --chunks 16 -t 6 --kl-divergence-base $BASE > kld-bf16-cpu.log 2>&1
$PPL -m $G -f $W -c 512 --chunks 16 -t 6 --kl-divergence-base $BASE --kl-divergence > kld-google-cpu.log 2>&1
$PPL -m $X -f $W -c 512 --chunks 16 -t 6 --kl-divergence-base $BASE --kl-divergence > kld-exact-cpu.log 2>&1
