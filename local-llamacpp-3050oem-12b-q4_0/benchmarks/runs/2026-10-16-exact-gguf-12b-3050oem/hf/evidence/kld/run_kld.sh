#!/bin/bash
# KL divergence vs a bf16 reference with Google's metadata, CPU only, wikitext-2 test 16 x 512.
# Same method as local-llamacpp-1650ti-4b-q4_0/benchmarks/runs/2026-09-30-exact-gguf-e4b-1650ti/kld,
# on this host's llama.cpp (7182ef71e) rather than f95b0d9.
set -eu
PPL=/home/xbill/llama.cpp/build/bin/llama-perplexity
S=/home/xbill/models/gemma-4-12B-it-qat-q4_0-unquantized
G=/home/xbill/models/gemma-4-12B-it-qat-q4_0-gguf/gemma-4-12b-it-qat-q4_0.gguf
X=/home/xbill/models/gemma-4-12B-it-qat-q4_0-exact/gemma-4-12B-it-q4_0-exact.gguf
B=/home/xbill/models/gemma-4-12B-it-qat-q4_0-exact/gemma-4-12B-it-bf16-ref.gguf
W=/home/xbill/models/src/wikitext-2-raw/wiki.test.raw
BASE=/tmp/claude-1000/-home-xbill-gemma4-dev/3d9e1c9b-1cd7-4c7b-b9ad-cdc8f81cf9c5/scratchpad/base-bf16-12b.kld
cd $(dirname $0)
[ -f $B ] || python3 ../gguf_exact.py $S $G $B --bf16 > bf16-build.json
export CUDA_VISIBLE_DEVICES=
$PPL -m $B -f $W -c 512 --chunks 16 -t 6 --kl-divergence-base $BASE > kld-bf16-cpu.log 2>&1
$PPL -m $G -f $W -c 512 --chunks 16 -t 6 --kl-divergence-base $BASE --kl-divergence > kld-google-cpu.log 2>&1
$PPL -m $X -f $W -c 512 --chunks 16 -t 6 --kl-divergence-base $BASE --kl-divergence > kld-exact-cpu.log 2>&1
