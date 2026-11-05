#!/bin/bash
# Everything after the one-request run, in order, one engine on the card at a time and nothing else on the
# CPUs. Each step is resumable: a step whose status.json says rc 0 is skipped.
#   1. mistral.rs one-request arms (engines.py, same results.jsonl), once the binary exists
#   2. load sweeps (t4chain.py sweep): every draft length on vLLM, SGLang and llama.cpp, mistral.rs without
#   3. accuracy (t4chain.py accuracy): every engine without a drafter, and with its best one-request drafter
#      on the T4 (vllm-d3, llama-d1, sglang-d4); LiteRT through its own run.sh
set -u
cd "$(dirname "$0")"
G=$HOME/gemma4-dev
say() { echo "$(date +%F\ %T) $*" | tee -a queue.log; }
rigdir() { case $1 in vllm*) echo $G/gpu-vllm-t4-2b-q4w4a16emb4 ;; sglang*) echo $G/gpu-sglang-t4-2b-q4_0 ;;
  llama*) echo $G/gpu-llamacpp-t4-2b-q4_0 ;; mistralrs*) echo $G/gpu-mistralrs-t4-2b-q4_0 ;;
  pytorch*) echo $G/gpu-pytorch-t4-2b-q4w4a16 ;; esac; }
step() {  # job arm run-name
  local out; out=$(rigdir $2)/benchmarks/runs/$3/$2
  if grep -q '"rc": 0' "$out/status.json" 2>/dev/null; then say "skip $1 $2 (done)"; return; fi
  say "START $1 $2 -> $out"
  python3 t4chain.py "$1" "$2" "$out" > /dev/null 2>&1
  say "END $1 $2 rc=$?"
}
[ -n "$(awk 'NR>1' /proc/swaps)" ] || sudo swapon /opt1/swapfile

if [ -x /opt1/mistralrs-v0.9.5/bin/mistralrs ]; then
  if ! grep -q '"arm": "mistralrs' results.jsonl; then
    say "START engines mistralrs"
    python3 engines.py . mistralrs,mistralrs-d1,mistralrs-d2,mistralrs-d3 > run-mistralrs.out 2>&1
    say "END engines mistralrs rc=$?"
  fi
else
  say "no mistral.rs binary; its arms are skipped"
fi

for a in vllm vllm-d1 vllm-d2 vllm-d3 vllm-d4 llama llama-d1 llama-d2 llama-d3 llama-d4 \
         sglang sglang-d1 sglang-d2 sglang-d3 sglang-d4; do
  step sweep $a 2026-11-06-concurrency-t4
done
[ -x /opt1/mistralrs-v0.9.5/bin/mistralrs ] && step sweep mistralrs 2026-11-06-concurrency-t4

for a in vllm llama pytorch vllm-d3 llama-d1 sglang sglang-d4; do
  step accuracy $a 2026-11-06-accuracy-t4
done
[ -x /opt1/mistralrs-v0.9.5/bin/mistralrs ] && step accuracy mistralrs 2026-11-06-accuracy-t4
L=$G/gpu-litertlm-t4-2b-int248/benchmarks/runs/2026-11-06-accuracy-t4
if ! grep -q DONE $L/driver.log 2>/dev/null; then say "START accuracy litert"; bash $L/run.sh; say "END accuracy litert rc=$?"; fi
say DONE
