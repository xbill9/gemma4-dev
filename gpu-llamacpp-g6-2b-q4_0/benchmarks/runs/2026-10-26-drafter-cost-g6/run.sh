#!/bin/bash
# Why the card sets the draft length, and whether the drafter pays with several requests at once, on the L4.
# 1. verify/: llama-bench, one batch of k = 1..6 tokens through the target (what checking k-1 drafted tokens
#    costs), at 0 / 8,192 / 32,768 tokens held, E2B and E4B, the rigs' flags, 10 repeats.
# 2. mixed/: the 3050's concurrency-mixed run (mixed.py unchanged): six chat prompts, 256 tokens, prompt cache
#    off, c = 1..64, 5 repeats, greedy, one shared KV cache (-kvu) at -c 32768 --parallel 64; no drafter, 2 and
#    3 drafted tokens. Run from /opt/bench-gap on the instance.
set -u
cd "$(dirname "$0")"; HERE=$PWD
PY=$(cat /opt/torch-g6-q4/PYTHON_BIN)
BIN=/opt/llamacpp/llama.cpp/build/bin
state() { echo "$(date +%T) $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,memory.used --format=csv,noheader) $*" >> driver.log; }
cool() { for i in $(seq 18); do [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 45 ] && break; sleep 5; done; }
envget() { grep "^$2=" $1 | cut -d= -f2- | tr -d '"'; }
state BEGIN
pkill -x llama-server; sleep 2
mkdir -p verify mixed
for spec in "e2b /opt/bench-ll/tpu.env" "e4b /opt/bench-e4b/tpu.env"; do
  set -- $spec; m=$1; E=$2
  M=$(envget $E MODEL_PATH); DR=$(envget $E SPEC_DRAFT_MODEL)
  sha256sum $M $DR >> models.sha256
  cool; state "START verify $m"
  $BIN/llama-bench -m $M -ngl 99 -fa 1 -t 4 -ot 'per_layer_token_embd\.weight=CUDA0' -p 1,2,3,4,5,6 -n 0 \
    -d 0,8192,32768 -r 10 -o jsonl > verify/$m.jsonl 2> verify/$m.log
  state "END verify $m rc=$?"
  name=$(envget $E MODEL_NAME)
  for arm in none q4x-n2 q4x-n3; do
    out=mixed/$m/$arm; mkdir -p $out
    extra=""; [ $arm != none ] && extra="-md $DR --spec-type draft-mtp -ngld all --spec-draft-n-max ${arm#q4x-n}"
    pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done
    cool
    setsid $BIN/llama-server -m $M --host 127.0.0.1 --port 8080 -ngl 99 -c 32768 --parallel 64 -kvu -ctk f16 -ctv f16 \
      -fa 1 -t 4 -tb 8 --reasoning off --metrics -ot 'per_layer_token_embd\.weight=CUDA0' $extra \
      > $out/llama-server.log 2>&1 < /dev/null &
    for i in $(seq 180); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
    state "START mixed $m $arm"
    $PY mixed.py --model $name --out $out/results.jsonl --levels 1,2,4,8,16,32,64 --mode greedy > $out/greedy.log 2>&1
    state "END mixed $m $arm rc=$?"
    curl -fsS http://127.0.0.1:8080/metrics > $out/metrics.prom
  done
done
pkill -x llama-server
state DONE
touch DONE
