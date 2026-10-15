#!/bin/bash
# Overnight queue on the 1650 Ti, all on llama.cpp 77c4540ce (the 3050's build). Waits for
# 2026-10-14-drafter-accuracy-1650ti/run.sh to finish, then:
#  1. E4B accuracy with the drafter at 1 token, the faster setting on this build (that run's speed
#     part), written beside its other arms in ../2026-10-14-drafter-accuracy-1650ti/accuracy/.
#  2. speed.py: Google's non-QAT drafter (Q8_0) against the exact Q4_0 QAT drafter, 1/2/3 tokens.
#  3. longctx.py: drafting with 0 to 120k tokens of text ahead of the question, at the largest
#     context that loads with the bigger (non-QAT Q8_0) drafter, from 131072 down.
# Inputs copied from the 3050 host and hash-checked against its records (sha256.log).
set -u
cd "$(dirname "$0")"
HERE=$PWD
ROOT=$HOME/gemma4-dev
J=$ROOT/jev-tpu-v5e1
M=$HOME/models
ACC=$ROOT/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-accuracy-1650ti/accuracy
G3=$ROOT/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-3050oem
export LLAMA_SERVER_BIN=$HOME/llama.cpp-77c4540ce/build/bin/llama-server
export DOC_PATH=$M/longctx-doc/NVIDIA_GLX-README.txt
LOG=$HERE/driver.log
state() { echo "$(date '+%F %T') gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" >> "$LOG"; }
stop() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }

until grep -q DONE "$ACC/driver.log" 2>/dev/null; do sleep 60; done
state BEGIN
$LLAMA_SERVER_BIN --version 2>&1 | tail -2 > host.txt
{ grep -E '^E[24]B rc=0 ' $G3/convert.txt | awk '{print $4"  "$5}'
  awk '{print $1"  '"$DOC_PATH"'"}' $G3/doc.sha256; } | sha256sum -c - > sha256.log 2>&1
state "inputs: $(tr '\n' ' ' < sha256.log)"

# 1. E4B accuracy, drafter at 1 token (flags as in the accuracy run).
stop
setsid $LLAMA_SERVER_BIN -m $M/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-q4_0-exact.gguf --host 127.0.0.1 --port 8099 \
  -ngl 99 -c 32768 --parallel 8 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 --reasoning off --metrics \
  -md $M/gemma-4-E4B-it-qat-assistant/gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf --spec-type draft-mtp -ngld all \
  --spec-draft-n-max 1 > "$ACC/server-e4b-q4x-n1.log" 2>&1 < /dev/null &
for i in $(seq 180); do curl -fsS http://127.0.0.1:8099/health >/dev/null 2>&1 && break; sleep 1; done
curl -fsS http://127.0.0.1:8099/metrics > "$ACC/metrics-before-e4b-q4x-n1.prom"
for t in gsm8k bfcl_simple; do
  state "START $t e4b-q4x-n1"
  (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8099 google/gemma-4-E4B-it $t "$ACC/$t-e4b-q4x-n1.jsonl" --concurrency 8) > "$ACC/$t-e4b-q4x-n1.log" 2>&1
  state "END $t e4b-q4x-n1 rc=$? $(tail -1 $ACC/$t-e4b-q4x-n1.log)"
done
curl -fsS http://127.0.0.1:8099/metrics > "$ACC/metrics-after-e4b-q4x-n1.prom"
stop

# 2. and 3., per model.
fits() {  # rig ctx: does the server load with the non-QAT drafter at this context?
  python3 - "$1" "$2" <<'EOF'
import sys
from common import load_env, drafts, start, stop
env = load_env(sys.argv[1]); D = drafts(env)
try:
    srv, log = start(env, int(sys.argv[2]), D["nq"], "2", "/dev/null")
except Exception:
    sys.exit(1)
stop(srv, log)
EOF
}
for m in e2b e4b; do
  case $m in e2b) rig=$ROOT/local-llamacpp-1650ti-2b-q4_0;; e4b) rig=$ROOT/local-llamacpp-1650ti-4b-q4_0;; esac
  mkdir -p $m/speed $m/longctx
  state "START speed $m"; python3 speed.py $rig $m/speed > $m/speed/run.log 2>&1; state "END speed $m rc=$?"
  python3 summarize_speed.py $m/speed/results.jsonl > $m/speed/summary.txt 2>&1
  ctx=""
  for c in 131072 98304 69632 49152 34816; do
    stop
    if fits $rig $c; then ctx=$c; break; fi
    state "longctx $m: -c $c does not load"
  done
  stop
  if [ -z "$ctx" ]; then state "longctx $m: nothing fits, skipped"; continue; fi
  state "START longctx $m -c $ctx"; CTX_SIZE=$ctx python3 longctx.py $rig $m/longctx > $m/longctx/run.log 2>&1
  state "END longctx $m rc=$?"
  python3 summarize_longctx.py $m/longctx/results.jsonl > $m/longctx/summary.txt 2>&1
done
stop
state DONE
