#!/bin/bash
# Two drafter gaps on the 1650 Ti, both measured on the 3050, on llama.cpp 77c4540ce. Waits for
# ../2026-10-14-drafter-gaps-1650ti/run.sh to finish. Drafter: the exact Q4_0 one at each rig's
# tpu.env SPEC_DRAFT_N_MAX (E2B 1, E4B 2), i.e. as the rigs serve.
#  A. Concurrency (as 2026-10-14-concurrency-drafter-3050oem): sweep.py, c = 1, 2, 4, 8, 16,
#     5 repeats, unique prompts, 512 in/128 out and 128 in/512 out, -c 16384 --parallel 16
#     (1,024 tokens per slot), drafter off and on. The rig's CLAUDE.md records that c = 2-8 can be
#     bimodal on this card; read the spread column.
#  B. Accuracy as the rigs serve (as 2026-10-14-accuracy-fullctx-3050oem): -c 8192 --parallel 1,
#     gen_eval.py --concurrency 1, prompt cache at the server default; GSM8K (2,048 tokens) and
#     BFCL simple. E2B drafter, E2B none, E4B drafter, E4B none.
set -u
cd "$(dirname "$0")"; HERE=$PWD
ROOT=$HOME/gemma4-dev; J=$ROOT/jev-tpu-v5e1; M=$HOME/models
BIN=$HOME/llama.cpp-77c4540ce/build/bin/llama-server
SWEEP=$ROOT/local-llamacpp-1650ti-2b-q4_0/sweep.py
E2B=$M/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
E4B=$M/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-q4_0-exact.gguf
D2=$M/gemma-4-E2B-it-qat-assistant/gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf
D4=$M/gemma-4-E4B-it-qat-assistant/gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf
n2=$(grep ^SPEC_DRAFT_N_MAX= $ROOT/local-llamacpp-1650ti-2b-q4_0/tpu.env | cut -d= -f2)
n4=$(grep ^SPEC_DRAFT_N_MAX= $ROOT/local-llamacpp-1650ti-4b-q4_0/tpu.env | cut -d= -f2)
LOG=$HERE/driver.log
state() { echo "$(date '+%F %T') gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" >> "$LOG"; }
stop() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
serve() {  # log ctx parallel gguf drafter n
  stop
  local d=(); [ -n "$5" ] && d=(-md "$5" --spec-type draft-mtp -ngld all --spec-draft-n-max "$6")
  setsid $BIN -m "$4" --host 127.0.0.1 --port 8080 -ngl 99 -c "$2" --parallel "$3" -ctk f16 -ctv f16 -fa 1 \
    -t 6 -tb 12 --reasoning off --metrics "${d[@]}" > "$1" 2>&1 < /dev/null &
  for i in $(seq 180); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 || { state "server failed: $*"; return 1; }
  state "server up $(basename $1): $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
}
until grep -q DONE ../2026-10-14-drafter-gaps-1650ti/driver.log 2>/dev/null; do sleep 60; done
state "BEGIN (drafter tokens: E2B $n2, E4B $n4)"
$BIN --version 2>&1 | tail -2 > host.txt

for spec in "e2b $E2B $D2 $n2 xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf" "e4b $E4B $D4 $n4 xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf"; do
  set -- $spec; m=$1 gguf=$2 drafter=$3 n=$4 name=$5
  for arm in none q4x; do
    out=$HERE/conc/$m/$arm; mkdir -p $out
    d=""; [ $arm = q4x ] && d=$drafter
    serve $out/llama-server.log 16384 16 $gguf "$d" $n || continue
    for shape in "512 128 in512-out128" "128 512 in128-out512"; do
      set -- $shape
      state "START A $m $arm $3"
      curl -fsS http://127.0.0.1:8080/metrics > $out/metrics-before-$3.prom
      (cd $ROOT/local-llamacpp-1650ti-2b-q4_0 && python3 $SWEEP --base http://127.0.0.1:8080/v1 --model $name \
        --out $out/$3 --concurrency 1,2,4,8,16 --conc-input $1 --conc-output $2 --repeats 5 --prompt-mode unique \
        --decode-source auto --expect-device gpu) > $out/sweep-$3.log 2>&1
      state "END A $m $arm $3 rc=$?"
      curl -fsS http://127.0.0.1:8080/metrics > $out/metrics-after-$3.prom
    done
  done
done
stop
python3 summarize_conc.py > conc/summary.txt 2>&1

mkdir -p acc
for spec in "e2b-q4x $E2B $D2 $n2 google/gemma-4-E2B-it" "e2b-none $E2B - - google/gemma-4-E2B-it" \
            "e4b-q4x $E4B $D4 $n4 google/gemma-4-E4B-it" "e4b-none $E4B - - google/gemma-4-E4B-it"; do
  set -- $spec; arm=$1 gguf=$2 d=$3 n=$4 name=$5; [ "$d" = - ] && d=""
  serve $HERE/acc/server-$arm.log 8192 1 $gguf "$d" $n || continue
  for t in bfcl_simple gsm8k; do
    state "START B $t $arm"
    (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8080 $name $t $HERE/acc/$t-$arm.jsonl --concurrency 1) > $HERE/acc/$t-$arm.log 2>&1
    state "END B $t $arm rc=$? $(tail -1 $HERE/acc/$t-$arm.log)"
  done
  curl -fsS http://127.0.0.1:8080/metrics > $HERE/acc/metrics-after-$arm.prom
done
stop
state DONE
