#!/bin/bash
# Function calling on the 3050: BFCL v3 simple (400), multiple (200) and irrelevance (240) on E2B, E4B and the
# 12B, greedy and Google's sampling settings (generation_config.json: temperature 1.0, top_k 64, top_p 0.95,
# seed 1), thinking off and on. Exact Q4_0 repacks, no drafter, one request at a time, serving flags as
# ../../../../local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-11-22-sampling-e2b-e4b-12b-3050oem.
# Replies capped at 8,192 tokens. The 12B greedy thinking-on cells run last (greedy thinking looped on the
# 12B in its GSM8K wave). Resumable; commits and pushes each cell.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=/home/xbill/gemma4-dev; REL=${HERE#$ROOT/}
G=$ROOT/benchmarks/gen-eval; BIN=/home/xbill/llama.cpp/build/bin/llama-server; DRIVER=$HERE/driver.log; M=/home/xbill/models
declare -A GGUF=([e2b]=$M/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
                 [e4b]=$M/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-q4_0-exact.gguf
                 [12b]=$M/gemma-4-12B-it-qat-q4_0-exact/gemma-4-12B-it-q4_0-exact.gguf)
declare -A NAME=([e2b]=google/gemma-4-E2B-it [e4b]=google/gemma-4-E4B-it [12b]=google/gemma-4-12B-it)
declare -A FLAGS=([e2b]="-c 131072 -ctk f16 -ctv f16 -ot per_layer_token_embd\.weight=CUDA0"
                  [e4b]="-c 131072 -ctk f16 -ctv f16 -ot per_layer_token_embd\.weight=CUDA0"
                  [12b]="-c 81920 -ctk q4_0 -ctv q4_0")
TASKS="bfcl_simple bfcl_multiple bfcl_irrelevance"
# model mode(greedy|sampled) thinking
CELLS=()
for m in e4b e2b 12b; do for mode in "greedy 0" "sampled 0" "sampled 1" "greedy 1"; do
  [ "$m $mode" = "12b greedy 1" ] && continue; for t in $TASKS; do CELLS+=("$m $t $mode"); done; done; done
for t in $TASKS; do CELLS+=("12b $t greedy 1"); done
state() { echo "$(date '+%F %T') $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$DRIVER"; }
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
commit_run() {
  ( cd $ROOT && git add "$REL" && \
    git -c user.name=xbill -c user.email=xbill9@gmail.com commit -q -m "$1" \
      -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$REL" && \
    { git push -q 2>/dev/null || { git pull --no-rebase -q --no-edit && git push -q; }; } ) \
    && state "PUSHED: $(git -C $ROOT log --oneline -1 | cut -c1-80)" || state "COMMIT/PUSH FAILED (left committed locally)"
}
CUR=""
serve() {
  [ "$CUR" = "$1" ] && return 0
  stop8080
  setsid $BIN -m "${GGUF[$1]}" --host 127.0.0.1 --port 8080 -ngl 99 --parallel 1 ${FLAGS[$1]} \
    -fa 1 -t 6 -tb 12 --reasoning off --metrics > $HERE/server-$1.log 2>&1 < /dev/null &
  for i in $(seq 600); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  curl -fsS http://127.0.0.1:8080/health >/dev/null || { state "server $1 did not load"; return 1; }
  CUR=$1; state "server up $1 ${FLAGS[$1]}: $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | tr '\n' ' ')"
}
state "BEGIN llama.cpp $(git -C /home/xbill/llama.cpp log --oneline -1 | cut -c1-60); ${#CELLS[@]} cells"
for c in "${CELLS[@]}"; do
  set -- $c; m=$1 t=$2 mode=$3 th=$4
  name=$t-$m-$mode-$([ $th = 1 ] && echo thinkon || echo thinkoff); out=$HERE/$name
  want=$(wc -l < $G/data/$t.jsonl)
  [ -s $out.jsonl ] && [ "$(wc -l < $out.jsonl)" -eq $want ] && { state "SKIP $name: done"; continue; }
  serve $m || continue
  S='{"temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": 1}'; [ $mode = greedy ] && S='{}'
  state "START $name ($([ -f $out.jsonl.partial ] && wc -l < $out.jsonl.partial || echo 0) saved)"
  (cd $G && GEN_EVAL_RESUME=1 GEN_EVAL_ENABLE_THINKING=$th GEN_EVAL_SAMPLING="$S" GEN_EVAL_BFCL_MAX_TOKENS=8192 \
     GEN_EVAL_TIMEOUT_S=1800 python3 gen_eval.py run http://127.0.0.1:8080 ${NAME[$m]} $t $out.jsonl --concurrency 1) >> $out.log 2>&1
  state "END $name rc=$? $(tail -1 $out.log | cut -c1-160)"
  commit_run "3050oem functions E2B/E4B/12B: $name"
done
stop8080; state "DONE"; commit_run "3050oem functions E2B/E4B/12B: done"
