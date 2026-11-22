#!/bin/bash
# Google's sampling (generation_config.json of all three: temperature 1.0, top_k 64, top_p 0.95) on E2B, E4B
# and the 12B, thinking on and off, so the 12B's sampled result (../2026-11-21-drafter-sampling-12b-3050oem:
# AIME 21/30 thinking on) has its E2B/E4B and thinking-off counterparts. Greedy cells for every model are in
# the waves (E2B 2026-11-16, E4B 2026-11-17, 12B 2026-11-17). Exact Q4_0 repacks, no drafter, one request
# at a time, 32,768-token limit; serving flags as each wave. Resumable; commits and pushes each cell.
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
# model task thinking limit seed
CELLS=(
  "e4b aime2025 1 0 1" "e2b aime2025 1 0 1" "12b aime2025 0 0 1" "e4b aime2025 0 0 1" "e2b aime2025 0 0 1"
  "e4b gsm8k 1 157 1" "e2b gsm8k 1 157 1" "12b gsm8k 0 157 1" "12b aime2025 1 0 2"
)
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
state "BEGIN llama.cpp $($BIN --version 2>&1 | grep -o 'commit [0-9a-f]*')"
for c in "${CELLS[@]}"; do
  set -- $c; m=$1 t=$2 th=$3 lim=$4 seed=$5
  name=$t-$m-sampled-seed$seed-$([ $th = 1 ] && echo thinkon || echo thinkoff); out=$HERE/$name
  want=$([ $lim -gt 0 ] && echo $lim || wc -l < $G/data/$t.jsonl)
  [ -s $out.jsonl ] && [ "$(wc -l < $out.jsonl)" -eq $want ] && { state "SKIP $name: done"; continue; }
  serve $m || continue
  S="{\"temperature\": 1.0, \"top_k\": 64, \"top_p\": 0.95, \"seed\": $seed}"
  L=(); [ $lim -gt 0 ] && L=(--limit $lim)
  state "START $name ($([ -f $out.jsonl.partial ] && wc -l < $out.jsonl.partial || echo 0) saved)"
  (cd $G && GEN_EVAL_RESUME=1 GEN_EVAL_ENABLE_THINKING=$th GEN_EVAL_SAMPLING="$S" GEN_EVAL_MAX_TOKENS=32768 \
     GEN_EVAL_HARD_MAX_TOKENS=32768 GEN_EVAL_TIMEOUT_S=3600 \
     python3 gen_eval.py run http://127.0.0.1:8080 ${NAME[$m]} $t $out.jsonl --concurrency 1 "${L[@]}") >> $out.log 2>&1
  state "END $name rc=$? $(tail -1 $out.log | cut -c1-160)"
  commit_run "3050oem sampling E2B/E4B/12B: $name"
done
stop8080; state "DONE"; commit_run "3050oem sampling E2B/E4B/12B: done"
