#!/bin/bash
# 12B follow-up to wave 3 (../2026-11-17-thinking-12b-3050oem), under a day on the RTX 3050 OEM:
#   why thinking on falls apart: the same first 157 GSM8K problems (and AIME) with Google's sampling
#   (generation_config.json: temperature 1.0, top_k 64, top_p 0.95; seed 1), and
#   the exact Q4_0 drafter at 2 tokens with greedy decoding, as waves 1-2.
# Exact Q4_0 repack, q4_0 cache, thinking per request. Resumable; commits and pushes each cell.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=/home/xbill/gemma4-dev; REL=${HERE#$ROOT/}
G=$ROOT/benchmarks/gen-eval; BIN=/home/xbill/llama.cpp/build/bin/llama-server; DRIVER=$HERE/driver.log
M=/home/xbill/models/gemma-4-12B-it-qat-q4_0-exact/gemma-4-12B-it-q4_0-exact.gguf
DRAFT=/home/xbill/models/gemma-4-12B-it-qat-assistant/gemma-4-12B-it-qat-assistant-q4_0-exact.gguf
GOOGLE='{"temperature": 1.0, "top_k": 64, "top_p": 0.95, "seed": 1}'
# name task thinking limit sampling drafter
CELLS=(
  "gsm8k-sampled-none-thinkon gsm8k 1 157 google -"
  "gsm8k-greedy-q4x-thinkoff gsm8k 0 0 greedy q4x"
  "aime2025-greedy-q4x-thinkoff aime2025 0 0 greedy q4x"
  "gsm8k-greedy-q4x-thinkon gsm8k 1 50 greedy q4x"
  "aime2025-sampled-none-thinkon aime2025 1 0 google -"
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
serve() {  # serve none|q4x
  [ "$CUR" = "$1" ] && return 0
  stop8080; extra=()
  [ "$1" = q4x ] && extra=(-md "$DRAFT" --spec-type draft-mtp -ngld all --spec-draft-n-max 2)
  setsid $BIN -m "$M" --host 127.0.0.1 --port 8080 -ngl 99 -c 81920 --parallel 1 -ctk q4_0 -ctv q4_0 \
    -fa 1 -t 6 -tb 12 --reasoning off --metrics "${extra[@]}" > $HERE/server-$1.log 2>&1 < /dev/null &
  for i in $(seq 600); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  curl -fsS http://127.0.0.1:8080/health >/dev/null || { state "server $1 did not load"; return 1; }
  CUR=$1; state "server up drafter=$1 -c 81920 -ctk q4_0: $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | tr '\n' ' ')"
}
state "BEGIN llama.cpp $($BIN --version 2>&1 | grep -o 'commit [0-9a-f]*')"
for c in "${CELLS[@]}"; do
  set -- $c; name=$1 t=$2 th=$3 lim=$4 samp=$5 d=$6; out=$HERE/$name
  want=$([ $lim -gt 0 ] && echo $lim || wc -l < $G/data/$t.jsonl)
  [ -s $out.jsonl ] && [ "$(wc -l < $out.jsonl)" -eq $want ] && { state "SKIP $name: done"; continue; }
  serve $d || continue
  S=""; [ $samp = google ] && S=$GOOGLE
  L=(); [ $lim -gt 0 ] && L=(--limit $lim)
  state "START $name ($([ -f $out.jsonl.partial ] && wc -l < $out.jsonl.partial || echo 0) saved)"
  (cd $G && GEN_EVAL_RESUME=1 GEN_EVAL_ENABLE_THINKING=$th GEN_EVAL_SAMPLING="$S" GEN_EVAL_MAX_TOKENS=32768 \
     GEN_EVAL_HARD_MAX_TOKENS=32768 GEN_EVAL_TIMEOUT_S=3600 \
     python3 gen_eval.py run http://127.0.0.1:8080 google/gemma-4-12B-it $t $out.jsonl --concurrency 1 "${L[@]}") >> $out.log 2>&1
  state "END $name rc=$? $(tail -1 $out.log | cut -c1-120)"
  commit_run "3050oem 12B drafter/sampling: $name"
done
stop8080; state "DONE"; commit_run "3050oem 12B drafter/sampling: done"
