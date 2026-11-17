#!/bin/bash
# Local thinking study, one model per wave, on the RTX 3050 OEM. Same design as wave 1
# (local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem):
#   weights:  repack (xbill9 exact Q4_0) | google (Google's QAT Q4_0 GGUF)
#   drafter:  none | q4x (exact Q4_0 QAT assistant, 2 drafted tokens)
#   thinking: off | on (chat_template_kwargs enable_thinking)
#   tasks:    gsm8k (1,319), aime2025 (30)
# Model, files, cache type and context are in wave.env beside this script. One request at a time,
# temperature 0, 32,768-token output limit. Resumable: a finished cell is skipped, an unfinished one
# keeps its saved replies (GEN_EVAL_RESUME=1). Every cell is committed and pushed when it ends.
# Run on the gpu host from this directory:  setsid nohup ./run.sh > nohup.out 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"; HERE=$PWD
. ./wave.env
ROOT=/home/xbill/gemma4-dev
BIN=/home/xbill/llama.cpp/build/bin/llama-server
G=$ROOT/benchmarks/gen-eval
REL=${HERE#$ROOT/}
DRIVER=$HERE/driver.log
CELLS=("gsm8k off" "aime2025 off" "aime2025 on" "gsm8k on")

for p in $(pgrep -f '^(/usr)?/bin/bash (\./)?run\.sh$'); do
  [ "$p" = $$ ] && continue; [ "$(ps -o ppid= -p $p 2>/dev/null | tr -d " ")" = "$$" ] && continue
  [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$HERE" ] && { echo "another run is live (pid $p); not starting"; exit 1; }
done

state() { echo "$(date '+%F %T') $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$DRIVER"; }
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 45 ]; do sleep 5; done; }
# Commit only this directory. No autostash: a stash can swap out files a run is appending to.
commit_run() {
  ( cd $ROOT && git add "$REL" && \
    git -c user.name=xbill -c user.email=xbill9@gmail.com commit -q -m "$1" \
      -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$REL" && \
    { git push -q 2>/dev/null || { git pull --no-rebase -q --no-edit && git push -q; }; } ) \
    && state "PUSHED: $(git -C $ROOT log --oneline -1 | cut -c1-80)" || state "COMMIT/PUSH FAILED (left committed locally)"
}
# serve LOG GGUF DRAFTER: tries each context in CTX_TRY until one loads; records which.
serve() {
  local log=$1 m=$2 d=$3 ctx extra
  for ctx in $CTX_TRY; do
    stop8080; cool
    extra=()
    [ -n "$OT" ] && extra+=(-ot "$OT")
    [ -n "$d" ] && extra+=(-md "$d" --spec-type draft-mtp -ngld all --spec-draft-n-max 2)
    setsid $BIN -m "$m" --host 127.0.0.1 --port 8080 -ngl 99 -c $ctx --parallel 1 -ctk $CTK -ctv $CTK \
      -fa 1 -t 6 -tb 12 --reasoning off --metrics "${extra[@]}" > "$log" 2>&1 < /dev/null &
    local pid=$! ok=0
    for i in $(seq 600); do
      curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && { ok=1; break; }
      kill -0 $pid 2>/dev/null || break
      sleep 1
    done
    if [ $ok = 1 ]; then
      state "server up $(basename $log) -c $ctx -ctk $CTK: $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | tr '\n' ' ')"
      return 0
    fi
    state "server did not load at -c $ctx ($(basename $log)): $(grep -m1 -iE 'out of memory|failed|error' $log | cut -c1-120)"
  done
  return 1
}
want() { wc -l < "$G/data/$1.jsonl"; }
cell_done() { [ -s "$1.jsonl" ] && [ "$(wc -l < "$1.jsonl")" -eq "$(want $2)" ]; }

state "BEGIN $LABEL llama.cpp $($BIN --version 2>&1 | grep -o 'commit [0-9a-f]*')"
for f in "$REPACK" "$GOOG" "$DRAFT"; do
  [ -s "$f" ] || { state "ABORT: $f missing"; exit 1; }
  grep -q " $f\$" weights.sha256 2>/dev/null || echo "$(sha256sum "$f" | cut -c1-64) $f" >> weights.sha256
done
for spec in "repack-none $REPACK -" "repack-q4x $REPACK $DRAFT" "google-none $GOOG -" "google-q4x $GOOG $DRAFT"; do
  set -- $spec; arm=$1 gguf=$2 d=$3; [ "$d" = - ] && d=""
  left=0
  for cell in "${CELLS[@]}"; do set -- $cell; cell_done $HERE/$1-$arm-think$2 $1 || left=$((left+1)); done
  [ $left -eq 0 ] && { state "SKIP $arm: all cells done"; continue; }
  serve $HERE/server-$arm.log $gguf "$d" || { state "SKIP $arm: server did not start"; continue; }
  for cell in "${CELLS[@]}"; do
    set -- $cell; t=$1 th=$2; th01=$([ $th = on ] && echo 1 || echo 0)
    out=$HERE/$t-$arm-think$th
    cell_done $out $t && { state "SKIP $t $arm thinking=$th: done"; continue; }
    have=$([ -f $out.jsonl.partial ] && wc -l < $out.jsonl.partial || echo 0)
    state "START $t $arm thinking=$th ($have saved)"
    (cd $G && GEN_EVAL_RESUME=1 GEN_EVAL_ENABLE_THINKING=$th01 GEN_EVAL_MAX_TOKENS=32768 GEN_EVAL_HARD_MAX_TOKENS=32768 \
       GEN_EVAL_TIMEOUT_S=3600 python3 gen_eval.py run http://127.0.0.1:8080 $SERVED $t $out.jsonl --concurrency 1) >> $out.log 2>&1
    state "END $t $arm thinking=$th rc=$? $(tail -1 $out.log | cut -c1-120)"
    commit_run "3050oem $LABEL thinking: $t $arm thinking $th"
  done
  curl -fsS http://127.0.0.1:8080/metrics > $HERE/metrics-after-$arm.prom
done
stop8080
state "DONE"
commit_run "3050oem $LABEL thinking: done"
