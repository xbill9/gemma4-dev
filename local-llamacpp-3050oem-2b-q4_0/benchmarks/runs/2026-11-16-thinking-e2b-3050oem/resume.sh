#!/bin/bash
# Pick wave 1 up after the machine or the run dropped. Same cells, order and settings as run.sh.
# A cell is done when <cell>.jsonl holds every problem; it is skipped. Any other cell is re-run with
# GEN_EVAL_RESUME=1, which keeps the replies already in <cell>.jsonl.partial and asks only the rest.
# An arm whose four cells are done never starts its server. Run on the gpu host, from this directory:
#   setsid nohup ./resume.sh > nohup-resume.out 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"; HERE=$PWD
. ../2026-10-14-overnight-3050oem/lib.sh
DRIVER=$HERE/driver.log
G=$ROOT/benchmarks/gen-eval
REL=local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem
GOOG=$M/gemma-4-E2B-it-qat-q4_0-google/gemma-4-E2B_q4_0-it.gguf
CELLS=("gsm8k off" "aime2025 off" "aime2025 on" "gsm8k on")

# Refuse to start beside a live run.sh or resume.sh in this directory.
for p in $(pgrep -f '^(/usr)?/bin/bash (./)?(run|resume)\.sh$'); do
  [ "$p" = $$ ] && continue; [ "$(ps -o ppid= -p $p 2>/dev/null | tr -d " ")" = "$$" ] && continue
  [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$HERE" ] && { echo "another run is live (pid $p); not starting"; exit 1; }
done

commit_run() {
  ( cd $ROOT && git add "$REL" && \
    git -c user.name=xbill -c user.email=xbill9@gmail.com commit -q -m "$1" \
      -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$REL" && \
    git pull --rebase -q && git push -q ) \
    && state "PUSHED: $(git -C $ROOT log --oneline -1 | cut -c1-80)" || state "COMMIT/PUSH FAILED (left committed locally)"
}
want() { wc -l < "$G/data/$1.jsonl"; }
cell_done() { [ -s "$1.jsonl" ] && [ "$(wc -l < "$1.jsonl")" -eq "$(want $2)" ]; }

state "RESUME llama.cpp $($BIN --version 2>&1 | grep -o 'commit [0-9a-f]*')"
for spec in "repack-none $E2B -" "repack-q4x $E2B $D2" "google-none $GOOG -" "google-q4x $GOOG $D2"; do
  set -- $spec; arm=$1 gguf=$2 d=$3; [ "$d" = - ] && d=""
  left=0
  for cell in "${CELLS[@]}"; do set -- $cell; cell_done $HERE/$1-$arm-think$2 $1 || left=$((left+1)); done
  [ $left -eq 0 ] && { state "SKIP $arm: all cells done"; continue; }
  [ -s "$gguf" ] || { state "SKIP $arm: $gguf missing"; continue; }
  [ "$gguf" = "$GOOG" ] && { grep -q '^google ' weights.sha256 || echo "google $(sha256sum $GOOG | cut -c1-64)" >> weights.sha256; }
  serve $HERE/server-$arm.resume.log 131072 1 $gguf "$d" 2 || { state "SKIP $arm: server did not start"; continue; }
  for cell in "${CELLS[@]}"; do
    set -- $cell; t=$1 th=$2; th01=$([ $th = on ] && echo 1 || echo 0)
    out=$HERE/$t-$arm-think$th
    cell_done $out $t && { state "SKIP $t $arm thinking=$th: done"; continue; }
    have=$([ -f $out.jsonl.partial ] && wc -l < $out.jsonl.partial || echo 0)
    state "START $t $arm thinking=$th (resume, $have saved)"
    (cd $G && GEN_EVAL_RESUME=1 GEN_EVAL_ENABLE_THINKING=$th01 GEN_EVAL_MAX_TOKENS=32768 GEN_EVAL_HARD_MAX_TOKENS=32768 \
       GEN_EVAL_TIMEOUT_S=1800 python3 gen_eval.py run http://127.0.0.1:8080 google/gemma-4-E2B-it $t $out.jsonl --concurrency 1) >> $out.log 2>&1
    state "END $t $arm thinking=$th rc=$? $(tail -1 $out.log | cut -c1-120)"
  done
  curl -fsS http://127.0.0.1:8080/metrics > $HERE/metrics-after-$arm.resume.prom
  commit_run "3050oem E2B thinking wave 1 (resumed): arm $arm"
done
stop8080
state "DONE (resume)"
commit_run "3050oem E2B thinking wave 1: done (resumed)"
