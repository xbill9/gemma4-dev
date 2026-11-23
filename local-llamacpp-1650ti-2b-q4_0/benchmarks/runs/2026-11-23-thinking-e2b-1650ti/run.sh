#!/bin/bash
# The 3050's E2B thinking run (local-llamacpp-3050oem-2b-q4_0/.../2026-11-16-thinking-e2b-3050oem) on the
# GTX 1650 Ti Max-Q (4 GB), same llama.cpp commit (7182ef7), weights, drafter, cells, order and settings:
#   weights:  repack = xbill9 exact Q4_0 (the rig's file)   google = google/gemma-4-E2B-it-qat-q4_0-gguf @675cff4
#   drafter:  none | q4x = exact Q4_0 E2B QAT assistant, 1 drafted token (this rig's live setting; the 3050 used 2)
#   thinking: off | on   (chat_template_kwargs enable_thinking)
#   tasks:    gsm8k (1,319) and aime2025 (30, I and II)
# Server: -c 131072 --parallel 1, f16 cache, one request at a time, temperature 0, prompt cache at the server
# default, 32,768-token output limit. The per-layer table stays in system RAM (this rig's setting; the 3050
# put it on the card). Resumable: a cell whose .jsonl holds every problem is skipped, any other cell resumes
# from its .jsonl.partial. Each arm is committed and pushed when it finishes.
#   setsid nohup ./run.sh > nohup.out 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"; HERE=$PWD
ROOT=/home/xbill/gemma4-dev
BIN=/home/xbill/llama.cpp/build/bin/llama-server
M=/home/xbill/models
E2B=$M/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
GOOG=$M/gemma-4-E2B-it-qat-q4_0-google/gemma-4-E2B_q4_0-it.gguf
D2=$M/gemma-4-E2B-it-qat-assistant/gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf
NDRAFT=1
DRIVER=$HERE/driver.log
G=$ROOT/benchmarks/gen-eval
REL=local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-11-23-thinking-e2b-1650ti
CELLS=("gsm8k off" "aime2025 off" "aime2025 on" "gsm8k on")

state() { echo "$(date '+%F %T') $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$DRIVER"; }
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | cut -d. -f1)" -le 50 ]; do sleep 5; done; }  # this driver prints 41.06
serve() {  # LOG GGUF [DRAFTER]
  local log=$1 m=$2 d=${3:-}
  stop8080; cool
  local extra=()
  [ -n "$d" ] && extra=(-md "$d" --spec-type draft-mtp -ngld all --spec-draft-n-max $NDRAFT)
  setsid $BIN -m "$m" --host 127.0.0.1 --port 8080 -ngl 99 -c 131072 --parallel 1 -ctk f16 -ctv f16 \
    -fa 1 -t 6 -tb 12 --reasoning off --metrics "${extra[@]}" > "$log" 2>&1 < /dev/null &
  for i in $(seq 300); do
    curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && { state "server up $(basename $log): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; return 0; }
    pgrep -x llama-server >/dev/null || { state "server FAILED $(basename $log)"; return 1; }
    sleep 1
  done
  state "server TIMEOUT $(basename $log)"; return 1
}
# Commit only this run directory. No autostash: a stash can swap out files a run is appending to.
commit_run() {
  ( cd $ROOT && git add "$REL" && \
    git commit -q -m "$1" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$REL" && \
    git pull --rebase -q && git push -q ) \
    && state "PUSHED: $(git -C $ROOT log --oneline -1 | cut -c1-80)" || state "COMMIT/PUSH FAILED (left committed locally)"
}
want() { wc -l < "$G/data/$1.jsonl"; }
cell_done() { [ -s "$1.jsonl" ] && [ "$(wc -l < "$1.jsonl")" -eq "$(want $2)" ]; }

for p in $(pgrep -f '^(/usr)?/bin/bash (./)?run\.sh$'); do
  [ "$p" = $$ ] && continue; [ "$(ps -o ppid= -p $p 2>/dev/null | tr -d " ")" = "$$" ] && continue
  [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$HERE" ] && { echo "another run is live (pid $p); not starting"; exit 1; }
done

state "BEGIN llama.cpp $($BIN --version 2>&1 | grep -o 'commit [0-9a-f]*') $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
[ -s weights.sha256 ] || { for f in "repack $E2B" "google $GOOG" "drafter $D2"; do set -- $f; echo "$1 $(sha256sum $2 | cut -c1-64)"; done > weights.sha256; }
for spec in "repack-none $E2B -" "repack-q4x $E2B $D2" "google-none $GOOG -" "google-q4x $GOOG $D2"; do
  set -- $spec; arm=$1 gguf=$2 d=$3; [ "$d" = - ] && d=""
  left=0
  for cell in "${CELLS[@]}"; do set -- $cell; cell_done $HERE/$1-$arm-think$2 $1 || left=$((left+1)); done
  [ $left -eq 0 ] && { state "SKIP $arm: all cells done"; continue; }
  n=$(ls $HERE/server-$arm*.log 2>/dev/null | wc -l)
  serve $HERE/server-$arm$([ $n -gt 0 ] && echo .$n).log $gguf "$d" || { state "SKIP $arm: server did not start"; continue; }
  for cell in "${CELLS[@]}"; do
    set -- $cell; t=$1 th=$2; th01=$([ $th = on ] && echo 1 || echo 0)
    out=$HERE/$t-$arm-think$th
    cell_done $out $t && { state "SKIP $t $arm thinking=$th: done"; continue; }
    have=$([ -f $out.jsonl.partial ] && wc -l < $out.jsonl.partial || echo 0)
    state "START $t $arm thinking=$th ($have saved)"
    (cd $G && GEN_EVAL_RESUME=1 GEN_EVAL_ENABLE_THINKING=$th01 GEN_EVAL_MAX_TOKENS=32768 GEN_EVAL_HARD_MAX_TOKENS=32768 \
       GEN_EVAL_TIMEOUT_S=1800 python3 gen_eval.py run http://127.0.0.1:8080 google/gemma-4-E2B-it $t $out.jsonl --concurrency 1) >> $out.log 2>&1
    state "END $t $arm thinking=$th rc=$? $(tail -1 $out.log | cut -c1-120)"
  done
  curl -fsS http://127.0.0.1:8080/metrics > $HERE/metrics-after-$arm.prom
  commit_run "1650ti E2B thinking: arm $arm (gsm8k, aime2025; thinking off and on)"
done
stop8080
state "DONE"
python3 analyze.py > /dev/null 2>&1 && state "analyze.py wrote RESULTS.md"
commit_run "1650ti E2B thinking: done"
