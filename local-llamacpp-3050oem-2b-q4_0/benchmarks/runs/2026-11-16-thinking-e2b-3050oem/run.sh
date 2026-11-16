#!/bin/bash
# Wave 1 of the local thinking study: Gemma 4 E2B only, on the RTX 3050 OEM.
#   weights:  repack = xbill9 exact Q4_0 (the rig's file)   google = google/gemma-4-E2B-it-qat-q4_0-gguf @675cff4
#   drafter:  none | q4x = exact Q4_0 E2B QAT assistant, 2 drafted tokens (the rig's live setting)
#   thinking: off | on   (chat_template_kwargs enable_thinking; the template has no levels)
#   tasks:    gsm8k (1,319) and aime2025 (30, I and II)
# Same as the rig serves: -c 131072 --parallel 1, f16 cache, one request at a time, temperature 0,
# prompt cache at the server default. Every cell gets the same 32,768-token output limit.
# Each server arm is committed when it finishes. Run on the gpu host: setsid nohup ./run.sh &
set -u
cd "$(dirname "$0")"; HERE=$PWD
. ../2026-10-14-overnight-3050oem/lib.sh
DRIVER=$HERE/driver.log
G=$ROOT/benchmarks/gen-eval
REL=local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem
GOOG=$M/gemma-4-E2B-it-qat-q4_0-google/gemma-4-E2B_q4_0-it.gguf
REPACK_SHA=419db9a6bf3bc15d770c85ccf9216827aa88c64fe8f3d72fbe5674f48efe2dc8

# Commit only this run directory. No autostash: a stash can swap out files a run is appending to.
commit_run() {
  ( cd $ROOT && git add "$REL" && \
    git -c user.name=xbill -c user.email=xbill9@gmail.com commit -q -m "$1" \
      -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$REL" && \
    git pull --rebase -q && git push -q ) \
    && state "PUSHED: $(git -C $ROOT log --oneline -1 | cut -c1-80)" || state "COMMIT/PUSH FAILED (left committed locally)"
}

state "BEGIN llama.cpp $($BIN --version 2>&1 | grep -o 'commit [0-9a-f]*')"
echo "repack $(sha256sum $E2B | cut -c1-64)" > weights.sha256
[ "$(cut -d' ' -f2 weights.sha256)" = "$REPACK_SHA" ] || state "WARN repack SHA differs from tpu.env"

for spec in "repack-none $E2B -" "repack-q4x $E2B $D2" "google-none GOOG -" "google-q4x GOOG $D2"; do
  set -- $spec; arm=$1 gguf=$2 d=$3; [ "$d" = - ] && d=""
  if [ "$gguf" = GOOG ]; then
    for i in $(seq 720); do [ -s "$GOOG" ] && ! pgrep -f gemma-4-E2B-it-qat-q4_0-gguf >/dev/null && break; sleep 10; done
    [ -s "$GOOG" ] || { state "SKIP $arm: Google GGUF missing"; continue; }
    grep -q '^google ' weights.sha256 || echo "google $(sha256sum $GOOG | cut -c1-64)" >> weights.sha256
    gguf=$GOOG
  fi
  serve $HERE/server-$arm.log 131072 1 $gguf "$d" 2 || { state "SKIP $arm"; continue; }
  for cell in "gsm8k off" "aime2025 off" "aime2025 on" "gsm8k on"; do
    set -- $cell; t=$1 th=$2; th01=$([ $th = on ] && echo 1 || echo 0)
    out=$HERE/$t-$arm-think$th
    state "START $t $arm thinking=$th"
    (cd $G && GEN_EVAL_ENABLE_THINKING=$th01 GEN_EVAL_MAX_TOKENS=32768 GEN_EVAL_HARD_MAX_TOKENS=32768 \
       GEN_EVAL_TIMEOUT_S=1800 python3 gen_eval.py run http://127.0.0.1:8080 google/gemma-4-E2B-it $t $out.jsonl --concurrency 1) > $out.log 2>&1
    state "END $t $arm thinking=$th rc=$? $(tail -1 $out.log | cut -c1-120)"
  done
  curl -fsS http://127.0.0.1:8080/metrics > $HERE/metrics-after-$arm.prom
  commit_run "3050oem E2B thinking wave 1: arm $arm (gsm8k, aime2025; thinking off and on)"
done
stop8080
state "DONE"
commit_run "3050oem E2B thinking wave 1: done"
