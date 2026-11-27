#!/bin/bash
# Long-context study, Gemma 4 E2B on the GTX 1650 Ti, the 3050 run at 16k and 32k (see longctx.py). Resumable; commits and pushes each arm.
set -u
export LONGCTX_DEPTHS="[16384, 32768]" LONGCTX_NDRAFT=1
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=/home/xbill/gemma4-dev; REL=${HERE#$ROOT/}
state() { echo "$(date '+%F %T') $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a $HERE/driver.log; }
commit_run() {
  ( cd $ROOT && git add "$REL" && \
    git -c user.name=xbill -c user.email=xbill9@gmail.com commit -q -m "$1" \
      -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$REL" && \
    { git push -q 2>/dev/null || { git pull --no-rebase -q --no-edit && git push -q; }; } ) \
    && state "PUSHED: $(git -C $ROOT log --oneline -1 | cut -c1-80)" || state "COMMIT/PUSH FAILED (left committed locally)"
}
state "BEGIN llama.cpp $(/home/xbill/llama.cpp/build/bin/llama-server --version 2>&1 | grep -o 'commit [0-9a-f]*')"
for arm in f16-draft q8-draft q4-draft f16-sampled-off f16-sampled-on f16-none; do
  state "START $arm"
  python3 $HERE/longctx.py $HERE $arm >> $HERE/longctx.out 2>&1
  state "END $arm rc=$? $(grep -c "\"arm\": \"$arm\"" $HERE/longctx.jsonl 2>/dev/null) trials"
  commit_run "1650ti E2B long context: $arm"
done
pkill -x llama-server; state "DONE"; commit_run "1650ti E2B long context: done"
