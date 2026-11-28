#!/bin/bash
# Weekend follow-on (2026-11-27 to 11-30): the 3050's three deeper depths on this card, most useful arm first.
# Waits for run.sh (16k/32k) to finish, then appends to the same longctx.jsonl. Resumable; commits and pushes each arm.
set -u
export LONGCTX_DEPTHS="[64000, 98304, 126976]" LONGCTX_NDRAFT=1 LONGCTX_LOGTAG=-deep
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=/home/xbill/gemma4-dev; REL=${HERE#$ROOT/}
state() { echo "$(date '+%F %T') $(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a $HERE/driver.log; }
commit_run() {
  ( cd $ROOT && git add "$REL" && \
    git -c user.name=xbill -c user.email=xbill9@gmail.com commit -q -m "$1" \
      -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$REL" && \
    { git push -q 2>/dev/null || { git pull --no-rebase -q --no-edit && git push -q; }; } ) \
    && state "PUSHED: $(git -C $ROOT log --oneline -1 | cut -c1-80)" || state "COMMIT/PUSH FAILED (left committed locally)"
}
while pgrep -f '/run\.sh$' > /dev/null; do sleep 60; done
state "BEGIN deep llama.cpp $(/home/xbill/llama.cpp/build/bin/llama-server --version 2>&1 | grep -o 'commit [0-9a-f]*')"
for arm in f16-sampled-on f16-sampled-off f16-draft f16-none; do
  state "START deep $arm"
  python3 $HERE/longctx.py $HERE $arm >> $HERE/longctx.out 2>&1
  state "END deep $arm rc=$? $(grep -c "\"arm\": \"$arm\"" $HERE/longctx.jsonl 2>/dev/null) trials (all depths)"
  commit_run "1650ti E2B long context, deep: $arm"
done
pkill -x llama-server; state "DONE deep"; commit_run "1650ti E2B long context, deep: done"
