#!/bin/bash
# Overnight queue on the T4, 2026-10-15. Waits for the drafter sweep (../2026-10-15-drafter-sweep-t4:
# run.sh, then run_q4ct.sh) to finish, then runs the gap jobs one after another. A failed job is
# logged and the queue moves on. Nothing else may hold the card or port 8000.
#   1. conc.py      drafter at 1/2/4/8 concurrent requests, E2B and E4B
#   2. quality.py   GSM8K + BFCL with and without the serving drafter; Nimble suite without
#                   (build_suite.sh rebuilds the suite beside it, CPU and network only)
#   3. spec_ab.py   drafters against the bf16 E2B target (gpu-vllm-t4-2b), in that rig
#   4. longctx.py   drafter with 0-15,000 tokens of context, E2B and E4B
cd "$(dirname "$0")"
R=$HOME/gemma4-dev
E2B=$R/gpu-vllm-t4-2b-w4a16
E4B=$R/gpu-vllm-t4-4b-q4w4a16emb4
SW=../2026-10-15-drafter-sweep-t4
BF=$R/gpu-vllm-t4-2b/benchmarks/runs/2026-10-15-drafter-bf16-t4
Q=queue.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $Q; }
job() {  # name, command...
  local name=$1; shift
  say "START $name"
  "$@" > $name.out 2>&1
  say "END $name rc=$?"
}
say "waiting for the drafter sweep"
until grep -q '^DONE' $SW/run.out 2>/dev/null && [ -e $SW/run_q4ct.out ] && ! pgrep -f run_q4ct.sh >/dev/null \
      && ! pgrep -f spec_ab.py >/dev/null; do sleep 30; done
say "drafter sweep finished: $(tail -1 $SW/run_q4ct.out)"
if curl -fs -o /dev/null --max-time 3 http://127.0.0.1:8000/health; then say "port 8000 is serving; abort"; exit 1; fi

ARMS=none,qat-n2,qat-n3,qat-n4
job conc-e2b python3 conc.py $E2B conc-e2b $ARMS
python3 summarize_conc.py conc-e2b > conc-e2b/summary.txt 2>&1
job conc-e4b python3 conc.py $E4B conc-e4b $ARMS
python3 summarize_conc.py conc-e4b > conc-e4b/summary.txt 2>&1

./build_suite.sh > build_suite.out 2>&1 & BUILD=$!
job quality-e2b python3 quality.py $E2B quality e2b none,qat-n3
job quality-e4b python3 quality.py $E4B quality e4b none,qat-n3
wait $BUILD; say "build_suite rc=$? $(tail -1 build_suite.out)"
for m in e2b:$E2B e4b:$E4B; do
  tag=${m%%:*}; rig=${m#*:}
  [ -d quality/suite-$tag ] || [ ! -e /opt1/data/nimble-suite/BUILD_OK ] || \
    SUITE_ONLY=1 job suite-$tag python3 quality.py $rig quality $tag none
done

mkdir -p $BF; cp $SW/spec_ab.py $SW/summarize.py $BF/
job bf16 env SPECULATIVE_MODEL=google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant \
  python3 $BF/spec_ab.py $R/gpu-vllm-t4-2b $BF/e2b-bf16 none,nq-n2,nq-n3,nq-n4,qat-n3,none-again
python3 $BF/summarize.py $BF/e2b-bf16 > $BF/e2b-bf16/summary.txt 2>&1

job longctx-e2b python3 longctx.py $E2B longctx-e2b $ARMS
job longctx-e4b python3 longctx.py $E4B longctx-e4b $ARMS
say "QUEUE DONE"
