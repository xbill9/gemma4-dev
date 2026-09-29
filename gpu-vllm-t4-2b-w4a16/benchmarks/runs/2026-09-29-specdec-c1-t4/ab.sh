#!/usr/bin/env bash
# Single-user A/B of Gemma 4's own drafter on the emb4 build.
#
#   bash ab.sh [LABEL:K:MAXSEQS ...]    K=0 is no drafter; default baseline, spec2, spec4
# Server flags are the rig's tpu.env values
# (the same ones ~/bin/vllm-t4 passes), plus --speculative-config.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
rig=$(cd "$here/../../.." && pwd)
export PYTHONUSERBASE=/opt1/pyuser TMPDIR=/opt1/tmp
MODEL=xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4
DRAFT=google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant

serve() {  # label, extra args...
  label=$1; shift
  echo "=== $(date -u +%FT%TZ) specdec A/B: $label $*" >> "$rig/run/vllm.log"
  nohup /usr/bin/python3.13 -m vllm.entrypoints.openai.api_server --model $MODEL \
    --host 127.0.0.1 --port 8000 --dtype float16 --kv-cache-dtype auto --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-seqs "$MAXSEQS" --language-model-only \
    "$@" >> "$rig/run/vllm.log" 2>&1 &
  pid=$!; echo $pid > "$rig/run/vllm.pid"
  t0=$(date +%s)
  until curl -fs -o /dev/null --max-time 5 http://127.0.0.1:8000/health; do
    kill -0 $pid 2>/dev/null || { echo "$label: SERVER DIED"; tail -25 "$rig/run/vllm.log"; return 1; }
    sleep 10
  done
  echo "$label: healthy after $(( $(date +%s)-t0 ))s"
  (cd "$here" && python3 demo_c1.py "$label" ${HARNESS_ARGS:-})
  kill $pid; while kill -0 $pid 2>/dev/null; do sleep 2; done; sleep 5
}

[ $# -gt 0 ] || set -- baseline:0:8 spec2:2:8 spec4:4:8
for cfg in "$@"; do
  IFS=: read -r label k MAXSEQS <<< "$cfg"
  if [ "$k" = 0 ]; then serve "$label"
  else serve "$label" --speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":$k}"; fi
done
echo DONE
