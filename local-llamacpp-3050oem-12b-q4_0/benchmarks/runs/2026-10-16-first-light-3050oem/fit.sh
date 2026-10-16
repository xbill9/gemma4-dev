#!/bin/bash
# Follow-up to bench.sh: the largest context that loads for Google's file, and for the exact
# rebuild with the exact Q4_0 drafter (32768 did not fit with it). Same alloc() as bench.sh.
set -u
cd "$(dirname "$0")"
BIN=/home/xbill/llama.cpp/build/bin
G=/home/xbill/models/gemma-4-12B-it-qat-q4_0-gguf/gemma-4-12b-it-qat-q4_0.gguf
X=/home/xbill/models/gemma-4-12B-it-qat-q4_0-exact/gemma-4-12B-it-q4_0-exact.gguf
DR=/home/xbill/models/gemma-4-12B-it-qat-assistant/gemma-4-12B-it-qat-assistant-q4_0-exact.gguf
DRQ8=/home/xbill/models/gemma-4-12B-it-qat-assistant/gemma-4-12B-it-qat-assistant-q8_0.gguf
alloc() {  # name, model, ctx, kv type, extra args...
  local n=$1 m=$2 c=$3 kv=$4; shift 4
  $BIN/llama-server -lv 4 -m $m --host 127.0.0.1 --port 8099 -ngl 99 -c $c -ctk $kv -ctv $kv -fa 1 \
    -t 6 -tb 12 --parallel 1 "$@" > alloc-$n.log 2>&1 &
  local pid=$!
  until curl -fs localhost:8099/health >/dev/null; do
    sleep 1; kill -0 $pid 2>/dev/null || { echo "== $n (-c $c, $kv): DID NOT LOAD"; return 1; }
  done
  echo "== $n (-c $c, $kv): nvidia-smi $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
  grep -E 'buffer size' alloc-$n.log | sed 's/^[0-9.]* //'
  kill $pid; wait $pid 2>/dev/null || true
}
D="--spec-type draft-mtp -ngld all --spec-draft-n-max 2"
for c in 28672 24576 20480 16384; do alloc google-$c $G $c f16 && break; done
for c in 40960 36864; do alloc exact-$c-f16 $X $c f16 && break; done
for c in 28672 24576 20480 16384 8192; do alloc exact-$c-f16-draft $X $c f16 -md $DR $D && break; done
for c in 49152 40960 32768; do alloc exact-$c-q8-draft $X $c q8_0 -md $DR $D && break; done
alloc exact-8192-f16-draftq8 $X 8192 f16 -md $DRQ8 $D
alloc exact-8192-f16 $X 8192 f16
