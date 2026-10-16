#!/bin/bash
# First light for the 12B exact Q4_0 GGUF on the RTX 3050 OEM.
# 1) allocation: Google's file and the exact rebuild at -c 32768, then the exact
#    rebuild at larger contexts (f16 and q8_0 KV) until it stops fitting.
# 2) llama-bench ABBA (google, exact, exact, google), pp512/tg128, r=5, cooled to <=40 C.
# 3) llama-bench by depth on the exact rebuild, r=3.
# 1b) the same with the exact Q4_0 12B drafter loaded.
# 4) E4B exact on the same binary, pp512/tg128 r=5, as the same-day reference.
set -u
cd "$(dirname "$0")"
BIN=/home/xbill/llama.cpp/build/bin
G=/home/xbill/models/gemma-4-12B-it-qat-q4_0-gguf/gemma-4-12b-it-qat-q4_0.gguf
X=/home/xbill/models/gemma-4-12B-it-qat-q4_0-exact/gemma-4-12B-it-q4_0-exact.gguf
E4B=/home/xbill/models/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-q4_0-exact.gguf
temp() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader; }
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 40 ]; do sleep 5; done; }
echo "llama.cpp $(git -C /home/xbill/llama.cpp rev-parse --short HEAD)"
alloc() {  # name, model, ctx, kv type, extra args...
  local n=$1 m=$2 c=$3 kv=$4; shift 4
  $BIN/llama-server -lv 4 -m $m --host 127.0.0.1 --port 8099 -ngl 99 -c $c -ctk $kv -ctv $kv -fa 1 \
    -t 6 -tb 12 --parallel 1 "$@" > alloc-$n.log 2>&1 &
  local pid=$!
  until curl -fs localhost:8099/health >/dev/null; do
    sleep 1; kill -0 $pid 2>/dev/null || { echo "== $n (-c $c, $kv): DID NOT LOAD"; grep -iE 'out of memory|failed|error' alloc-$n.log | head -3; return 1; }
  done
  echo "== $n (-c $c, $kv): nvidia-smi $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
  grep -E 'buffer size' alloc-$n.log | sed 's/^[0-9.]* //'
  kill $pid; wait $pid 2>/dev/null || true
}
alloc google-32k $G 32768 f16
alloc exact-32k $X 32768 f16
for c in 49152 65536 98304 131072; do alloc exact-${c}-f16 $X $c f16 || break; done
for c in 65536 98304 131072; do alloc exact-${c}-q8 $X $c q8_0 || break; done
DR=/home/xbill/models/gemma-4-12B-it-qat-assistant/gemma-4-12B-it-qat-assistant-q4_0-exact.gguf
for c in 32768 49152 65536; do alloc exact-${c}-f16-draft $X $c f16 -md $DR --spec-type draft-mtp -ngld all --spec-draft-n-max 2 || break; done
echo
pass() { local n=$1 l=$2 m=$3; cool; echo "gpu before pass $n ($l): $(temp)"
  $BIN/llama-bench -m $m -ngl 99 -fa 1 -t 6 -p 512 -n 128 -r 5 -o md; echo "gpu after: $(temp)"; echo; }
pass 1 google $G
pass 2 exact $X
pass 3 exact $X
pass 4 google $G
cool; echo "depth sweep, exact, gpu before: $(temp)"
$BIN/llama-bench -m $X -ngl 99 -fa 1 -t 6 -p 512 -n 128 -d 0,8192,16384,32768 -r 3 -o md
echo "gpu after: $(temp)"; echo
pass 5 e4b-exact $E4B
