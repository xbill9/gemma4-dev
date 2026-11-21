#!/bin/bash
# After the sampling batch: stop wave 3 (resumable), serve the same GGUF with an f16 cache, resend the
# same eight problems greedily, then restart wave 3.
set -u
H=$(pwd); W=/home/xbill/gemma4-dev/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-11-17-thinking-12b-3050oem
log() { echo "$(date '+%F %T') $*" | tee -a notes.txt; }
while pgrep -f "loops.py google" >/dev/null; do sleep 5; done
log "sampling batch done; stopping chain.sh 920329, run.sh, gen_eval, llama-server"
kill 920329 2>/dev/null; pkill -f '^/bin/bash ./run.sh$'; kill -CONT 3081993; kill 3081993
pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done
log "wave-3 partial rows: $(wc -l < $W/gsm8k-repack-none-thinkon.jsonl.partial)"
setsid /home/xbill/llama.cpp/build/bin/llama-server -m /home/xbill/models/gemma-4-12B-it-qat-q4_0-exact/gemma-4-12B-it-q4_0-exact.gguf \
  --host 127.0.0.1 --port 8080 -ngl 99 -c 40960 --parallel 1 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 --reasoning off --metrics \
  > server-f16.log 2>&1 < /dev/null &
for i in $(seq 600); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
log "f16 server: $(curl -fsS http://127.0.0.1:8080/health) $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
python3 loops.py greedy greedy-f16kv.jsonl 8 > greedy-f16kv.log 2>&1
log "greedy f16 batch done"
pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done
cd $W && setsid nohup ./run.sh >> nohup-run.out 2>&1 < /dev/null &
log "wave 3 restarted"
