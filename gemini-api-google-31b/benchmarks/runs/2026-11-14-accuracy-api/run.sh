#!/bin/bash
# Apples-to-apples quality run, 2026-11-14: seven models (five Gemini, Gemma 4 26B and 31B) through
# the same Gemini API endpoint, started together, with identical settings. GSM8K (1,319, up to 2048
# tokens) and BFCL simple (400) via jev-tpu-v5e1/gen_eval.py, temperature 0, scored in code. One arm:
# the lowest thinking level the model accepts (LOWEST_REASONING_EFFORT in tpu.env). Concurrency 8,
# up to 8 retries on HTTP 429/5xx, network errors and requests over 180 s.
set -u
cd "$(dirname "$0")"; HERE=$PWD
RIG=$(cd ../../.. && pwd)
set -a; . "$RIG/tpu.env"; set +a
J=$(cd "$RIG/../jev-tpu-v5e1" && pwd)
export JEV_CHAT_URL=$GEMINI_OPENAI_URL JEV_API_KEY_FILE=$GEMINI_API_KEY_FILE JEV_RETRIES=8 JEV_TIMEOUT_S=180 JEV_GEN_MAX_TOKENS=2048
export JEV_REASONING_EFFORT=$LOWEST_REASONING_EFFORT
LOG=$HERE/driver.log
state() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
state "BEGIN $MODEL_NAME reasoning_effort=$JEV_REASONING_EFFORT"
for t in bfcl_simple gsm8k; do
  [ -s "$HERE/$t.jsonl" ] && { state "SKIP $t (exists)"; continue; }
  state "START $t"
  (cd "$J" && python3 gen_eval.py run x "$MODEL_NAME" $t "$HERE/$t.jsonl" --concurrency 8) > "$HERE/$t.log" 2>&1
  state "END $t rc=$? $(tail -1 "$HERE/$t.log")"
done
state "DONE"
