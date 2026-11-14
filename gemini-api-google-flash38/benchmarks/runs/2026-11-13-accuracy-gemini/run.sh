#!/bin/bash
# Answer quality of Gemini 3.8 Flash (gemini-3.8-flash) through the Gemini API's OpenAI-compatible endpoint, with the
# Gemma rigs' method: GSM8K (1,319, up to 2048 tokens) and BFCL simple (400) via
# jev-tpu-v5e1/gen_eval.py, temperature 0, scored in code. Two arms: reasoning_effort=none (the
# lowest level this model accepts) and low. Concurrency 8, failed requests retried up to 8 times
# (HTTP 429/5xx and network errors). Thinking tokens are recorded per row as reasoning_tokens.
set -u
cd "$(dirname "$0")"; HERE=$PWD
RIG=$(cd ../../.. && pwd)
set -a; . "$RIG/tpu.env"; set +a
J=$(cd "$RIG/../jev-tpu-v5e1" && pwd)
export JEV_CHAT_URL=$GEMINI_OPENAI_URL JEV_API_KEY_FILE=$GEMINI_API_KEY_FILE JEV_RETRIES=8 JEV_GEN_MAX_TOKENS=2048
LOG=$HERE/driver.log
state() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
state "BEGIN $MODEL_NAME"
for arm in $LOWEST_REASONING_EFFORT low; do
  for t in bfcl_simple gsm8k; do
    [ -s "$HERE/$t-$arm.jsonl" ] && { state "SKIP $t $arm (exists)"; continue; }
    state "START $t $arm"
    (cd "$J" && JEV_REASONING_EFFORT=$arm python3 gen_eval.py run x "$MODEL_NAME" $t "$HERE/$t-$arm.jsonl" --concurrency 8) > "$HERE/$t-$arm.log" 2>&1
    state "END $t $arm rc=$? $(tail -1 "$HERE/$t-$arm.log")"
  done
done
state "DONE"
