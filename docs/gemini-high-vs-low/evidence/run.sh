#!/bin/bash
# High vs low thinking on hard problems, 2026-11-15. AIME 2025 I (15 problems, opencompass/AIME2025
# at a pinned revision), seven models through the Gemini API's OpenAI-compatible endpoint, temperature 0,
# a 32,768-token output limit for every model and arm (Carbon's and Argon's own per-request maximum).
#   arm high: each model's top accepted level (top-levels.txt): xhigh for Hydrogen, high for the rest
#   arm low:  each model's lowest accepted level, as in the 2026-11-14 run: minimal (Gemma, Flash-Lite), none (rest)
# Each arm runs under its own $20 stop against a $16.62 worst case, so both together stay under $40.
set -u
cd "$(dirname "$0")"; HERE=$PWD
J=$(cd ../../../benchmarks/gen-eval && pwd)
export GEN_EVAL_CHAT_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
export GEN_EVAL_API_KEY_FILE=~/gemini.key GEN_EVAL_RETRIES=${GEN_EVAL_RETRIES:-8} GEN_EVAL_TIMEOUT_S=${GEN_EVAL_TIMEOUT_S:-3600} GEN_EVAL_HARD_MAX_TOKENS=32768
# The low arm was restarted 2026-11-15 13:20 as GEN_EVAL_TIMEOUT_S=1200 GEN_EVAL_RETRIES=3 ./run.sh low, after one
# Gemma 4 26B request sat open over 30 minutes (aime-low.attempt1.log); rows already saved were kept.
# The high arm was restarted 13:46 the same way, after one request sat open over 30 minutes
# (aime-high.attempt1.log); its unwritten replies to problem 9 were requested again.
for arm in "$@"; do
  (cd "$J" && python3 budget_run.py "$HERE/models-$arm.json" aime2025 "$HERE/aime-$arm" --budget 20 --limit 15 --wave 3) \
    > "$HERE/aime-$arm.log" 2>&1 &
done
wait
