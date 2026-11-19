#!/bin/bash
# Gemini 4 Neon at each of its four thinking levels, 2026-11-19. AIME 2025 I (15 problems, the pinned set
# docs/gemini-high-vs-low used), temperature 0, 32,768-token output limit for every level (the limit
# the seven-model high/low run used; Neon's own maximum is 65,536). budget_run.py sends every level the
# same problems in waves of 3 and stops before a wave whose worst case would pass the cap.
set -u
cd "$(dirname "$0")"; HERE=$PWD
J=$(cd ../../../../benchmarks/gen-eval && pwd)
export GEN_EVAL_CHAT_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
export GEN_EVAL_API_KEY_FILE=~/gemini.key GEN_EVAL_RETRIES=${GEN_EVAL_RETRIES:-3} GEN_EVAL_TIMEOUT_S=${GEN_EVAL_TIMEOUT_S:-1200} GEN_EVAL_HARD_MAX_TOKENS=32768
(cd "$J" && python3 budget_run.py "$HERE/models.json" aime2025 "$HERE/aime" --budget ${BUDGET:-30} --limit 15 --wave 3) > "$HERE/aime.log" 2>&1
