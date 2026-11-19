#!/bin/bash
# Which model for which job, 2026-11-19. AIME 2025 II (15 problems: the second half of the pinned
# opencompass/AIME2025 set, budget_run.py --offset 15), eight models through the Gemini API's
# OpenAI-compatible endpoint, all started together so their times compare, temperature 0, a
# 32,768-token output limit for every model and setting.
#   *-low:  each model's lowest accepted setting, as docs/gemini-high-vs-low (minimal for Gemma and
#           Flash-Lite, none for the rest; none runs as low on Gemini 3.8 and 4)
#   *-high: each model's top setting from docs/gemini-high-vs-low (xhigh for Hydrogen, high for the rest),
#           Neon at high, Google's default and the setting of its Carbon comparison
# A $20 cap: --per-wave-cap skips the whole-run refusal (every reply at full length would be ~$48),
# and the per-wave check stops before any wave of 3 problems whose worst case could pass $20.
set -u
cd "$(dirname "$0")"; HERE=$PWD
J=$(cd ../../../benchmarks/gen-eval && pwd)
export GEN_EVAL_CHAT_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
export GEN_EVAL_API_KEY_FILE=~/gemini.key GEN_EVAL_RETRIES=3 GEN_EVAL_TIMEOUT_S=1200 GEN_EVAL_HARD_MAX_TOKENS=32768
# Restarted 2026-11-19 ~13:12 as two runs, rows already saved kept: the Gemini models (models-paid.json) under the
# $20 cap, and Gemma 4 on the free tier (models-gemma.json) on its own, because each wave waits for its slowest
# reply and Gemma 4 26B free-tier replies took up to 80 minutes (docs/gemini-high-vs-low). Both still run at once.
(cd "$J" && python3 budget_run.py "$HERE/models-paid.json" aime2025 "$HERE/aime2" --budget 20 --offset 15 --limit 15 --wave 3 --per-wave-cap) > "$HERE/aime2.log" 2>&1 &
(cd "$J" && python3 budget_run.py "$HERE/models-gemma.json" aime2025 "$HERE/aime2" --budget 1 --offset 15 --limit 15 --wave 3) > "$HERE/aime2-gemma.log" 2>&1 &
wait
