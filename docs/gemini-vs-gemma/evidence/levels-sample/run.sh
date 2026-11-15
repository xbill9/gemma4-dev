#!/bin/bash
# What does reasoning_effort "none" do on the models that reject "minimal"? First 100 GSM8K problems,
# same script and settings as the 2026-11-14-accuracy-api run, at none / low / medium / unset (model default).
cd "$(dirname "$0")"; HERE=$PWD; J=$(cd ../../../../benchmarks/gen-eval && pwd)
export JEV_CHAT_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions JEV_API_KEY_FILE=$HOME/gemini.key JEV_RETRIES=8 JEV_TIMEOUT_S=180 JEV_GEN_MAX_TOKENS=2048
for m in gemini-4-hydrogen gemini-3.8-flash gemini-4-carbon gemini-4-argon; do
  ( for e in none low medium unset; do
      [ -s "$HERE/$m-$e.jsonl" ] && continue
      if [ $e = unset ]; then unset JEV_REASONING_EFFORT; else export JEV_REASONING_EFFORT=$e; fi
      (cd "$J" && python3 gen_eval.py run x $m gsm8k "$HERE/$m-$e.jsonl" --limit 100 --concurrency 8) > "$HERE/$m-$e.log" 2>&1
    done ) &
done; wait; date '+%F %T %Z DONE' > "$HERE/done.txt"
