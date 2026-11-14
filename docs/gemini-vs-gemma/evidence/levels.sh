#!/bin/bash
# Which reasoning_effort values each model accepts on the Gemini API's OpenAI-compatible endpoint
# (and what it does with none sent), with the thinking tokens one short question used
# (thinking = total_tokens - prompt_tokens - completion_tokens).
K=$(cat ~/gemini.key)
Q="Janet has 16 eggs, eats 3, bakes with 4, sells the rest at \$2. How much does she make? End with 'Answer: <number>'."
for m in gemma-4-31b-it gemma-4-26b-a4b-it gemini-3.5-flash-lite gemini-3.8-flash gemini-4-hydrogen gemini-4-carbon gemini-4-argon; do
  for eff in minimal none low unset; do
    if [ $eff = unset ]; then E=""; else E="\"reasoning_effort\":\"$eff\","; fi
    printf "%-22s %-8s " $m $eff
    curl -sS -m 300 https://generativelanguage.googleapis.com/v1beta/openai/chat/completions \
      -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$m\",\"temperature\":0,\"max_tokens\":2048,$E\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}]}" |
    python3 -c "
import json,sys;r=json.load(sys.stdin)
if isinstance(r,list): r=r[0]
if 'error' in r: print('rejected')
else:
  u=r['usage'];c=r['choices'][0];last=(c['message']['content'] or '').strip().splitlines()[-1:]
  print('thinking',u['total_tokens']-u['prompt_tokens']-u['completion_tokens'],' finish',c['finish_reason'],' ',last[0][:24] if last and c['finish_reason']=='stop' else 'no answer')"
  done
done
