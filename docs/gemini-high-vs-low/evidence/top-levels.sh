#!/bin/bash
# Which top thinking level each model accepts through the OpenAI-compatible endpoint, 2026-11-15.
# One tiny request per (model, level), max_tokens 256; prints status, thinking tokens (total - prompt - completion).
K=$(cat ~/gemini.key); U=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
for ml in gemma-4-31b-it:high gemma-4-26b-a4b-it:high gemini-3.5-flash-lite:high gemini-3.8-flash:high gemini-4-hydrogen:high gemini-4-hydrogen:xhigh gemini-4-carbon:high gemini-4-carbon:xhigh gemini-4-argon:high gemini-4-argon:xhigh; do
  m=${ml%%:*}; e=${ml##*:}
  printf "%-22s %-6s " $m $e
  curl -sS -m 120 $U -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$m\",\"reasoning_effort\":\"$e\",\"max_tokens\":256,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? End with 'Answer: <number>'.\"}]}" | python3 -c "
import json,sys
r=json.load(sys.stdin)
if isinstance(r,list): r=r[0]
if 'error' in r: print('rejected:', str(r['error'].get('message'))[:100])
else:
  u=r['usage']; print('ok  thinking', u.get('total_tokens',0)-u.get('prompt_tokens',0)-u['completion_tokens'], 'answer', u['completion_tokens'], 'finish', r['choices'][0]['finish_reason'])"
done
