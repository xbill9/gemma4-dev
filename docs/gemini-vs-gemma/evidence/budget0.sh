#!/bin/bash
# Native generateContent with thinkingConfig.thinkingBudget = 0: does any model stop thinking?
K=$(cat ~/gemini.key); Q="Janet has 16 eggs, eats 3, bakes with 4, sells the rest at \$2. How much does she make? End with 'Answer: <number>'."
for m in gemini-3.5-flash-lite gemini-3.8-flash gemini-4-hydrogen gemini-4-carbon gemini-4-argon; do
  printf "%-22s thinkingBudget=0  " $m
  curl -sS -m 120 "https://generativelanguage.googleapis.com/v1beta/models/$m:generateContent" -H "x-goog-api-key: $K" -H 'Content-Type: application/json' \
   -d "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"$Q\"}]}],\"generationConfig\":{\"temperature\":0,\"maxOutputTokens\":2048,\"thinkingConfig\":{\"thinkingBudget\":0}}}" |
  python3 -c "
import json,sys;r=json.load(sys.stdin)
if 'error' in r: print('rejected:', r['error']['message'][:80])
else: u=r['usageMetadata']; print('ok  prompt',u.get('promptTokenCount'),' answer',u.get('candidatesTokenCount'),' thinking',u.get('thoughtsTokenCount',0))"
done
