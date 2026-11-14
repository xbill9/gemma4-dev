#!/bin/bash
# Native generateContent: every thinkingConfig variant on Hydrogen and 3.8 Flash, reading thoughtsTokenCount.
K=$(cat ~/gemini.key); Q="Janet has 16 eggs, eats 3, bakes with 4, sells the rest at \$2. How much does she make? End with 'Answer: <number>'."
for m in gemini-4-hydrogen gemini-3.8-flash; do for tc in '{"thinkingBudget":0}' '{"thinkingLevel":"NONE"}' '{"thinkingLevel":"MINIMAL"}' '{"thinkingLevel":"LOW"}' '{"thinkingBudget":0,"includeThoughts":false}' '{"thinkingBudget":-1}' '{"thinkingBudget":0,"includeThoughts":true}'; do
printf "%-18s %-45s " $m "$tc"
curl -sS -m 120 "https://generativelanguage.googleapis.com/v1beta/models/$m:generateContent" -H "x-goog-api-key: $K" -H 'Content-Type: application/json' \
 -d "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"$Q\"}]}],\"generationConfig\":{\"temperature\":0,\"maxOutputTokens\":2048,\"thinkingConfig\":$tc}}" | python3 -c "
import json,sys;r=json.load(sys.stdin)
if 'error' in r: print('rejected:', r['error']['message'][:90])
else:
  u=r['usageMetadata']; parts=r['candidates'][0]['content'].get('parts',[])
  print('answer',u.get('candidatesTokenCount'),'thoughts',u.get('thoughtsTokenCount',0),'total',u.get('totalTokenCount'),'thought-parts',sum(1 for p in parts if p.get('thought')))"
done; done
