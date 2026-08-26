#!/usr/bin/env bash
# Publish the dev.to article as a DRAFT (front matter sets published: false).
# Key is read from $DEV_TO_API_KEY or ~/.devto.key -- never passed on the command line.
set -euo pipefail
ART="${1:-devto-gke-gemma4-v6e1-step-by-step.md}"
KEY="${DEV_TO_API_KEY:-}"
[ -z "$KEY" ] && [ -f "$HOME/.devto.key" ] && KEY=$(tr -d '\r\n' < "$HOME/.devto.key")
if [ -z "$KEY" ]; then
  echo "No API key. Either:  export DEV_TO_API_KEY=...   or:  printf %s '<key>' > ~/.devto.key && chmod 600 ~/.devto.key" >&2
  exit 1
fi
[ -f "$ART" ] || { echo "missing $ART" >&2; exit 1; }
python3 - "$ART" > /tmp/devto-payload.json <<'PY'
import json,sys
md=open(sys.argv[1]).read()
print(json.dumps({"article":{"body_markdown":md}}))
PY
echo "Posting $ART ($(wc -c < "$ART") bytes) as a draft..."
curl -sS -X POST https://dev.to/api/articles \
  -H "api-key: $KEY" -H "Content-Type: application/json" \
  --data @/tmp/devto-payload.json \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
if 'error' in d: print('ERROR:', d.get('error'), d.get('status','')); raise SystemExit(1)
print('OK  id=%s  published=%s' % (d.get('id'), d.get('published')))
print('edit:', d.get('url') or ('https://dev.to/dashboard'))
"
