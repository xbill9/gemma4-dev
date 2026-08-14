#!/usr/bin/env bash
# Publish or update a dev.to article.
#
#   publish-devto.sh [FILE]              create a NEW draft (front matter sets published: false)
#   publish-devto.sh --list              list your articles with their ids
#   publish-devto.sh --update ID [FILE]  overwrite an EXISTING article in place
#
# The article's front matter (title, tags, cover_image, published) is part of
# body_markdown, so --update rewrites those too. Changing the title does not
# change the slug of an already-published article, so existing links survive.
#
# Key is read from $DEV_TO_API_KEY or ~/.devto.key -- never passed on the command line.
set -euo pipefail

MODE=create
ID=""
case "${1:-}" in
  --list)   MODE=list; shift ;;
  --update) MODE=update; ID="${2:-}"; shift 2 || true ;;
esac
ART="${1:-devto-gemma4-g5g-graviton-turing.md}"

KEY="${DEV_TO_API_KEY:-}"
[ -z "$KEY" ] && [ -f "$HOME/.devto.key" ] && KEY=$(tr -d '\r\n' < "$HOME/.devto.key")
if [ -z "$KEY" ]; then
  echo "No API key. Either:  export DEV_TO_API_KEY=...   or:  printf %s '<key>' > ~/.devto.key && chmod 600 ~/.devto.key" >&2
  exit 1
fi

if [ "$MODE" = list ]; then
  curl -sS -H "api-key: $KEY" 'https://dev.to/api/articles/me/all?per_page=100' | python3 -c "
import json,sys
d=json.load(sys.stdin)
if isinstance(d,dict) and 'error' in d: print('ERROR:', d.get('error')); raise SystemExit(1)
for a in d:
    print('%-10s %-9s %s' % (a['id'], 'published' if a['published'] else 'draft', a['title']))
    print('%-10s %s' % ('', a.get('url','')))
"
  exit 0
fi

[ -f "$ART" ] || { echo "missing $ART" >&2; exit 1; }
if [ "$MODE" = update ] && ! [[ "$ID" =~ ^[0-9]+$ ]]; then
  echo "--update needs a numeric article id. Run:  $0 --list" >&2
  exit 1
fi

python3 - "$ART" > /tmp/devto-payload.json <<'PY'
import json,sys
md=open(sys.argv[1]).read()
print(json.dumps({"article":{"body_markdown":md}}))
PY

if [ "$MODE" = update ]; then
  echo "Updating article $ID from $ART ($(wc -c < "$ART") bytes)..."
  METHOD=PUT
  URL="https://dev.to/api/articles/$ID"
else
  echo "Posting $ART ($(wc -c < "$ART") bytes) as a draft..."
  METHOD=POST
  URL="https://dev.to/api/articles"
fi

curl -sS -X "$METHOD" "$URL" \
  -H "api-key: $KEY" -H "Content-Type: application/json" \
  --data @/tmp/devto-payload.json \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
if 'error' in d: print('ERROR:', d.get('error'), d.get('status','')); raise SystemExit(1)
print('OK  id=%s  published=%s' % (d.get('id'), d.get('published')))
print('title:', d.get('title'))
print('url:', d.get('url') or 'https://dev.to/dashboard')
"
