#!/bin/bash
# Launch the tpu-2B-v5e1-devops-agent MCP server with the parameters from tpu.env.
#
# The MCP client configs point here rather than straight at server.py so the zone,
# project, and model live in exactly one place (tpu.env) instead of being duplicated
# into every mcp_config.json.
#
# Only variables that are not already set are exported, so a value inherited from the
# environment always wins over tpu.env — the same precedence python-dotenv gives
# server.py when it is run directly.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$HERE/tpu.env" ]; then
  while IFS='=' read -r key value; do
    case "$key" in
      ''|'#'*) continue ;;
    esac
    [ -z "${!key:-}" ] && export "$key=$value"
  done < "$HERE/tpu.env"
fi

exec python3 "$HERE/server.py"
