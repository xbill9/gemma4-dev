#!/bin/bash
# Launch this rig's MCP server with the parameters from tpu.env. The server names itself
# after the rig directory (RIG_NAME in server.py), so it reports as tpu-vllm-v5p1-2b.
# Register it in the client under that same name — the client's key is what prefixes the
# tools (mcp__tpu-vllm-v5p1-2b__find_tpu), and sibling rigs all answering to one shared
# name is what made it impossible to tell which rig a tool call would reach. Set
# MCP_SERVER_NAME in tpu.env or the environment if the client key has to be something else.
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
