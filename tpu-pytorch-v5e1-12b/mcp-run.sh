#!/bin/bash
# Launch the tpu-pytorch-v5e1-12b MCP server with the parameters from tpu.env.
#
# MCP client configs can point here instead of straight at server.py so the zone,
# model, and accelerator live in exactly one place (tpu.env) rather than being
# duplicated into every .mcp.json. server.py also reads tpu.env on its own, so
# either entry point works; this one additionally exports the values to any
# subprocess it spawns.
#
# Only variables that are not already set are exported, so a value inherited
# from the environment always wins over tpu.env — the same precedence server.py
# applies when it loads the file itself.
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
