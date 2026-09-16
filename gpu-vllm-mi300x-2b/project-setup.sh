#!/usr/bin/env bash
# Register the gpu-vllm-mi300x-2b MCP server in a target project (or globally)
# and install its skill.
#
# The registered key prefixes every tool name
# (mcp__gpu-vllm-mi300x-2b__serving_status), so it must match the rig directory
# or two loaded rigs are indistinguishable at the call site. --server-name sets
# both the registered key and what the server advertises.
#
# Never creates a virtualenv: these rigs install into the system python3.
set -euo pipefail

RIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIG_NAME="$(basename "$RIG_DIR")"
SERVER_NAME="$RIG_NAME"
SKILL_NAME="$RIG_NAME-management"
TARGET=""
GLOBAL=0
TAG=""

usage() {
  cat <<USAGE
usage: project-setup.sh <target-dir> [--server-name NAME] [--tag DROPLET_TAG]
       project-setup.sh --global [--server-name NAME] [--tag DROPLET_TAG]

  <target-dir>     project to write .mcp.json into
  --global         register in ~/.claude.json instead of a project
  --server-name    override the registered MCP key (default: $RIG_NAME)
  --tag            DigitalOcean tag scoping which droplets are visible
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --global) GLOBAL=1; shift ;;
    --server-name) SERVER_NAME="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown flag: $1" >&2; usage; exit 1 ;;
    *) TARGET="$1"; shift ;;
  esac
done

if [ "$GLOBAL" -eq 0 ] && [ -z "$TARGET" ]; then
  usage; exit 1
fi

# Dependencies go into the system python3. A missing one is a warning with the
# command to fix it, not a venv created behind your back.
if ! python3 -c 'import mcp, httpx, dotenv' 2>/dev/null; then
  echo "warning: dependencies missing. Install them with:"
  echo "    python3 -m pip install -r $RIG_DIR/requirements.txt"
fi

if [ -n "$TAG" ]; then
  echo "note: set DROPLET_TAG=$TAG in $RIG_DIR/tpu.env — it is the committed source of truth."
fi

CONFIG_JSON=$(cat <<JSON
{
  "mcpServers": {
    "$SERVER_NAME": {
      "command": "python3",
      "args": ["$RIG_DIR/server.py"],
      "env": {
        "MCP_SERVER_NAME": "$SERVER_NAME"
      }
    }
  }
}
JSON
)

if [ "$GLOBAL" -eq 1 ]; then
  echo "$CONFIG_JSON" > "$HOME/.mcp-$SERVER_NAME.json"
  echo "wrote $HOME/.mcp-$SERVER_NAME.json"
  echo "Add it with: claude mcp add-json $SERVER_NAME \"\$(cat $HOME/.mcp-$SERVER_NAME.json)\""
else
  mkdir -p "$TARGET"
  echo "$CONFIG_JSON" > "$TARGET/.mcp.json"
  echo "wrote $TARGET/.mcp.json (gitignored by convention — it is generated)"
  mkdir -p "$TARGET/.claude/skills"
  rm -rf "${TARGET:?}/.claude/skills/$SKILL_NAME"
  cp -r "$RIG_DIR/.claude/skills/$SKILL_NAME" "$TARGET/.claude/skills/$SKILL_NAME"
  echo "installed skill -> $TARGET/.claude/skills/$SKILL_NAME"
fi

echo "✅ $SERVER_NAME registered. Tools will appear as mcp__${SERVER_NAME}__*"
