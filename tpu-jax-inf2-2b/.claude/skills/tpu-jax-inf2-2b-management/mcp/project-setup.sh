#!/usr/bin/env bash
# Install the bundled skill and register its AWS Inf2 MCP server.
set -euo pipefail

err() { echo "error: $*" >&2; exit 1; }
info() { echo "==> $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.claude/skills/tpu-jax-inf2-2b-management/SKILL.md" ]; then
  SKILL_SRC="$SCRIPT_DIR/.claude/skills/tpu-jax-inf2-2b-management"
  DEFAULT_SERVER_NAME="$(basename "$SCRIPT_DIR")"   # ...which is the rig directory
elif [ -f "$SCRIPT_DIR/../SKILL.md" ]; then
  SKILL_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
  DEFAULT_SERVER_NAME=""                            # no rig dir here; see TARGET_DIR below
else
  err "cannot locate the bundled skill"
fi

TARGET_DIR=""
GLOBAL=0
SKIP_DEPS=0
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
AWS_PROFILE="${AWS_PROFILE:-}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
INSTANCE_TYPE="${INSTANCE_TYPE:-inf2.xlarge}"
SERVER_NAME=""

while [ $# -gt 0 ]; do
  case "$1" in
    --global) GLOBAL=1 ;;
    --region) AWS_REGION="$2"; shift ;;
    --profile) AWS_PROFILE="$2"; shift ;;
    --model) MODEL_NAME="$2"; shift ;;
    --instance-type) INSTANCE_TYPE="$2"; shift ;;
    --server-name) SERVER_NAME="$2"; shift ;;
    --skip-deps) SKIP_DEPS=1 ;;
    -h|--help)
      echo "Usage: ./project-setup.sh [TARGET_DIR] [options]"
      echo "  --global                 Install for all Claude Code projects"
      echo "  --region REGION          AWS region (default: us-east-1)"
      echo "  --profile PROFILE        Optional AWS shared-config profile"
      echo "  --model MODEL            Hugging Face model ID"
      echo "  --instance-type TYPE     inf2.xlarge, inf2.8xlarge, inf2.24xlarge, or inf2.48xlarge"
      echo "  --server-name NAME       MCP server name; the key it is registered under, which"
      echo "                           prefixes every tool as mcp__<name>__find_inf2."
      echo "                           (default: the rig directory, tpu-jax-inf2-2b)"
      echo "  --skip-deps              Skip the dependency import check"
      exit 0 ;;
    -*) err "unknown option: $1" ;;
    *) [ -z "$TARGET_DIR" ] || err "unexpected argument: $1"; TARGET_DIR="$1" ;;
  esac
  shift
done

case "$INSTANCE_TYPE" in
  inf2.xlarge|inf2.8xlarge|inf2.24xlarge|inf2.48xlarge) ;;
  *) err "unsupported instance type: $INSTANCE_TYPE" ;;
esac
[ "$GLOBAL" -eq 0 ] || [ -z "$TARGET_DIR" ] || err "--global and TARGET_DIR are mutually exclusive"
TARGET_DIR="${TARGET_DIR:-$PWD}"
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"

# The client's key for this server prefixes every tool (mcp__<key>__find_inf2), so it has to
# identify the rig. A shared constant is what made the sibling rigs indistinguishable in
# /mcp. Default to the rig directory this script ships in; from an unzipped bundle there is
# no rig directory, so fall back to the project being set up.
SERVER_NAME="${SERVER_NAME:-${DEFAULT_SERVER_NAME:-$(basename "$TARGET_DIR")}}"

if [ "$GLOBAL" -eq 1 ]; then
  SKILL_DEST="$HOME/.claude/skills/tpu-jax-inf2-2b-management"
else
  SKILL_DEST="$TARGET_DIR/.claude/skills/tpu-jax-inf2-2b-management"
fi
mkdir -p "$(dirname "$SKILL_DEST")"
if [ "$SKILL_SRC" != "$SKILL_DEST" ]; then
  rm -rf "$SKILL_DEST"
  cp -r "$SKILL_SRC" "$SKILL_DEST"
fi
info "skill installed: $SKILL_DEST"

if [ "$SKIP_DEPS" -eq 0 ] && ! python3 -c 'import boto3,httpx,openai,mcp' >/dev/null 2>&1; then
  echo "warning: install dependencies with: python3 -m pip install -r $SKILL_DEST/mcp/requirements.txt" >&2
fi

ENV_JSON="$(python3 - "$AWS_REGION" "$AWS_PROFILE" "$MODEL_NAME" "$INSTANCE_TYPE" "$SERVER_NAME" <<'PY'
import json, sys
region, profile, model, instance_type, server_name = sys.argv[1:]
# MCP_SERVER_NAME keeps the name server.py advertises equal to the key it is registered
# under; they have to agree or /mcp and the tool prefix disagree about what this is.
env = {"AWS_REGION": region, "MODEL_NAME": model, "INSTANCE_TYPE": instance_type,
       "MCP_SERVER_NAME": server_name}
if profile:
    env["AWS_PROFILE"] = profile
print(json.dumps(env))
PY
)"

if [ "$GLOBAL" -eq 1 ]; then
  command -v claude >/dev/null 2>&1 || err "--global requires the claude CLI"
  SERVER_JSON="$(python3 -c 'import json,sys; print(json.dumps({"command":"python3","args":[sys.argv[1]],"env":json.loads(sys.argv[2])}))' "$SKILL_DEST/mcp/server.py" "$ENV_JSON")"
  claude mcp remove --scope user "$SERVER_NAME" >/dev/null 2>&1 || true
  claude mcp add-json --scope user "$SERVER_NAME" "$SERVER_JSON"
else
  python3 - "$TARGET_DIR/.mcp.json" "$SERVER_NAME" "$ENV_JSON" <<'PY'
import json, sys
path, name, env = sys.argv[1:]
try:
    with open(path) as handle:
        data = json.load(handle)
except FileNotFoundError:
    data = {}
data.setdefault("mcpServers", {})[name] = {
    "command": "python3",
    "args": [".claude/skills/tpu-jax-inf2-2b-management/mcp/server.py"],
    "env": json.loads(env),
}
with open(path, "w") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
PY
fi

info "registered MCP server '$SERVER_NAME' for AWS region $AWS_REGION"
echo "Run 'aws sts get-caller-identity' to verify credentials, then restart Claude Code."
