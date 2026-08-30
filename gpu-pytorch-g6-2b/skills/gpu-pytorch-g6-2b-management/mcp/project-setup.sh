#!/usr/bin/env bash
# Install the bundled skill and register its EC2 G6 MCP server.
set -euo pipefail

err() { echo "error: $*" >&2; exit 1; }
info() { echo "==> $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Derived, never a literal: the skill name carries the rig directory (NAMING.md).
# A hardcoded stem here survived the vllm->jax fork and pointed .mcp.json at a
# skill path that does not exist -- and failed the lookup below outright.
SKILL_STEM="$(basename "$SCRIPT_DIR")-management"
if [ -f "$SCRIPT_DIR/.claude/skills/$SKILL_STEM/SKILL.md" ]; then
  SKILL_SRC="$SCRIPT_DIR/.claude/skills/$SKILL_STEM"
  DEFAULT_SERVER_NAME="$(basename "$SCRIPT_DIR")"   # ...which is the rig directory
elif [ -f "$SCRIPT_DIR/../SKILL.md" ]; then
  # Unzipped bundle: this script sits at <skill>/mcp/, so the parent is the skill.
  SKILL_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
  SKILL_STEM="$(basename "$SKILL_SRC")"
  DEFAULT_SERVER_NAME=""                            # no rig dir here; see TARGET_DIR below
else
  err "cannot locate the bundled skill"
fi

TARGET_DIR=""
GLOBAL=0
SKIP_DEPS=0
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
AWS_PROFILE="${AWS_PROFILE:-}"
MODEL_NAME="${MODEL_NAME:-google/gemma-4-E2B-it}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g6.2xlarge}"
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
      echo "  --instance-type TYPE     g6.xlarge, g6.2xlarge, g6.4xlarge, g6.8xlarge,"
      echo "                           g6.12xlarge, g6.16xlarge, g6.24xlarge, g6.48xlarge"
      echo "                           (g6.xlarge gets a swapfile: 16 GiB host RAM)"
      echo "  --server-name NAME       MCP server name; the key it is registered under, which"
      echo "                           prefixes every tool as mcp__<name>__create_g6_instance."
      echo "                           (default: the rig directory, gpu-pytorch-g6-2b)"
      echo "  --skip-deps              Skip the dependency import check"
      exit 0 ;;
    -*) err "unknown option: $1" ;;
    *) [ -z "$TARGET_DIR" ] || err "unexpected argument: $1"; TARGET_DIR="$1" ;;
  esac
  shift
done

# Every G6 size is supported. The G5g rig rejected its xlarge here because that
# size had 8 GiB of host RAM; G6 has twice the RAM at every suffix, so g6.xlarge
# is 16 GiB and merely gets a swapfile (see _SWAP_AT_OR_BELOW_HOST_RAM_GB).
# There is no g6.metal.
case "$INSTANCE_TYPE" in
  g6.xlarge|g6.2xlarge|g6.4xlarge|g6.8xlarge|g6.12xlarge|g6.16xlarge|g6.24xlarge|g6.48xlarge) ;;
  *) err "unsupported instance type: $INSTANCE_TYPE" ;;
esac
[ "$GLOBAL" -eq 0 ] || [ -z "$TARGET_DIR" ] || err "--global and TARGET_DIR are mutually exclusive"
TARGET_DIR="${TARGET_DIR:-$PWD}"
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"

# The client's key for this server prefixes every tool, so it has to identify the
# rig. A shared constant is what made the sibling rigs indistinguishable in /mcp.
# Default to the rig directory this script ships in; from an unzipped bundle
# there is no rig directory, so fall back to the project being set up.
SERVER_NAME="${SERVER_NAME:-${DEFAULT_SERVER_NAME:-$(basename "$TARGET_DIR")}}"

if [ "$GLOBAL" -eq 1 ]; then
  SKILL_DEST="$HOME/.claude/skills/$SKILL_STEM"
else
  SKILL_DEST="$TARGET_DIR/.claude/skills/$SKILL_STEM"
fi
mkdir -p "$(dirname "$SKILL_DEST")"
if [ "$SKILL_SRC" != "$SKILL_DEST" ]; then
  rm -rf "$SKILL_DEST"
  cp -r "$SKILL_SRC" "$SKILL_DEST"
fi
info "skill installed: $SKILL_DEST"

if [ "$SKIP_DEPS" -eq 0 ] && ! python3 -c 'import boto3,httpx,mcp' >/dev/null 2>&1; then
  echo "warning: install dependencies with: python3 -m pip install -r $SKILL_DEST/mcp/requirements.txt" >&2
fi

ENV_JSON="$(python3 - "$AWS_REGION" "$AWS_PROFILE" "$MODEL_NAME" "$INSTANCE_TYPE" "$SERVER_NAME" <<'PY'
import json, sys
region, profile, model, instance_type, server_name = sys.argv[1:]
# MCP_SERVER_NAME keeps the name server.py advertises equal to the key it is
# registered under; they have to agree or /mcp and the tool prefix disagree.
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
  python3 - "$TARGET_DIR/.mcp.json" "$SERVER_NAME" "$SKILL_STEM" "$ENV_JSON" <<'PY'
import json, sys
path, name, stem, env = sys.argv[1:]
try:
    with open(path) as handle:
        data = json.load(handle)
except FileNotFoundError:
    data = {}
data.setdefault("mcpServers", {})[name] = {
    "command": "python3",
    "args": [f".claude/skills/{stem}/mcp/server.py"],
    "env": json.loads(env),
}
with open(path, "w") as handle:
    json.dump(data, handle, indent=2)
    handle.write("\n")
PY
fi

info "registered MCP server '$SERVER_NAME' for AWS region $AWS_REGION"
echo "Run 'aws sts get-caller-identity' to verify credentials, then restart Claude Code."
