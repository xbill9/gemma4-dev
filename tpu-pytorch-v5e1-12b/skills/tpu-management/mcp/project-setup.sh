#!/usr/bin/env bash
# One-command setup of the tpu-management skill + tpu-devops MCP server for a project.
#
# Usage:
#   ./project-setup.sh [TARGET_DIR] [options]
#
#   TARGET_DIR              Project to set up (default: current directory).
#                           Installs the skill into TARGET_DIR/.claude/skills/tpu-management
#                           and registers the MCP server in TARGET_DIR/.mcp.json.
#
# Options:
#   --global                Install the skill to ~/.claude/skills and register the MCP
#                           server at user scope (all projects) instead of one project.
#   --project ID            GOOGLE_CLOUD_PROJECT for the agent (default: gcloud config,
#                           else the server.py built-in default).
#   --model NAME            MODEL_NAME            (default: google/gemma-4-31B-it)
#   --accelerator TYPE      ACCELERATOR_TYPE      (default: v6e-8)
#   --tp N                  TENSOR_PARALLEL_SIZE  (default: 8)
#   --server-name NAME      MCP server name       (default: tpu-devops)
#   --skip-deps             Don't check Python dependencies.
#   -h, --help              Show this help.
#
# The script is idempotent: re-running refreshes the skill copy and updates the
# existing .mcp.json entry in place, leaving other servers untouched.
#
# Works from either checkout layout:
#   - the skill repo root (uses .claude/skills/tpu-management), or
#   - inside an unzipped skill bundle (mcp/project-setup.sh next to mcp/server.py).

set -euo pipefail

err()  { echo "error: $*" >&2; exit 1; }
info() { echo "==> $*"; }

# --- Locate the skill source relative to this script ---------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.claude/skills/tpu-management/SKILL.md" ]; then
  SKILL_SRC="$SCRIPT_DIR/.claude/skills/tpu-management"     # repo root
elif [ -f "$SCRIPT_DIR/../SKILL.md" ]; then
  SKILL_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"                 # skill-bundle mcp/ dir
else
  err "cannot find the tpu-management skill next to $SCRIPT_DIR"
fi

# --- Parse arguments -----------------------------------------------------------------
TARGET_DIR=""
GLOBAL=0
SKIP_DEPS=0
GCP_PROJECT=""
MODEL_NAME="google/gemma-4-31B-it"
ACCELERATOR_TYPE="v6e-8"
TENSOR_PARALLEL_SIZE="8"
SERVER_NAME="tpu-devops"

while [ $# -gt 0 ]; do
  case "$1" in
    --global)       GLOBAL=1 ;;
    --project)      GCP_PROJECT="$2"; shift ;;
    --model)        MODEL_NAME="$2"; shift ;;
    --accelerator)  ACCELERATOR_TYPE="$2"; shift ;;
    --tp)           TENSOR_PARALLEL_SIZE="$2"; shift ;;
    --server-name)  SERVER_NAME="$2"; shift ;;
    --skip-deps)    SKIP_DEPS=1 ;;
    -h|--help)      sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)             err "unknown option: $1 (see --help)" ;;
    *)              [ -n "$TARGET_DIR" ] && err "unexpected argument: $1"
                    TARGET_DIR="$1" ;;
  esac
  shift
done

if [ "$GLOBAL" -eq 1 ] && [ -n "$TARGET_DIR" ]; then
  err "--global and TARGET_DIR are mutually exclusive"
fi
TARGET_DIR="${TARGET_DIR:-$PWD}"
[ -d "$TARGET_DIR" ] || err "target directory does not exist: $TARGET_DIR"
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"

if [ -z "$GCP_PROJECT" ] && command -v gcloud >/dev/null 2>&1; then
  GCP_PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
  [ "$GCP_PROJECT" = "(unset)" ] && GCP_PROJECT=""
fi

# --- Install the skill copy ----------------------------------------------------------
if [ "$GLOBAL" -eq 1 ]; then
  SKILL_DEST="$HOME/.claude/skills/tpu-management"
else
  SKILL_DEST="$TARGET_DIR/.claude/skills/tpu-management"
fi

if [ "$SKILL_SRC" != "$SKILL_DEST" ]; then
  mkdir -p "$(dirname "$SKILL_DEST")"
  rm -rf "$SKILL_DEST"
  cp -r "$SKILL_SRC" "$SKILL_DEST"
  info "skill installed: $SKILL_DEST"
else
  info "skill source and destination are the same, skipping copy"
fi

# --- Python dependencies -------------------------------------------------------------
# The server runs on the system python3. If the imports don't resolve, warn with the
# exact pip command — never create a venv.
PYTHON_BIN="python3"
DEPS_CHECK='import mcp, httpx, openai, google.cloud.secretmanager'

if [ "$SKIP_DEPS" -eq 0 ]; then
  if python3 -c "$DEPS_CHECK" >/dev/null 2>&1; then
    info "python dependencies OK (system python3)"
  else
    echo "warning: python3 is missing server dependencies; install them with:" >&2
    echo "    pip install -r $SKILL_DEST/mcp/requirements.txt" >&2
  fi
else
  info "skipping dependency check (--skip-deps)"
fi

# --- Register the MCP server ---------------------------------------------------------
build_env_json() {
  # Emits the "env" object for the server entry; omits GOOGLE_CLOUD_PROJECT when unset
  # so server.py falls back to its built-in default.
  "$PYTHON_BIN" - "$GCP_PROJECT" "$MODEL_NAME" "$ACCELERATOR_TYPE" "$TENSOR_PARALLEL_SIZE" <<'EOF'
import json, sys
project, model, accel, tp = sys.argv[1:5]
env = {"MODEL_NAME": model, "ACCELERATOR_TYPE": accel, "TENSOR_PARALLEL_SIZE": tp}
if project:
    env["GOOGLE_CLOUD_PROJECT"] = project
print(json.dumps(env))
EOF
}

ENV_JSON="$(build_env_json)"

if [ "$GLOBAL" -eq 1 ]; then
  command -v claude >/dev/null 2>&1 || err "--global needs the claude CLI on PATH"
  SERVER_JSON="$("$PYTHON_BIN" -c 'import json,sys; print(json.dumps({
      "command": sys.argv[1],
      "args": [sys.argv[2]],
      "env": json.loads(sys.argv[3])}))' \
      "$PYTHON_BIN" "$SKILL_DEST/mcp/server.py" "$ENV_JSON")"
  claude mcp remove --scope user "$SERVER_NAME" >/dev/null 2>&1 || true
  claude mcp add-json --scope user "$SERVER_NAME" "$SERVER_JSON"
  info "MCP server '$SERVER_NAME' registered at user scope"
else
  MCP_JSON="$TARGET_DIR/.mcp.json"
  "$PYTHON_BIN" - "$MCP_JSON" "$SERVER_NAME" "$PYTHON_BIN" "$ENV_JSON" <<'EOF'
import json, sys
path, name, python_bin, env_json = sys.argv[1:5]
try:
    with open(path) as f:
        config = json.load(f)
except FileNotFoundError:
    config = {}
config.setdefault("mcpServers", {})[name] = {
    "command": python_bin,
    "args": [".claude/skills/tpu-management/mcp/server.py"],
    "env": json.loads(env_json),
}
with open(path, "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
print(f"==> MCP server '{name}' registered in {path}")
EOF
fi

# --- Summary -------------------------------------------------------------------------
echo
echo "Done. Configuration:"
echo "  skill:                $SKILL_DEST"
echo "  python:               $PYTHON_BIN"
echo "  GOOGLE_CLOUD_PROJECT: ${GCP_PROJECT:-<server.py default>}"
echo "  MODEL_NAME:           $MODEL_NAME"
echo "  ACCELERATOR_TYPE:     $ACCELERATOR_TYPE"
echo "  TENSOR_PARALLEL_SIZE: $TENSOR_PARALLEL_SIZE"
echo
echo "Next steps:"
if [ "$GLOBAL" -eq 1 ]; then
  echo "  1. Restart Claude Code; /mcp should list '$SERVER_NAME' in every project."
else
  echo "  1. Restart Claude Code in $TARGET_DIR; approve the project MCP server"
  echo "     when prompted, then /mcp should list '$SERVER_NAME'."
fi
echo "  2. Ensure gcloud is authenticated (gcloud auth login + application-default login)."
echo "  3. Store a Hugging Face token via the agent's save_hf_token tool (Secret Manager"
echo "     secret 'hf-token') before creating any TPU resources."
