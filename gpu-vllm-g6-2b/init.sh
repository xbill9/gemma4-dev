#!/usr/bin/env bash
# Validate AWS access, install dependencies, and register the G5g MCP server.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"

command -v aws >/dev/null 2>&1 || { echo "error: AWS CLI is required" >&2; exit 1; }
aws sts get-caller-identity --region "$REGION" >/dev/null
echo "AWS identity OK in region $REGION"
python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"
"$SCRIPT_DIR/project-setup.sh" "$SCRIPT_DIR" --region "$REGION" --skip-deps
echo "Setup complete. Restart Claude Code and verify $(basename "$SCRIPT_DIR") under /mcp."
