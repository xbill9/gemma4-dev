#!/usr/bin/env bash
# Export this rig's committed configuration into the current shell.
#
# tpu.env is the source of truth and is committed. This script only exports it —
# it does not generate it, and a real environment variable always wins (every
# assignment below is guarded).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/tpu.env"

[ -f "$ENV_FILE" ] || { echo "error: $ENV_FILE not found" >&2; exit 1; }

while IFS= read -r line; do
  case "$line" in
    ''|\#*) continue ;;
  esac
  key="${line%%=*}"
  value="${line#*=}"
  # Only set keys that are not already present in the environment.
  if [ -z "${!key:-}" ]; then
    export "$key=$value"
  fi
done < "$ENV_FILE"

echo "Exported $(grep -c '^[A-Z]' "$ENV_FILE") settings from tpu.env"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  INSTANCE_TYPE=$INSTANCE_TYPE  AWS_REGION=$AWS_REGION"
echo "  DTYPE=$DTYPE  (Turing has no bf16)"
