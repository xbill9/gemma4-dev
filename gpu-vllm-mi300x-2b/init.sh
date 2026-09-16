#!/usr/bin/env bash
# First-run setup for gpu-vllm-mi300x-2b: install deps, refresh the skill and
# register the MCP server in this rig directory.
#
# Runs non-interactively on purpose — no `read` in the error path, so it is
# safe from a script or a CI step.
set -euo pipefail

RIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RIG_DIR"

echo "==> dependencies (system python3, never a virtualenv)"
python3 -m pip install -r requirements.txt || {
  echo "pip install failed. On Debian 13 try: apt-get install python3-httpx python3-dotenv" >&2
  echo "continuing — the rest of setup does not need them" >&2
}

echo "==> skill snapshots"
make skill

echo "==> registering the MCP server in $RIG_DIR"
./project-setup.sh "$RIG_DIR"

cat <<'NEXT'

Next:
  1. Put DIGITALOCEAN_ACCESS_TOKEN in .env (mode 0600) — never in tpu.env.
  2. Tag the MI300X droplet `gemma`, or set DROPLET_TAG in tpu.env.
  3. Ask the agent: list_droplets, then gpu_status, then serving_status.

If the card is missing, reboot the droplet once — a fresh one has no /dev/kfd
until it is rebooted, and nothing needs installing.
NEXT
