#!/bin/bash
# Writes a .env for the Gemini CLI / ADK integration and points gcloud at the
# saved project. Reads the project ID from ~/project_id.txt (written by init.sh)
# and the Gemini API key from ~/gemini.key (prompting once if missing).
set -euo pipefail

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "Error: No active gcloud account found." >&2
    echo "Please run 'gcloud auth login' and try again." >&2
    exit 1
fi

# Get current project
PROJECT_ID=""
if [ -f "$HOME/project_id.txt" ]; then
    PROJECT_ID=$(tr -d '[:space:]' < "$HOME/project_id.txt")
fi

if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" == "(unset)" ]; then
    echo "Error: no project ID found in \$HOME/project_id.txt." >&2
    echo "Run ./init.sh first, or 'gcloud config set project [PROJECT_ID]'." >&2
    exit 1
fi

gcloud config set project "$PROJECT_ID"

if [ -f "$HOME/gemini.key" ]; then
    GOOGLE_API_KEY=$(tr -d '[:space:]' < "$HOME/gemini.key")
else
    read -r -p "Enter Gemini KEY: " GOOGLE_API_KEY
    [ -n "$GOOGLE_API_KEY" ] || { echo "Error: no key provided." >&2; exit 1; }
    (umask 077; echo "$GOOGLE_API_KEY" > "$HOME/gemini.key")
fi

MCP_SERVER_URL=https://mcp-https-python-wgcq55zbfq-rj.a.run.app/mcp

cat <<EOF > .env
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GOOGLE_CLOUD_LOCATION=europe-west4
GOOGLE_CLOUD_REGION=europe-west4
GOOGLE_CLOUD_ZONE=europe-west4-a
MODEL=google/gemma-4-E4B-it
MODEL_NAME=google/gemma-4-E4B-it
GENAI_MODEL="gemini-2.5-flash"
GOOGLE_API_KEY=$GOOGLE_API_KEY
GEMINI_API_KEY=$GOOGLE_API_KEY
MCP_SERVER_URL=$MCP_SERVER_URL
ACCELERATOR_TYPE=v6e-1
TENSOR_PARALLEL_SIZE=1
EOF
chmod 600 .env

echo "Wrote .env (API keys masked below):"
grep -v -E '^(GOOGLE_API_KEY|GEMINI_API_KEY)=' .env
echo "GOOGLE_API_KEY=***  GEMINI_API_KEY=***"

echo "Cloud Login"
gcloud auth list

echo "Config List"
gcloud config list

echo "ADK Version"
adk --version
