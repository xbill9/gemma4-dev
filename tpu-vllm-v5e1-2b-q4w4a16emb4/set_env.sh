#!/bin/bash

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "Error: No active gcloud account found."
    echo "Please run 'gcloud auth login' and try again."
fi

# Get current project
if [ -f "$HOME/project_id.txt" ]; then
    PROJECT_ID=$(cat "$HOME/project_id.txt")
fi

if [ "$PROJECT_ID" == "(unset)" ] || [ -z "$PROJECT_ID" ]; then
    echo "Warning: No gcloud project is currently set."
    echo "Run 'gcloud config set project [PROJECT_ID]' to configure it."
fi

gcloud config set project $PROJECT_ID


if [ -f "$HOME/gemini.key" ]; then
    GOOGLE_API_KEY=$(cat "$HOME/gemini.key")
else
    read -p "Enter Gemini KEY: " GOOGLE_API_KEY
    echo "$GOOGLE_API_KEY" > "$HOME/gemini.key"
fi

export MCP_SERVER_URL=https://mcp-https-python-wgcq55zbfq-rj.a.run.app/mcp

# Deployment parameters come from tpu.env so the zone/region/model are defined once.
if [ -f "$(dirname "${BASH_SOURCE[0]}")/tpu.env" ]; then
    source "$(dirname "${BASH_SOURCE[0]}")/tpu.env"
fi

cat <<EOF > .env
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_REGION
GOOGLE_CLOUD_REGION=$GOOGLE_CLOUD_REGION
GOOGLE_CLOUD_ZONE=$GOOGLE_CLOUD_ZONE
MODEL=$MODEL_NAME
MODEL_NAME=$MODEL_NAME
GENAI_MODEL="gemini-2.5-flash"
GOOGLE_API_KEY=$GOOGLE_API_KEY
GEMINI_API_KEY=$GOOGLE_API_KEY
MCP_SERVER_URL=$MCP_SERVER_URL
ACCELERATOR_TYPE=$ACCELERATOR_TYPE
TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE
EOF

echo "Sourcing Env"
source .env

echo "Current Environment"
cat .env

echo "Cloud Login"
gcloud auth list

echo "Config List"
gcloud config list

echo "ADK Version"
adk --version
