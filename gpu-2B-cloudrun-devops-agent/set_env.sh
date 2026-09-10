#!/bin/bash

# Environment configuration for the Cloud Run GPU (NVIDIA L4) Gemma 4 SRE agent.
#
# server.py never calls load_dotenv, so these must be exported into the shell that
# launches the MCP server. SOURCE this script — running it does nothing useful:
#
#   source ./set_env.sh
#
# Anything already set in your shell wins, so you can override per-session:
#
#   GOOGLE_CLOUD_LOCATION=us-central1 source ./set_env.sh
#
# Pass --resolve-endpoint to pin VLLM_BASE_URL to the deployed service URL.
# Left unset, server.py's discover_vllm_url() finds it via gcloud at call time.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "⚠️  This script must be sourced: source ./set_env.sh" >&2
    exit 1
fi

# Keep these in sync with the Makefile (PROJECT_ID / REGION / SERVICE_NAME / MODEL_PATH).
# All four are read by server.py; SERVICE_NAME becomes its DEFAULT_SERVICE_NAME.
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-aisprint-491218}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-us-east4}"
export SERVICE_NAME="${SERVICE_NAME:-gpu-2b-l4-devops-agent}"

# The model id vLLM actually serves — the container is started with
# --model=/mnt/models/<MODEL_PATH>, so that path is the name in the OpenAI API.
export MODEL_NAME="${MODEL_NAME:-/mnt/models/gemma-4-E2B-it}"

if [ "$1" = "--resolve-endpoint" ]; then
    echo "Resolving endpoint for $SERVICE_NAME in $GOOGLE_CLOUD_LOCATION..."
    RESOLVED_URL=$(gcloud run services describe "$SERVICE_NAME" \
        --project="$GOOGLE_CLOUD_PROJECT" \
        --region="$GOOGLE_CLOUD_LOCATION" \
        --format='value(status.url)' 2>/dev/null)
    if [ -n "$RESOLVED_URL" ]; then
        export VLLM_BASE_URL="$RESOLVED_URL"
    else
        echo "⚠️  Could not resolve the service URL — is it deployed? Falling back to gcloud discovery at call time." >&2
    fi
    unset RESOLVED_URL
fi

echo "Current Environment:"
echo "  GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"
echo "  GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_LOCATION"
echo "  SERVICE_NAME=$SERVICE_NAME"
echo "  MODEL_NAME=$MODEL_NAME"
echo "  VLLM_BASE_URL=${VLLM_BASE_URL:-<unset — discovered via gcloud>}"
echo
echo "Cloud Run here is --no-allow-unauthenticated. If calls fail, run: source ./set_adc.sh"
