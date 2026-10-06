#!/bin/bash

# Simple environment configuration for Local Gemma 4 SRE Agent

cat <<EOF > .env
MODEL_NAME=google/gemma-4-E2B-it
LOCAL_DOCKER_IMAGE=ollama/ollama:latest
LOCAL_VLLM_PORT=8080
EOF

echo "Sourcing Env"
source .env

echo "Current Environment:"
cat .env
