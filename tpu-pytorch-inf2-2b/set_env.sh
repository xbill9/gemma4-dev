#!/usr/bin/env bash
# Create a non-secret local environment template for AWS Inf2 development.
set -euo pipefail
cat >.env <<'EOF'
AWS_REGION=us-east-1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
INSTANCE_TYPE=inf2.xlarge
SERVICE_NAME=vllm-inf2
HF_SECRET_ID=vllm/hf-token
VLLM_PORT=8000
EOF
chmod 600 .env
echo "Wrote .env. Add AWS_PROFILE if you do not use environment or role credentials."
