#!/usr/bin/env bash
# Create a non-secret local environment template for AWS Trainium (trn1) development.
set -euo pipefail
cat >.env <<'EOF'
AWS_REGION=us-east-2
MODEL_NAME=google/gemma-4-E2B-it
INSTANCE_TYPE=trn1.2xlarge
SERVICE_NAME=tpu-vllm-trn1-2b
SERVE_PORT=8000
EOF
chmod 600 .env
echo "Wrote .env. Add AWS_PROFILE if you do not use environment or role credentials."
