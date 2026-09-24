#!/bin/bash
# Startup script for the jev-tpu v6e-1 VM: Docker, the HF token from Secret Manager, the
# vLLM TPU image, and an E2B server on :8000 for the smoke read. The measurement itself is
# driven by tpu/run.sh, copied in afterwards. Logs go to the serial console.
exec > >(tee -a /var/log/jev-boot.log /dev/console) 2>&1
set -u
log() { echo "[jev-boot $(date -u +%FT%TZ)] $*"; }
PROJECT=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/project/project-id)

if ! command -v docker >/dev/null; then
  log "installing docker"
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io >/dev/null
fi

log "fetching hf-token"
for i in $(seq 1 60); do
  AT=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token | python3 -c 'import json,sys;print(json.load(sys.stdin)["access_token"])')
  HF_TOKEN=$(curl -s -H "Authorization: Bearer $AT" "https://secretmanager.googleapis.com/v1/projects/$PROJECT/secrets/hf-token/versions/latest:access" | python3 -c 'import json,sys,base64;print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode())' 2>/dev/null)
  [ -n "$HF_TOKEN" ] && break
  sleep 30
done
[ -n "$HF_TOKEN" ] || { log "no hf-token"; exit 1; }
install -m 600 /dev/null /root/hf_token && printf '%s' "$HF_TOKEN" > /root/hf_token

log "pulling vllm/vllm-tpu:nightly"
docker pull -q vllm/vllm-tpu:nightly
log "image $(docker inspect --format '{{index .RepoDigests 0}}' vllm/vllm-tpu:nightly)"
log "READY-FOR-RUN"
