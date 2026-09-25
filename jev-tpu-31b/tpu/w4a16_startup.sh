#!/bin/bash
# Startup script for the unattended W4A16 probe on one v6e chip: Docker, the HF token from
# Secret Manager, the pinned vLLM TPU image, then w4a16_probe.sh. Metadata: jev-code names
# the code tarball, jev-run the run prefix, jev-keep=1 leaves the VM up (31B still serving)
# instead of deleting it at the end.
exec > >(tee -a /var/log/jev-boot.log /dev/console) 2>&1
set -u
log() { echo "[w4a16-boot $(date -u +%FT%TZ)] $*"; }
md() { curl -s -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/$1"; }
PROJECT=$(md project/project-id)
CODE=$(md instance/attributes/jev-code)

mkdir -p /opt/w4a16
gcloud storage cp -q "gs://aisprint-491218-bucket/jev-tpu-31b/inputs/$CODE" /tmp/code.tgz && tar xzf /tmp/code.tgz -C /opt/w4a16

if ! command -v docker >/dev/null; then
  log "installing docker"
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io >/dev/null
fi

log "fetching hf-token"
for i in $(seq 1 60); do
  AT=$(md instance/service-accounts/default/token | python3 -c 'import json,sys;print(json.load(sys.stdin)["access_token"])')
  HF_TOKEN=$(curl -s -H "Authorization: Bearer $AT" "https://secretmanager.googleapis.com/v1/projects/$PROJECT/secrets/hf-token/versions/latest:access" | python3 -c 'import json,sys,base64;print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode())' 2>/dev/null)
  [ -n "$HF_TOKEN" ] && break
  sleep 30
done
[ -n "$HF_TOKEN" ] || { log "no hf-token"; exit 1; }
install -m 600 /dev/null /root/hf_token && printf '%s' "$HF_TOKEN" > /root/hf_token

bash /opt/w4a16/tpu/w4a16_probe.sh
