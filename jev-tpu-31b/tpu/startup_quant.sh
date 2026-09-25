#!/bin/bash
# Startup script for the unattended 4-bit accuracy read: the code bundle (metadata jev-code,
# under gs://.../jev-tpu-31b/inputs/) into /opt/jev-tpu, Docker, patch, the HF token from
# Secret Manager, then run_quant.sh with metadata jev-run as the prefix.
exec > >(tee -a /var/log/jev-boot.log /dev/console) 2>&1
set -u
md() { curl -s -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/$1"; }
PROJECT=$(md project/project-id)
mkdir -p /opt/jev-tpu/results
gcloud storage cp -q "gs://aisprint-491218-bucket/jev-tpu-31b/inputs/$(md instance/attributes/jev-code)" /tmp/code.tgz && tar xzf /tmp/code.tgz -C /opt/jev-tpu
command -v docker >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io >/dev/null; }
command -v patch >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq patch >/dev/null; }
for i in $(seq 1 60); do
  AT=$(md instance/service-accounts/default/token | python3 -c 'import json,sys;print(json.load(sys.stdin)["access_token"])')
  HF_TOKEN=$(curl -s -H "Authorization: Bearer $AT" "https://secretmanager.googleapis.com/v1/projects/$PROJECT/secrets/hf-token/versions/latest:access" | python3 -c 'import json,sys,base64;print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode())' 2>/dev/null)
  [ -n "$HF_TOKEN" ] && break; sleep 30
done
[ -n "$HF_TOKEN" ] || { echo "no hf-token"; exit 1; }
install -m 600 /dev/null /root/hf_token && printf '%s' "$HF_TOKEN" > /root/hf_token
bash /opt/jev-tpu/tpu/run_quant.sh "$(md instance/attributes/jev-run)"
