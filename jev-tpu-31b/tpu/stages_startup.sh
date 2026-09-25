#!/bin/bash
# Startup script for an unattended probe_stages.sh run on a fresh v6e-1 VM. Metadata:
# jev-code (tarball under gs://.../jev-tpu-31b/inputs/), jev-run (run prefix), jev-patches
# (comma-separated diffs), jev-stages (space-separated stage specs). Deletes the VM at the end.
exec > >(tee -a /var/log/jev-boot.log /dev/console) 2>&1
set -u
md() { curl -s -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/$1"; }
PROJECT=$(md project/project-id); ZONE=$(md instance/zone | awk -F/ '{print $NF}')
mkdir -p /opt/w4a16
gcloud storage cp -q "gs://aisprint-491218-bucket/jev-tpu-31b/inputs/$(md instance/attributes/jev-code)" /tmp/code.tgz && tar xzf /tmp/code.tgz -C /opt/w4a16
command -v docker >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io >/dev/null; }
command -v patch >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq patch >/dev/null; }
for i in $(seq 1 60); do
  AT=$(md instance/service-accounts/default/token | python3 -c 'import json,sys;print(json.load(sys.stdin)["access_token"])')
  HF_TOKEN=$(curl -s -H "Authorization: Bearer $AT" "https://secretmanager.googleapis.com/v1/projects/$PROJECT/secrets/hf-token/versions/latest:access" | python3 -c 'import json,sys,base64;print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode())' 2>/dev/null)
  [ -n "$HF_TOKEN" ] && break; sleep 30
done
install -m 600 /dev/null /root/hf_token && printf '%s' "$HF_TOKEN" > /root/hf_token
# Optional metadata jev-boot-timeout / jev-xla-cache feed serve.sh's BOOT_TIMEOUT / XLA_CACHE_DIR.
BT=$(md instance/attributes/jev-boot-timeout); [ -n "$BT" ] && [ "${BT:0:1}" != "<" ] && export BOOT_TIMEOUT=$BT
XC=$(md instance/attributes/jev-xla-cache); [ -n "$XC" ] && [ "${XC:0:1}" != "<" ] && export XLA_CACHE_DIR=$XC
bash /opt/w4a16/tpu/probe_stages.sh "$(md instance/attributes/jev-run)" "$(md instance/attributes/jev-patches)" $(md instance/attributes/jev-stages)
gcloud compute instances delete "$(hostname)" --zone "$ZONE" --quiet
