#!/bin/bash
# Startup script for the GPU cross-check: the code bundle (metadata jev-code) into /opt/jev-tpu,
# Docker with the NVIDIA runtime, then gpu/run_gpu.sh with metadata jev-run as the prefix.
exec > >(tee -a /var/log/jev-boot.log /dev/console) 2>&1
set -u
md() { curl -s -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/$1"; }
mkdir -p /opt/jev-tpu/results
gcloud storage cp -q "gs://aisprint-491218-bucket/jev-tpu-31b/inputs/$(md instance/attributes/jev-code)" /tmp/code.tgz && tar xzf /tmp/code.tgz -C /opt/jev-tpu
if ! command -v docker >/dev/null; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io >/dev/null
fi
if ! docker info 2>/dev/null | grep -qi nvidia; then
  command -v nvidia-ctk >/dev/null || {
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-container-toolkit >/dev/null
  }
  nvidia-ctk runtime configure --runtime=docker >/dev/null && systemctl restart docker
fi
bash /opt/jev-tpu/gpu/run_gpu.sh "$(md instance/attributes/jev-run)"
