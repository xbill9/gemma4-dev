#!/bin/bash
exec > >(tee /var/log/sweep-startup.log > /dev/console) 2>&1
set -ex
echo "Starting sweep VM bootstrap..."
for i in $(seq 1 30); do ping -c 1 8.8.8.8 && break; sleep 5; done
if ! command -v docker >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y && apt-get install -y docker.io
  systemctl enable --now docker
fi
for i in $(seq 1 5); do docker pull vllm/vllm-tpu:nightly && break; sleep 20; done
echo "SWEEP-VM-READY"
