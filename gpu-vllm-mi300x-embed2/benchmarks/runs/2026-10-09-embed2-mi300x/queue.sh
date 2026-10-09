#!/bin/bash
# Waits for the AITER chain (gpu-vllm-mi300x-2b/benchmarks/runs/2026-10-09-aiter-mi300x) and the CPU
# reference to finish, frees port 8000, then runs embed_run.py on the same droplet.
set -u
cd "$(dirname "$0")/../../.."   # gpu-vllm-mi300x-embed2
IP=129.212.191.240
while pgrep -f "chain-aiter.sh" >/dev/null || pgrep -f "python3 reference.py" >/dev/null; do sleep 30; done
ssh -i ~/amd -o BatchMode=yes root@$IP 'docker rm -f vllm >/dev/null 2>&1; docker ps --format "{{.Names}}"'
python3 -u embed_run.py --ip $IP --run-id 2026-10-09-embed2-mi300x
