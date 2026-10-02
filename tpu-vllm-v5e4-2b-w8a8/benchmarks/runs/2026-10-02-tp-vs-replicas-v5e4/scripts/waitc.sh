#!/bin/bash
# waitc.sh <container> [<container>...] — wait until each container logs "Application startup complete" or exits
N=tpu-vllm-v5e4-2b-w8a8-node
g(){ gcloud compute tpus tpu-vm ssh $N --zone=us-west4-a --project=aisprint-491218 --worker=0 --quiet --command="$1" 2>/dev/null; }
for i in $(seq 1 40); do
  s=$(g "for c in $*; do echo \$c \$(sudo docker ps -a --filter name=^\$c\$ --format '{{.Status}}' | cut -d' ' -f1) \$(sudo docker logs \$c 2>&1 | grep -c 'Application startup complete'); done")
  echo "$s" | grep -q Exited && { echo "$(date -u +%T) EXITED: $s"; for c in $*; do g "sudo docker logs $c 2>&1 | grep -E 'Error|error' | tail -4" | cut -c1-300; done; exit 1; }
  [ $(echo "$s" | awk '$3>=1' | wc -l) -eq $# ] && { echo "$(date -u +%T) READY: $*"; exit 0; }
  sleep 30
done
echo TIMEOUT; echo "$s"; exit 2
