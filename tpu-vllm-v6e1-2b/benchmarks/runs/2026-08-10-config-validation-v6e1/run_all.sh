#!/usr/bin/env bash
# Drive the whole config-validation run against the live v6e-1 node.
#
# Order matters: the allocation is verified BEFORE any throughput cell runs, because a
# throughput number measured against a config that did not take effect is worse than no
# number at all — it looks like evidence. If verify_allocation.py fails, this stops.
#
# Usage: ./run_all.sh [resource_id] [zone]
set -euo pipefail

RESOURCE_ID="${1:-tpu-vllm-v6e1-2b}"
ZONE="${2:-us-east5-b}"
# Wall-clock budget for the sweep itself. Flex-start self-terminates 4h after ACTIVE, so this
# must leave room for boot (~15 min of kernel compile) plus margin — a node that vanishes
# mid-request loses the JSON entirely, whereas the budget guard writes what it has.
SWEEP_BUDGET_SECONDS="${3:-9000}"
PROJECT="${GOOGLE_CLOUD_PROJECT:-aisprint-491218}"
NODE="${RESOURCE_ID}-node"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="vllm-gemma4"

mkdir -p "$HERE/results" "$HERE/logs"

ssh_vm() {
  gcloud compute tpus tpu-vm ssh "$NODE" --zone="$ZONE" --project="$PROJECT" \
    --worker=0 --command="$1" 2>&1
}

echo "=== 1. node reachable? ==="
ssh_vm "hostname && uptime" | tail -3

echo
echo "=== 2. wait for vLLM to answer /v1/models (compile is ~700s cold) ==="
for i in $(seq 1 120); do
  if ssh_vm "curl -sf -m 5 http://localhost:8000/v1/models >/dev/null && echo UP" | grep -q UP; then
    echo "vLLM is answering after ~$((i * 30))s"
    break
  fi
  echo "  [$i/120] not yet..."
  sleep 30
done

echo
echo "=== 3. capture the boot log ==="
ssh_vm "sudo docker logs $CONTAINER 2>&1 | tail -4000" > "$HERE/logs/boot.log"
wc -l "$HERE/logs/boot.log"

echo
echo "=== 4. verify the allocation BEFORE benchmarking ==="
python3 "$HERE/verify_allocation.py" --log "$HERE/logs/boot.log" --json "$HERE/results/allocation.json"

echo
echo "=== 5. throughput cells ==="
gcloud compute tpus tpu-vm scp "$HERE/run_cells.py" "$NODE:~/run_cells.py" \
  --zone="$ZONE" --project="$PROJECT" --worker=0
ssh_vm "python3 ~/run_cells.py --output ~/cells_v6e1.json --container $CONTAINER \
  --deadline-seconds $SWEEP_BUDGET_SECONDS" | tee "$HERE/logs/cells_v6e1.log"
gcloud compute tpus tpu-vm scp "$NODE:~/cells_v6e1.json" "$HERE/results/cells_v6e1.json" \
  --zone="$ZONE" --project="$PROJECT" --worker=0

echo
echo "=== done. results in $HERE/results/ ==="
