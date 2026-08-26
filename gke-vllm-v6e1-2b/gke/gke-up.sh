#!/usr/bin/env bash
# Create the minimal GKE deployment this rig is named for: one zonal cluster, one
# single-host TPU v6e-1 node pool, one node, one chip.
#
# Idempotent — re-running against an existing cluster or node pool skips creation.
# Nothing here talks to `gcloud compute instances`: on this rig the accelerator is a
# property of a NODE POOL, and the node is created by the cluster, not by us.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[ -f tpu.env ] && set -a && . ./tpu.env && set +a

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT in tpu.env}"
LOCATION="${GKE_LOCATION:?set GKE_LOCATION in tpu.env}"
CLUSTER="${GKE_CLUSTER_NAME:?set GKE_CLUSTER_NAME in tpu.env}"
POOL="${GKE_NODE_POOL:?set GKE_NODE_POOL in tpu.env}"
MACHINE_TYPE="${MACHINE_TYPE:?set MACHINE_TYPE in tpu.env}"
TPU_TOPOLOGY="${TPU_TOPOLOGY:-1x1}"
NUM_NODES="${GKE_NUM_NODES:-1}"
SYSTEM_MACHINE_TYPE="${GKE_SYSTEM_MACHINE_TYPE:-e2-standard-4}"
SYSTEM_DISK_GB="${GKE_SYSTEM_DISK_SIZE_GB:-50}"
NODE_DISK_GB="${GKE_NODE_DISK_SIZE_GB:-200}"
RELEASE_CHANNEL="${GKE_RELEASE_CHANNEL:-rapid}"
PROVISIONING="${GKE_NODE_PROVISIONING:-ondemand}"

# Same four names as server.py's PROVISIONING_MODELS, and the same flags as
# _node_pool_provisioning_flags. Keep the two in step — one vocabulary per rig.
case "$PROVISIONING" in
  on-demand) POOL_PROVISIONING_FLAGS=(--num-nodes="$NUM_NODES") ;;
  spot)      POOL_PROVISIONING_FLAGS=(--spot --num-nodes="$NUM_NODES") ;;
  flex-start) POOL_PROVISIONING_FLAGS=(--flex-start --enable-autoscaling --num-nodes=0 --total-min-nodes=0 --total-max-nodes="$NUM_NODES" --location-policy=ANY --reservation-affinity=none --no-enable-autorepair) ;;
  reservation-bound)
    [ -n "${RESERVATION_NAME:-}" ] || { echo "❌ reservation-bound needs RESERVATION_NAME in tpu.env" >&2; exit 2; }
    POOL_PROVISIONING_FLAGS=(--reservation-affinity=specific --reservation="$RESERVATION_NAME" --num-nodes="$NUM_NODES") ;;
  *) echo "❌ GKE_NODE_PROVISIONING must be on-demand|spot|flex-start|reservation-bound, got '$PROVISIONING'" >&2; exit 2 ;;
esac

echo "🔧 project=$PROJECT_ID location=$LOCATION cluster=$CLUSTER"
echo "🔧 node pool=$POOL machine=$MACHINE_TYPE topology=$TPU_TOPOLOGY nodes=$NUM_NODES model=$PROVISIONING"

# ── 1. The cluster ────────────────────────────────────────────────────────────
# A small default pool exists to run the system workloads (kube-dns, metrics-server).
# Keeping them off the TPU node is the point: a v6e node is billed by the chip and
# should not be kept alive by CoreDNS after the model pod is gone.
if gcloud container clusters describe "$CLUSTER" --project="$PROJECT_ID" --location="$LOCATION" >/dev/null 2>&1; then
  echo "✅ cluster $CLUSTER already exists"
else
  echo "🚀 creating cluster $CLUSTER ($RELEASE_CHANNEL channel)..."
  gcloud container clusters create "$CLUSTER" \
    --project="$PROJECT_ID" \
    --location="$LOCATION" \
    --release-channel="$RELEASE_CHANNEL" \
    --num-nodes=1 \
    --machine-type="$SYSTEM_MACHINE_TYPE" \
    --disk-size="$SYSTEM_DISK_GB" \
    --labels=rig="$(basename "$PWD")"
fi

# ── 2. The TPU node pool ──────────────────────────────────────────────────────
# The machine type alone makes this a TPU pool: ct6e-standard-1t carries exactly one
# v6e chip, which is why TENSOR_PARALLEL_SIZE stays 1.
#
# DO NOT pass --tpu-topology for a single-host slice. Verified 2026-08-25: this rig's
# first node-pool create sent --tpu-topology=1x1 and the API refused it outright —
#   TPU topology can't be specified with single-host TPU slice pool
# The flag belongs to MULTI-host slices, where it describes how several nodes are wired
# into one slice. ct6e-standard-1t/4t/8t at one node each are single-host; a 1x1
# "topology" is not a smaller version of that flag, it is the absence of it.
if gcloud container node-pools describe "$POOL" --cluster="$CLUSTER" --project="$PROJECT_ID" --location="$LOCATION" >/dev/null 2>&1; then
  echo "✅ node pool $POOL already exists"
else
  # Set GKE_TPU_TOPOLOGY only for a multi-host slice (e.g. 4x4). Empty = single-host.
  TOPOLOGY_FLAGS=()
  [ -n "${GKE_TPU_TOPOLOGY:-}" ] && TOPOLOGY_FLAGS=(--tpu-topology="$GKE_TPU_TOPOLOGY" --placement-type=COMPACT)
  echo "🚀 creating TPU node pool $POOL ($MACHINE_TYPE, single-host${GKE_TPU_TOPOLOGY:+, topology $GKE_TPU_TOPOLOGY})..."
  gcloud container node-pools create "$POOL" \
    --project="$PROJECT_ID" \
    --location="$LOCATION" \
    --cluster="$CLUSTER" \
    --node-locations="$LOCATION" \
    --machine-type="$MACHINE_TYPE" \
    "${TOPOLOGY_FLAGS[@]}" \
    --disk-size="$NODE_DISK_GB" \
    "${POOL_PROVISIONING_FLAGS[@]}"
fi

echo
echo "🔍 nodes:"
gcloud container clusters describe "$CLUSTER" --project="$PROJECT_ID" --location="$LOCATION" \
  --format='table(name, currentNodeCount, status)'
echo
echo "Next: make gke-deploy   # applies the vLLM Deployment + Service to this cluster"
