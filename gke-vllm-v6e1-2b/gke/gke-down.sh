#!/usr/bin/env bash
# Tear down what gke-up.sh created.
#
#   gke/gke-down.sh --pool-only   delete just the TPU node pool (stops the chip bill,
#                                 keeps the cluster and its credentials)
#   gke/gke-down.sh               delete the whole cluster
#
# Asks before doing either unless FORCE=1. Deleting the pool releases the v6e chip, and
# capacity can take a long time to come back — the same caution the sibling rigs carry
# about flex-start VMs applies here.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[ -f tpu.env ] && set -a && . ./tpu.env && set +a

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:?}"
LOCATION="${GKE_LOCATION:?}"
CLUSTER="${GKE_CLUSTER_NAME:?}"
POOL="${GKE_NODE_POOL:?}"

POOL_ONLY=0
[ "${1:-}" = "--pool-only" ] && POOL_ONLY=1

if [ "$POOL_ONLY" = 1 ]; then
  TARGET="node pool $POOL in cluster $CLUSTER"
else
  TARGET="cluster $CLUSTER (and every node pool in it)"
fi

if [ "${FORCE:-0}" != "1" ]; then
  read -r -p "🗑️  Delete $TARGET in $LOCATION? [y/N] " reply
  case "$reply" in [yY]*) ;; *) echo "aborted"; exit 1 ;; esac
fi

if [ "$POOL_ONLY" = 1 ]; then
  gcloud container node-pools delete "$POOL" \
    --cluster="$CLUSTER" --project="$PROJECT_ID" --location="$LOCATION" --quiet
else
  gcloud container clusters delete "$CLUSTER" \
    --project="$PROJECT_ID" --location="$LOCATION" --quiet
fi
echo "✅ deleted $TARGET"
