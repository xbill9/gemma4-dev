#!/usr/bin/env bash
# Request Compute Engine TPU quota for a region.
#
# Lives at the monorepo root because quota is a property of the PROJECT, not of a rig.
# Every rig here provisions into the same project and competes for the same pools.
#
# THIS REQUESTS COMPUTE ENGINE QUOTA, NOT CLOUD TPU API QUOTA. The two control planes meter
# against disjoint pools and holding chips on one buys nothing on the other. Verified
# 2026-08-11: this project held 512 v6e chips in us-east5 under the Cloud TPU API and
# nothing at all in us-east5 under Compute Engine, which is what made gce-vllm-v6e1-2b's
# original default zone unusable. See @HARDWARE.md for the control-plane table.
#
# Quota ids differ per generation in a way no analogy predicts, so they are mapped
# explicitly below rather than derived. All read off `gcloud quotas info` on 2026-08-11:
#
#   v6e  TPUS-PER-TPU-FAMILY-per-project-region  (region, tpu_family=CT6E)  <- STANDARD, and
#                                                       the documented FLEX_START fallback
#        PREEMPTIBLE-TPU-V6E-per-project-region  (region)  <- SPOT, and FLEX_START first
#   v5p  TPU-V5P-per-project-region              (region)
#        PREEMPTIBLE-TPU-V5P-per-project-region  (region)
#
# v6e has NO dedicated non-preemptible id — no TPU-V6E-per-project-{region,zone} exists —
# so it falls back to the generic family quota. v5p publishes both halves itself and is not
# in the family quota at all (its families are CT3, CT3P, CT6E only). v5e is deliberately
# unsupported here: it has no Compute Engine create path, so CE quota for it buys nothing.
#
# WHICH ID GOVERNS WHICH MODEL. Google's Compute Engine provisioning-models page states:
#   "When you create a Flex-start VM, preemptible quota is consumed. If your project lacks
#    preemptible quota, then standard quota is consumed."
#   https://docs.cloud.google.com/compute/docs/instances/provisioning-models
# So FLEX_START draws on the PREEMPTIBLE id first and falls back to the family/standard one —
# counterintuitive, because flex-start is not preemptible in behaviour. SPOT uses the
# preemptible id; STANDARD (on-demand) uses the family quota. This header has been wrong
# twice: it first asserted the opposite mapping and called the question unresolved, then
# omitted the fallback. Both halves matter.
#
# BOTH HALVES ARE STILL REQUESTED. Flex-start needs the preemptible id, on-demand needs the
# other, and most rigs here use both models at some point. Asking for both is free.
#
# Do not try to settle quota questions with a create. A FLEX_START create short of quota is
# ACCEPTED and queues rather than erroring, and a capacity stockout produces an identical
# PENDING — verified 2026-08-11 in us-east5-b (no quota) and us-central1-a (no capacity).
# A SPOT create is the useful probe: it fails fast with an explicit `reason: stockout`.
#
# Non-destructive: a QuotaPreference is a request. Decisions observed 2026-08-11 were
# IMMEDIATE in both directions — 3 approved and 4 denied within seconds of submission, all
# automated (quotaConfig.stateDetail carries "Quota request approved to N" / "Quota request
# denied"). Denials clustered in the busier regions (us-central1, us-east4, us-west1),
# consistent with capacity pressure rather than policy, though that is inference only.
#
# Usage:
#   REGION=us-central1 ./request-quota.sh              # v6e, 32 chips
#   REGION=us-east5 GENERATION=v5p ./request-quota.sh
#   REGION=us-east4 CHIPS=8 ./request-quota.sh
#
# Check afterwards:
#   gcloud quotas preferences list --project=<project>

set -euo pipefail

PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-}"
GENERATION="${GENERATION:-v6e}"
CHIPS="${CHIPS:-32}"
CONTACT="${CONTACT:-$(gcloud config get-value account 2>/dev/null)}"

if [ -z "$PROJECT" ]; then
  echo "ERROR: no project. Set GOOGLE_CLOUD_PROJECT or run 'gcloud config set project ...'." >&2
  exit 1
fi
if [ -z "$REGION" ]; then
  echo "ERROR: REGION is required, e.g. REGION=us-east5 $0" >&2
  echo "       Regions that publish a machine type are not necessarily regions you have" >&2
  echo "       quota in — check both:" >&2
  echo "         gcloud compute machine-types list --filter='name=ct6e-standard-1t' --format='value(zone)'" >&2
  exit 1
fi

case "$GENERATION" in
  v6e)
    STD_QUOTA="TPUS-PER-TPU-FAMILY-per-project-region"
    STD_DIMS="region=$REGION,tpu_family=CT6E"
    STD_ID="ct6e-family-$REGION"
    STD_FAMILY="CT6E"
    PRE_QUOTA="PREEMPTIBLE-TPU-V6E-per-project-region"
    ;;
  v5p)
    STD_QUOTA="TPU-V5P-per-project-region"
    STD_DIMS="region=$REGION"
    STD_ID="tpu-v5p-$REGION"
    STD_FAMILY=""
    PRE_QUOTA="PREEMPTIBLE-TPU-V5P-per-project-region"
    ;;
  v5e)
    echo "ERROR: v5e has no Compute Engine create path — 'instances create' refuses" >&2
    echo "       ct5lp-* with 'This user agent is not allowed to use the machine type'." >&2
    echo "       Compute Engine quota for it buys nothing. Use the Cloud TPU API's quota." >&2
    exit 1
    ;;
  *)
    echo "ERROR: unknown GENERATION '$GENERATION'. Supported: v6e, v5p." >&2
    exit 1
    ;;
esac

JUSTIFICATION="${JUSTIFICATION:-Serving Gemma 4 under vLLM on a single TPU $GENERATION chip via FLEX_START. This project holds $GENERATION quota on the Cloud TPU API in this region but not on Compute Engine, and the Cloud TPU API is deprecated.}"

echo "Project    : $PROJECT"
echo "Region     : $REGION"
echo "Generation : $GENERATION"
echo "Chips      : $CHIPS"
echo "Contact    : $CONTACT"
echo

# Current effective value for one quota id in this region.
#
# NEEDED BECAUSE ASKING FOR LESS THAN YOU HOLD IS AN ERROR, not a no-op:
#   FAILED_PRECONDITION: The quota override ... decreases effective quota unsafely
# Hit on 2026-08-11 in us-central1, us-east1 and us-west1, where the preemptible quota
# already sat at its inherited default of 1536 and a blanket request for 32 was a
# reduction. The two quotas carry DIFFERENT defaults — an unlisted region inherits 0 on
# the family quota and 1536 on the preemptible one — so the two halves of this script
# cannot share one answer. Prints the empty string when unset, which callers treat as 0.
current_value() {
  local quota_id="$1" want_family="$2"
  gcloud alpha quotas info describe "$quota_id" \
    --service=compute.googleapis.com --project="$PROJECT" --format=json 2>/dev/null |
    REGION="$REGION" WANT_FAMILY="$want_family" python3 -c '
import json, os, sys
region, want = os.environ["REGION"], os.environ["WANT_FAMILY"]
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
exact, default = None, None
for di in d.get("dimensionsInfos", []):
    dim = di.get("dimensions", {}) or {}
    if want and dim.get("tpu_family") != want:
        continue
    val = (di.get("details") or {}).get("value")
    if dim.get("region") == region:
        exact = val
    elif not dim.get("region"):
        default = val
val = exact if exact is not None else default
print("" if val is None else val)
'
}

# Submit unless the region already holds at least CHIPS on that metric.
request() {
  local quota_id="$1" dims="$2" pref_id="$3" want_family="$4"
  local have
  have="$(current_value "$quota_id" "$want_family")"
  echo "==> $quota_id in $REGION (currently: ${have:-0})"
  if [ -n "$have" ] && [ "$have" -ge "$CHIPS" ] 2>/dev/null; then
    echo "    SKIP — already at $have, and requesting $CHIPS would be a decrease."
    return 0
  fi
  gcloud quotas preferences create \
    --service=compute.googleapis.com \
    --project="$PROJECT" \
    --quota-id="$quota_id" \
    --dimensions="$dims" \
    --preferred-value="$CHIPS" \
    --preference-id="$pref_id" \
    --email="$CONTACT" \
    --justification="$JUSTIFICATION" || echo "    request failed (see error above)"
}

request "$STD_QUOTA" "$STD_DIMS" "$STD_ID" "$STD_FAMILY"
request "$PRE_QUOTA" "region=$REGION" "preemptible-tpu-$GENERATION-$REGION" ""

echo
echo "Submitted. Check status with:"
echo "  gcloud quotas preferences list --project=$PROJECT"
