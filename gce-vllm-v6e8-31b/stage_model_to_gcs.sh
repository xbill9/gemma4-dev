#!/bin/bash
#
# Stage this rig's checkpoint into GCS so every future boot restores it in-region.
#
# WHY THIS EXISTS
# ---------------
# The 62 GB pull of google/gemma-4-31B-it from Hugging Face IS this rig's boot-readiness
# budget: it is why startup_script_template.sh waits 90 minutes rather than the 20 it waited
# while the rig served E2B. Eight v6e chips bill at $10.80/hr (flex-start list, europe-west4,
# read from the Cloud Billing Catalog 2026-08-25), so that pull is roughly $16 of the cost of
# EVERY boot, paid before a single token is served.
#
# Staged in-region, the same bytes restore in minutes. The staging itself runs on a cheap CPU
# VM in the same region, so this script never touches a TPU.
#
# WHAT IT DOES
#   1. creates the bucket (europe-west4, matching GOOGLE_CLOUD_REGION) if it does not exist
#   2. grants the TPU VM's service account read access on that bucket
#   3. boots one e2-standard-8, which downloads the checkpoint and streams a tar of the
#      Hugging Face cache into the bucket
#   4. deletes itself when done — including on failure, via a trap
#
# The object it writes is what MODEL_GCS_URI in tpu.env points at.
#
# SAFETY: this script creates a CPU VM and a bucket. It provisions NO TPU capacity and
# touches no existing instance. Re-running it overwrites the staged tar and nothing else.
#
# Usage:  ./stage_model_to_gcs.sh            # stage
#         ./stage_model_to_gcs.sh --status   # is it done yet?
#         ./stage_model_to_gcs.sh --logs     # why did it fail? (works after the VM is gone)
#         ./stage_model_to_gcs.sh --keep     # do not self-delete; leave the VM for inspection
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# tpu.env is the source of truth; a real environment variable still wins, as everywhere else.
if [ -f "$HERE/tpu.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$HERE/tpu.env"
  set +a
fi

PROJECT="${GOOGLE_CLOUD_PROJECT:-aisprint-491218}"
REGION="${GOOGLE_CLOUD_REGION:-europe-west4}"
ZONE="${GOOGLE_CLOUD_ZONE:-europe-west4-a}"
MODEL="${MODEL_NAME:-google/gemma-4-31B-it}"
SECRET="${HF_SECRET_ID:-hf-token}"
BUCKET="${MODEL_GCS_BUCKET:-${PROJECT}-hf-cache-${REGION}}"
# Object name derives from the model, so two checkpoints never collide in one bucket.
OBJECT="${MODEL//\//__}.hfcache.tar"
URI="gs://${BUCKET}/${OBJECT}"
# The stager uploads its log here before self-deleting, so a failed run stays diagnosable.
LOG_URI="gs://${BUCKET}/${OBJECT%.tar}.stage.log"
KEEP=0
STAGER="stage-hf-cache-$(echo "$MODEL" | tr '/[:upper:]' '-[:lower:]' | tr -cd 'a-z0-9-' | cut -c1-40)"

case "${1:-}" in
  --keep) KEEP=1; shift ;;
  --logs)
    # The stager uploads its log before self-deleting, so this works on a FAILED run too —
    # which is the whole reason the upload happens before the delete.
    echo "Log for the last run: $LOG_URI"
    gcloud storage cat "$LOG_URI" --project="$PROJECT" 2>/dev/null || echo "(no log staged yet)"
    exit 0
    ;;
esac

if [ "${1:-}" = "--status" ]; then
  echo "Target: $URI"
  if gcloud storage ls -l "$URI" --project="$PROJECT" 2>/dev/null; then
    echo "✅ Staged. Set MODEL_GCS_URI=$URI in tpu.env"
  else
    echo "⏳ Not present yet."
  fi
  echo
  echo "Stager VM:"
  gcloud compute instances describe "$STAGER" --zone="$ZONE" --project="$PROJECT" \
    --format="value(status)" 2>/dev/null || echo "  (gone — it deletes itself when finished)"
  echo
  echo "Progress:  gcloud compute ssh $STAGER --zone=$ZONE --project=$PROJECT \\"
  echo "             --command='sudo tail -50 /var/log/stage-hf-cache.log'"
  exit 0
fi

echo "▶ Project : $PROJECT"
echo "▶ Model   : $MODEL"
echo "▶ Target  : $URI"
echo "▶ Stager  : $STAGER in $ZONE"
echo

# --- 1. Bucket -----------------------------------------------------------------------------
# MUST be in the same region as the TPU instance. A bucket in us-east4 would restore over the
# public backbone and give back the entire saving this script exists to create.
if gcloud storage buckets describe "gs://$BUCKET" --project="$PROJECT" >/dev/null 2>&1; then
  echo "✅ Bucket gs://$BUCKET exists."
else
  echo "📦 Creating gs://$BUCKET in $REGION..."
  gcloud storage buckets create "gs://$BUCKET" \
    --project="$PROJECT" --location="$REGION" \
    --default-storage-class=STANDARD --uniform-bucket-level-access
fi

# --- 2. Let the TPU VM read it -------------------------------------------------------------
# The TPU instance runs as the default compute service account and already carries
# --scopes=cloud-platform; without this binding the restore fails and the startup script
# falls back to the 90-minute online pull, silently costing what this script saves.
PROJNUM="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
TPU_SA="${PROJNUM}-compute@developer.gserviceaccount.com"
echo "🔑 Granting roles/storage.objectViewer on the bucket to $TPU_SA..."
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --project="$PROJECT" \
  --member="serviceAccount:${TPU_SA}" \
  --role=roles/storage.objectViewer >/dev/null
echo "✅ Granted."

# --- 3. The stager's own startup script ----------------------------------------------------
# Written to a temp file rather than inlined so no quoting of this heredoc reaches gcloud.
STARTUP="$(mktemp)"
trap 'rm -f "$STARTUP"' EXIT

cat > "$STARTUP" <<STAGER_EOF
#!/bin/bash
LOG=/var/log/stage-hf-cache.log
exec > "\$LOG" 2>&1
set -x

MODEL="$MODEL"
SECRET="$SECRET"
URI="$URI"
LOG_URI="$LOG_URI"
PROJECT="$PROJECT"
ZONE="$ZONE"
SELF="$STAGER"
KEEP="$KEEP"

# Ship the log to GCS BEFORE deleting the VM.
#
# THIS ORDERING IS THE WHOLE POINT. The first version of this script deleted the VM in its
# trap and nothing else, so when the run failed six minutes in it destroyed the only copy of
# its own log and left an empty bucket and no way to find out why. A cleanup path that
# removes the evidence of its own failure is worse than no cleanup path.
cleanup() {
  RC=\$?
  set +x
  echo "=== EXIT rc=\$RC ==="
  gcloud storage cp "\$LOG" "\$LOG_URI" 2>/dev/null || echo "(could not upload log)"
  if [ "\$KEEP" = "1" ]; then
    echo "--keep set: leaving \$SELF running for inspection. DELETE IT YOURSELF when done."
    exit \$RC
  fi
  gcloud --quiet compute instances delete "\$SELF" --zone="\$ZONE" --project="\$PROJECT" || true
}
trap cleanup EXIT

# Every step is checked and names itself on failure, so the uploaded log says WHERE it died
# rather than just that it did.
die() { echo "FATAL: \$*"; exit 1; }

echo "=== step: apt ==="
apt-get update -qq || die "apt-get update failed"
apt-get install -y -qq python3-pip || die "installing python3-pip failed"

echo "=== step: pip ==="
# --break-system-packages is a PEP 668 flag that Ubuntu 22.04's pip 22.x does not have, and
# passing it there is a hard "no such option" failure. Try with, fall back to without.
# Plain package names, no extras: huggingface_hub 1.x provides neither a 'cli' extra (the hf
# CLI is built in now) nor an 'hf_transfer' one. Asking for them only emits
# "does not provide the extra" WARNINGS and skips them, so hf_transfer silently never lands
# and the 62 GB download runs single-stream.
PIP_PKGS="huggingface_hub hf_transfer"
pip3 install -q --break-system-packages \$PIP_PKGS \
  || pip3 install -q \$PIP_PKGS \
  || die "pip install of \$PIP_PKGS failed"
python3 -c "import hf_transfer" || die "hf_transfer did not install; HF_HUB_ENABLE_HF_TRANSFER would fail closed"

# pip may land the entry point outside the metadata runner's PATH.
export PATH="\$PATH:/usr/local/bin:/root/.local/bin"
command -v hf || die "hf CLI not on PATH after install"
hf version || die "hf CLI will not run"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/mnt/hf
mkdir -p "\$HF_HOME" || die "could not create \$HF_HOME"

echo "=== step: token ==="
# Same Secret Manager path the TPU rig uses. Tracing off across the token, as there.
set +x
HF_TOKEN="\$(gcloud secrets versions access latest --secret="\$SECRET" --project="\$PROJECT" 2>&1)"
if [ -z "\$HF_TOKEN" ] || [ "\${HF_TOKEN#ERROR}" != "\$HF_TOKEN" ]; then
  set -x
  die "could not read secret \$SECRET (check the VM service account has secretAccessor)"
fi
# Length only — never the value, and never into a logged string.
echo "token retrieved, length \${#HF_TOKEN}"
export HF_TOKEN
set -x

echo "=== step: download ==="
# No --quiet: "hf download" has no such flag and an unknown option aborts the download.
# NOTE: no backticks anywhere in this heredoc. It is unquoted (so \$VAR interpolates from the
# outer script), which means a backtick is command substitution run by the OUTER shell at
# generation time. A backtick pair in THIS comment previously ran "hf download" locally and
# baked its usage text into the generated script as executable lines.
DL_START=\$(date +%s)
hf download "\$MODEL" --max-workers 16 || die "hf download failed (gated repo? token lacks access to \$MODEL?)"
echo "Downloaded in \$((\$(date +%s) - DL_START))s"
du -sh "\$HF_HOME/hub" || true
[ -d "\$HF_HOME/hub" ] || die "no hub/ tree after download"

echo "=== step: upload ==="
# Stream a tar of the WHOLE hub/ tree straight to GCS.
#   - tar preserves the blobs/ + snapshots/ symlink layout; gcloud storage cp -r would
#     follow the symlinks and upload 124 GB of duplicated weights instead of 62.
#   - one object, so the restore is a single sequential read.
#   - uncompressed: bf16 weights do not compress, and gzip would just add a CPU bottleneck.
UP_START=\$(date +%s)
set -o pipefail
tar -C "\$HF_HOME" -cf - hub | gcloud storage cp - "\$URI" || die "tar/upload to \$URI failed"
set +o pipefail
echo "Uploaded in \$((\$(date +%s) - UP_START))s"

gcloud storage ls -l "\$URI" || die "object missing after upload"
echo "STAGING COMPLETE"
STAGER_EOF

# The generated script is what actually runs, and heredoc expansion can corrupt it in ways
# that are invisible in this file — see the backtick note inside. Check it before shipping.
bash -n "$STARTUP" || {
  echo "❌ The GENERATED stager script is not valid bash. Not booting a VM to run it."
  echo "   First 40 lines of what was generated:"
  sed -n "1,40p" "$STARTUP" | sed "s/^/     /"
  exit 1
}
grep -qE "^(Download files|Arguments:)" "$STARTUP" && {
  echo "❌ The generated script contains CLI help text — a backtick or \$( ) in the heredoc"
  echo "   was expanded by this shell. Offending lines:"
  grep -nE "^(Download files|Arguments:)" "$STARTUP" | sed "s/^/     /"
  exit 1
}
echo "✅ Generated stager script checks out."

# --- 4. Boot the stager --------------------------------------------------------------------
# e2-standard-8: 8 vCPU is enough to saturate the HF download and caps egress at 16 Gbps.
# 200 GB balanced disk holds the 62 GB cache with room for the tar stream's buffers.
echo "🚀 Booting $STAGER (e2-standard-8, 200 GB) in $ZONE..."
gcloud compute instances create "$STAGER" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type=e2-standard-8 \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB --boot-disk-type=pd-balanced \
  --scopes=cloud-platform \
  --labels=rig=gce-vllm-v6e8-31b,purpose=hf-cache-staging \
  --metadata-from-file=startup-script="$STARTUP"

cat <<DONE

✅ Stager booted. It downloads, uploads, and then DELETES ITSELF.

   Expect roughly 30-60 minutes, dominated by the Hugging Face download.
   An e2-standard-8 is a few cents an hour — this is not the expensive part.

   Watch:   ./stage_model_to_gcs.sh --status
   Log:     gcloud compute ssh $STAGER --zone=$ZONE --project=$PROJECT \\
              --command='sudo tail -f /var/log/stage-hf-cache.log'

   When the object exists, set this in tpu.env:

     MODEL_GCS_URI=$URI

DONE
