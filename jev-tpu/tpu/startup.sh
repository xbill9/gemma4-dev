#!/bin/bash
# Startup script for an unattended run: boot.sh (Docker, token, image), then the code from
# GCS, then run.sh. Metadata key jev-code names the code tarball, jev-run the run prefix.
md() { curl -s -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
BUCKET=aisprint-491218-bucket  # set to your own bucket, as in run.sh
CODE=$(md jev-code); RUN=$(md jev-run)
mkdir -p /opt/jev-tpu/results
gcloud storage cp -q "gs://$BUCKET/jev-tpu/inputs/$CODE" /tmp/code.tgz && tar xzf /tmp/code.tgz -C /opt/jev-tpu
bash /opt/jev-tpu/tpu/boot.sh
bash /opt/jev-tpu/tpu/run.sh "$RUN"
