#!/bin/bash
# Unattended SGLang-JAX run:  sudo systemd-run --unit=ladder-sgl bash run-sgl.sh QR
# setup-sgl.sh, then ladder.py one+acc+sweep:sglang, then finish.sh with ALWAYS_DELETE=1 (nobody is there to rerun
# a failed job, so the VM uploads what it has and deletes itself whatever the outcome).
set -u
QR=$1
export LADDER=/opt/ladder HOME=${HOME:-/root} RUN=2026-11-08-ladder-sglang-v5e1
O=/opt/ladder-out/$RUN
mkdir -p $O
cd $LADDER/flight
bash setup-sgl.sh > $O/setup-sgl.log 2>&1 && cp $LADDER/versions-sgl.txt $O/versions.txt \
  && python3.13 ladder.py $O one+acc+sweep:sglang > $O/ladder.stdout 2>&1
echo "$(date -u '+%F %T') run-sgl exit $?" >> $O/driver.log
ALWAYS_DELETE=1 bash finish.sh "$QR" none
