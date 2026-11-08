#!/bin/bash
# Runs on a ladder VM:  sudo systemd-run --unit=ladder-finish bash finish.sh QR UNIT [SINCE]
# SINCE ("YYYY-MM-DD HH:MM:SS", UTC) checks only driver.log lines from then on, for a rerun after a failed job.
# Waits for the ladder unit UNIT to end, uploads the run directory to the bucket, reads it back to check it,
# then deletes this VM's own queued resource QR, unless the run log shows a failed job (then the VM stays up
# until its 24-hour flex-start limit so the failed job can be rerun by hand).
# ALWAYS_DELETE=1 deletes after a verified upload even with failures (unattended runs); RUN overrides the run name.
# Everything it does is logged to $O/finish.log and uploaded as finish.log beside the archive.
set -u
QR=$1
UNIT=$2
SINCE=${3:-0}
RUN=${RUN:-2026-11-06-ladder-v5e1}
O=/opt/ladder-out/$RUN
B=gs://aisprint-491218-ladder-v5e1/$RUN/$QR
Z=(--zone us-west4-a --project aisprint-491218)
export HOME=${HOME:-/root}
log() { echo "$(date -u '+%F %T') $*" | tee -a $O/finish.log; }

log "waiting for $UNIT"
while systemctl is-active --quiet "$UNIT"; do sleep 120; done
log "$UNIT ended: $(systemctl show -p Result --value "$UNIT")"

ok=0
for i in 1 2 3; do
  tar czf /tmp/ladder-run.tgz -C /opt/ladder-out $RUN
  want=$(sha256sum /tmp/ladder-run.tgz | cut -d' ' -f1)
  gcloud storage cp /tmp/ladder-run.tgz $B/ladder-run.tgz >/dev/null 2>&1
  rm -f /tmp/ladder-run.back
  gcloud storage cp $B/ladder-run.tgz /tmp/ladder-run.back >/dev/null 2>&1
  got=$(sha256sum /tmp/ladder-run.back 2>/dev/null | cut -d' ' -f1)
  if [ "$want" = "$got" ]; then ok=1; break; fi
  log "upload attempt $i did not read back ($want / $got)"
  sleep 60
done
if [ $ok != 1 ]; then
  log "UPLOAD FAILED, not deleting $QR"
  gcloud storage cp $O/finish.log $B/finish.log >/dev/null 2>&1
  exit 1
fi
echo "$want  ladder-run.tgz" > /tmp/ladder-run.sha256
gcloud storage cp /tmp/ladder-run.sha256 $B/ladder-run.sha256 >/dev/null 2>&1
log "uploaded $B/ladder-run.tgz sha256 $want ($(tar tzf /tmp/ladder-run.tgz | wc -l) entries)"

fails=$(awk -v s="$SINCE" 'substr($0,1,19) >= s' $O/driver.log | grep -E "FAIL|ok=False" | grep -v torchtpu)
if [ -n "$fails" ] && [ "${ALWAYS_DELETE:-0}" != 1 ]; then
  log "run log has failures, NOT deleting $QR:"
  echo "$fails" | tee -a $O/finish.log
  gcloud storage cp $O/finish.log $B/finish.log >/dev/null 2>&1
  exit 2
fi
log "deleting queued resource $QR"
gcloud storage cp $O/finish.log $B/finish.log >/dev/null 2>&1
gcloud alpha compute tpus queued-resources delete "$QR" "${Z[@]}" --force --async --quiet >>$O/finish.log 2>&1
log "delete requested (rc=$?)"
gcloud storage cp $O/finish.log $B/finish.log >/dev/null 2>&1
