#!/bin/bash
# --- 2026-11-10: finish the T4 accuracy queue without an SSH session; results go to GCS ---
# Runs only while metadata t4-queue=run. Removed when the queue reports DONE.
if [ "$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/attributes/t4-queue)" = run ]; then
cat > /usr/local/bin/t4-queue-run.sh <<'EOF'
#!/bin/bash
U=xbill_glitnir_com; H=/home/$U; G=$H/gemma4-dev; B=gs://comglitn-gemma4-t4-results
Q=$G/gpu-vllm-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-05-engines-t4
log() { echo "$(date -Is) $*" >> /var/log/t4-queue-run.log; echo "t4-queue: $*" > /dev/ttyS0; }
up() {
  for p in gpu-pytorch-t4-2b-q4w4a16/benchmarks/runs/2026-11-06-accuracy-t4 \
           gpu-vllm-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-06-accuracy-t4 \
           gpu-llamacpp-t4-2b-q4_0/benchmarks/runs/2026-11-06-accuracy-t4 \
           gpu-sglang-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-10-int4-t4/accuracy \
           gpu-litertlm-t4-2b-int248/benchmarks/runs/2026-11-06-accuracy-t4; do
    [ -d $G/$p ] && gcloud storage rsync -r -q $G/$p $B/$p >> /var/log/t4-queue-run.log 2>&1
  done
  gcloud storage cp -q $Q/queue.log /var/log/t4-queue-run.log $B/ >/dev/null 2>&1
  log "uploaded; queue: $(tail -1 $Q/queue.log)"
}
for i in $(seq 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
mountpoint -q /opt1 || mount /opt1
[ -n "$(awk 'NR>1' /proc/swaps)" ] || swapon /opt1/swapfile
log "nvidia-smi ok=$(nvidia-smi >/dev/null 2>&1 && echo y); opt1=$(ls /opt1 | head -3 | tr '\n' ,); swap=$(awk 'NR>1{print $1}' /proc/swaps)"
if tail -1 $Q/queue.log | grep -q DONE; then log "queue already DONE"; up; exit 0; fi
sudo -u $U -i bash -c "cd $Q && exec bash queue.sh > queue-acc.out 2>&1 < /dev/null" &
QP=$!
log "queue started pid $QP"
while kill -0 $QP 2>/dev/null; do sleep 600; up; done
up
log "queue exited: $(tail -1 $Q/queue.log)"
EOF
chmod +x /usr/local/bin/t4-queue-run.sh
setsid nohup /usr/local/bin/t4-queue-run.sh > /dev/null 2>&1 < /dev/null &
fi
# --- end ---
