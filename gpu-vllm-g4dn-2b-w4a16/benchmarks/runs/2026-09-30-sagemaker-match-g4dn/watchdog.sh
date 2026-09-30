#!/bin/bash
# When ec2_measure.py exits (or is killed), terminate the instance it launched.
cd /home/xbill/gemma4-dev/gpu-vllm-g4dn-2b-w4a16
R=benchmarks/runs/2026-09-30-sagemaker-match-g4dn
while pgrep -f "ec2_measure.py $R" >/dev/null; do sleep 20; done
[ -f $R/instance-id.txt ] || { echo "$(date -u +%FT%TZ) no instance launched"; exit 0; }
iid=$(cat $R/instance-id.txt)
echo "$(date -u +%FT%TZ) terminate $iid: $(aws ec2 terminate-instances --region us-east-2 --instance-ids $iid --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1)"
