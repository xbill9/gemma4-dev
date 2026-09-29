#!/bin/bash
# When ec2_measure.py exits (or is killed), terminate any instance this rig left running.
while pgrep -f "ec2_measure.py benchmarks/runs/2026-09-29-sagemaker-match-g6" >/dev/null; do sleep 30; done
ids=$(aws ec2 describe-instances --region us-east-2 --filters Name=tag:ManagedBy,Values=gpu-vllm-g6-2b-w4a16 Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[].Instances[].InstanceId' --output text)
[ -n "$ids" ] && aws ec2 terminate-instances --region us-east-2 --instance-ids $ids --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text
echo "$(date -u +%FT%TZ) watchdog done (left running: ${ids:-none})"
