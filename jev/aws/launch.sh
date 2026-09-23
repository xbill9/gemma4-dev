#!/usr/bin/env bash
# Launch one self-driving G6 (one NVIDIA L4) instance in the first region and
# zone with capacity, running the given user data. Re-exports short-lived
# credentials from the `aws login` session first.
#
#   bash aws/launch.sh <user-data-file> <name-tag>
set -uo pipefail
cd "$(dirname "$0")/.."
UD=${1:?user-data file}; NAME=${2:?name tag}
"$HOME/bin/save-aws-creds.sh" "$HOME/.aws/jev.aws_creds" >/dev/null || { echo "aws login session expired: run aws login --remote"; exit 1; }
set -a; . "$HOME/.aws/jev.aws_creds"; set +a
MYIP=$(curl -s -4 ifconfig.me)
AMI_PARAM=/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-26.04/latest/ami-id
for R in us-east-1 us-west-2 us-east-2; do
  Q=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region $R --query Quota.Value --output text 2>/dev/null)
  echo "$R G-instance quota: $Q"
  [ "${Q%.*}" -ge 4 ] 2>/dev/null || continue
  if [ "$R" = us-east-1 ]; then VPC=vpc-0bfdd15d906e0c008; else
    VPC=$(aws ec2 describe-vpcs --region $R --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text); fi
  SG=$(aws ec2 describe-security-groups --region $R --filters Name=group-name,Values=jev-eval-sg Name=vpc-id,Values=$VPC --query 'SecurityGroups[0].GroupId' --output text)
  [ "$SG" = None ] && SG=$(aws ec2 create-security-group --region $R --group-name jev-eval-sg --description "jev eval: 8000 from one IP" \
      --vpc-id $VPC --tag-specifications 'ResourceType=security-group,Tags=[{Key=ManagedBy,Value=jev}]' --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --region $R --group-id $SG --protocol tcp --port 8000 --cidr $MYIP/32 >/dev/null 2>&1 || true
  AMI=$(aws ssm get-parameter --region $R --name $AMI_PARAM --query Parameter.Value --output text)
  for T in g6.xlarge g6.2xlarge g6.4xlarge; do
    for SUB in $(aws ec2 describe-subnets --region $R --filters Name=vpc-id,Values=$VPC --query 'Subnets[].SubnetId' --output text); do
      out=$(aws ec2 run-instances --region $R --image-id $AMI --instance-type $T --count 1 --subnet-id $SUB \
        --security-group-ids $SG --iam-instance-profile Name=g6-vllm-instance-profile \
        --instance-initiated-shutdown-behavior terminate --user-data file://$UD \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,Throughput=500,Iops=6000,DeleteOnTermination=true}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=ManagedBy,Value=jev}]" \
        --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone,LaunchTime]' --output text 2>&1 | tail -1)
      case "$out" in
        i-*) ID=${out%%[[:space:]]*}
             aws ec2 wait instance-running --region $R --instance-ids $ID
             IP=$(aws ec2 describe-instances --region $R --instance-ids $ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
             echo "$out $R $SG $IP" | tee results/.instance
             echo "LAUNCHED $T in $R at $IP"; exit 0 ;;
        *) echo "$R $T $SUB: no capacity" ;;
      esac
    done
  done
done
echo "NO CAPACITY in any region with quota"; exit 2
