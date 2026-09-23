#!/usr/bin/env bash
# Launch the self-driving pass-2 instance (see PREREGISTRATION.md addendum) in
# the first region and zone with a single-L4 G6 available. Needs credentials in
# ~/.aws/jev.aws_creds from save-aws-creds.sh; every call it makes fits in one
# short credential window, and the instance needs none afterwards.
#
#   bash aws/launch_pass2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . "$HOME/.aws/jev.aws_creds"; set +a
aws sts get-caller-identity --query Account --output text >/dev/null || { echo "credentials expired"; exit 1; }

python3 aws/make_pass2_userdata.py > /tmp/jev-pass2-userdata.sh
MYIP=$(curl -s -4 ifconfig.me)
AMI_PARAM=/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-26.04/latest/ami-id

for pair in "us-east-2 bb241257e20d4afdac3e0f43f66a7c16Ex0Zai1b" "us-west-2 be3271e9d20c40189ccb74cd04432542QzFS0ARR"; do
  set -- $pair
  echo "quota request $1: $(aws service-quotas get-requested-service-quota-change --request-id $2 --region $1 --query RequestedQuota.Status --output text 2>&1 | tail -1)"
done

for R in us-east-1 us-west-2 us-east-2; do
  Q=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region $R --query Quota.Value --output text 2>/dev/null)
  echo "$R G-instance quota: $Q"
  [ "${Q%.*}" -ge 4 ] 2>/dev/null || continue
  if [ "$R" = us-east-1 ]; then VPC=vpc-0bfdd15d906e0c008; else
    VPC=$(aws ec2 describe-vpcs --region $R --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text); fi
  SG=$(aws ec2 describe-security-groups --region $R --filters Name=group-name,Values=jev-eval-sg2 Name=vpc-id,Values=$VPC --query 'SecurityGroups[0].GroupId' --output text)
  if [ "$SG" = None ]; then
    SG=$(aws ec2 create-security-group --region $R --group-name jev-eval-sg2 --description "jev eval pass 2: 8000 from one IP" \
      --vpc-id $VPC --tag-specifications 'ResourceType=security-group,Tags=[{Key=ManagedBy,Value=jev}]' --query GroupId --output text)
  fi
  aws ec2 authorize-security-group-ingress --region $R --group-id $SG --protocol tcp --port 8000 --cidr $MYIP/32 >/dev/null 2>&1 || true
  AMI=$(aws ssm get-parameter --region $R --name $AMI_PARAM --query Parameter.Value --output text)
  for T in g6.xlarge g6.2xlarge g6.4xlarge; do
    for SUB in $(aws ec2 describe-subnets --region $R --filters Name=vpc-id,Values=$VPC --query 'Subnets[].SubnetId' --output text); do
      out=$(aws ec2 run-instances --region $R --image-id $AMI --instance-type $T --count 1 --subnet-id $SUB \
        --security-group-ids $SG --iam-instance-profile Name=g6-vllm-instance-profile \
        --instance-initiated-shutdown-behavior terminate --user-data file:///tmp/jev-pass2-userdata.sh \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,Throughput=500,Iops=6000,DeleteOnTermination=true}' \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=jev-eval-pass2},{Key=ManagedBy,Value=jev}]' \
        --query 'Instances[0].[InstanceId,InstanceType,Placement.AvailabilityZone,LaunchTime]' --output text 2>&1 | tail -1)
      case "$out" in
        i-*) ID=${out%%[[:space:]]*}
             aws ec2 wait instance-running --region $R --instance-ids $ID
             IP=$(aws ec2 describe-instances --region $R --instance-ids $ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
             echo "$out $R $SG $IP" | tee results/.pass2-instance
             echo "LAUNCHED $T in $R; results will be served at http://$IP:8000/out/ when done"
             exit 0 ;;
        *) echo "$R $T $SUB: no capacity" ;;
      esac
    done
  done
done
echo "NO CAPACITY in any region with quota"
exit 2
