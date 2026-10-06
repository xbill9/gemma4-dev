# Restart the Gemma-4 Option B inf2 box (from AMI)

The spot box `i-01718af33c99f0eeb` was AMI'd and terminated on 2026-07-06.
Everything (compiled neffs, real model, transformers-5.13 venv, all `/workspace` scripts)
is baked into the AMI — launch from it and the environment is exactly as left.

## Backup artifact
- **AMI:** `ami-0c13e7feb3fe2e01e`  (name `gemma4-optb-inf2-20260706`, region **us-east-1**)
- Snapshot: `snap-0149d318c5f36cfb9` (300 GB root)

## Original launch config (replicate this)
| field | value |
|---|---|
| instance type | `inf2.8xlarge` |
| AZ / subnet | `us-east-1f` / `subnet-09e0f13c0a5b43092` |
| security group | `sg-065c6975cbd84dad3` |
| key pair | `alinux` |
| IAM instance profile | `aws-elasticbeanstalk-ec2-role` |

## Launch tomorrow (on-demand — simplest, reliable)
```bash
aws ec2 run-instances --region us-east-1 \
  --image-id ami-0c13e7feb3fe2e01e \
  --instance-type inf2.8xlarge \
  --subnet-id subnet-09e0f13c0a5b43092 \
  --security-group-ids sg-065c6975cbd84dad3 \
  --key-name alinux \
  --iam-instance-profile Name=aws-elasticbeanstalk-ec2-role \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=inferentia-2b-devops-agent}]' \
  --query 'Instances[0].InstanceId' --output text
```
For **spot** instead (cheaper, can be interrupted), add:
```
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}'
```
Then wait for it to boot + SSM to register (~2 min):
```bash
aws ec2 wait instance-running --region us-east-1 --instance-ids <NEW_ID>
aws ssm describe-instance-information --region us-east-1 \
  --filters Key=InstanceIds,Values=<NEW_ID> --query 'InstanceInformationList[].PingStatus' --output text
```

## Bring the inference server back up (on the new box, via SSM)
Nothing to rebuild — the neffs and venv are already there. Just start the server:
```bash
# 256-token / 64-prompt build (recommended):
aws ssm send-command --region us-east-1 --instance-ids <NEW_ID> \
  --document-name AWS-RunShellScript --parameters 'commands=[
    "export PATH=/opt/aws_neuronx_venv_pytorch_2_8/bin:$PATH",
    "cd /workspace && pkill -f optb_server.py 2>/dev/null; sleep 2",
    "KV_MAX=256 KV_BUCKET=64 KV_PRE_OUT=/workspace/kv_pre_big.pt KV_DEC_OUT=/workspace/kv_dec_big.pt nohup python /workspace/optb_server.py > /workspace/server.log 2>&1 &",
    "sleep 90; grep READY /workspace/server.log"]'
```
Then query it (from the box, port 8080):
```bash
curl -s -X POST localhost:8080/generate -d '{"prompt":"What is the capital of France?"}'
```

## Key on-box paths (all inside the AMI)
- venv: `/opt/aws_neuronx_venv_pytorch_2_8` (torch 2.8 + torch_neuronx + transformers 5.13.0);
  run with `export PATH=/opt/aws_neuronx_venv_pytorch_2_8/bin:$PATH` (NOT `source activate` under dash)
- model: `/workspace/real-gemma4-E2B-it` (real 5.12B multimodal, model_type=gemma4)
- neffs: `kv_pre_neff.pt`+`kv_dec_neff.pt` (128/32), `kv_pre_big.pt`+`kv_dec_big.pt` (256/64)
- scripts: `/workspace/optb_kv.py` (cpu|trace), `optb_kv_run.py`, `optb_server.py`, `optb_gen.py`, `optb_ask.py`
  (all also in git under this dir)

## To recompile neffs from scratch (only if AMI is lost)
```bash
export PATH=/opt/aws_neuronx_venv_pytorch_2_8/bin:$PATH
# 256/64 pair (~18 min):
KV_MAX=256 KV_BUCKET=64 KV_PRE_OUT=/workspace/kv_pre_big.pt KV_DEC_OUT=/workspace/kv_dec_big.pt \
  python /workspace/optb_kv.py trace
```

## S3 backup (secondary, in addition to the AMI)
Made 2026-07-06 to `s3://xbill-gemma4-patches-2b/optb-backup/` (29 objects, 32 GiB):
- `neffs/` — kv_pre_neff.pt+kv_dec_neff.pt (128/32), kv_pre_big.pt+kv_dec_big.pt (256/64), optb_neff.pt, optb_gen_neff.pt
- `model/real-gemma4-E2B-it/` — real 5.12B checkpoint
- `scripts/optb_*.py`, `MANIFEST.txt`

Restore onto a fresh box WITHOUT the AMI (e.g. plain Neuron DLAMI):
```bash
export PATH=/opt/aws_neuronx_venv_pytorch_2_8/bin:$PATH
pip install transformers==5.13.0
mkdir -p /workspace && cd /workspace
aws s3 sync s3://xbill-gemma4-patches-2b/optb-backup/neffs/  /workspace/
aws s3 sync s3://xbill-gemma4-patches-2b/optb-backup/scripts/ /workspace/
aws s3 sync s3://xbill-gemma4-patches-2b/optb-backup/model/real-gemma4-E2B-it /workspace/real-gemma4-E2B-it
# then start optb_server.py as above
```
(Box role has AmazonS3FullAccess as of 2026-07-06.)

## us-east-2 (fallback region — where capacity actually was on 2026-07-06)
us-east-1 spot inf2.8xlarge was unfulfillable for hours; **us-east-2 had capacity** (and cheaper spot).
AMI copied cross-region: us-east-1 `ami-0c13e7feb3fe2e01e` → **us-east-2 `ami-0f07bf96d8551d36c`**.
Current box: spot `i-09d0091339c8a6ee9` (us-east-2c). Launch template `lt-0bca7ea41c9e213e5`
(`gemma4-optb-lt-e2`, auto-start-server user-data, no key → SSM, default-VPC SG `sg-0f5a179467ec1eb0e`).
Relaunch spot in us-east-2 (auto-starts server, no manual step):
```bash
aws ec2 create-fleet --region us-east-2 --cli-input-json file://fleet_e2.json   # subnets 2a/2b/2c
```
ALL us-east-2 commands need `--region us-east-2`. Query: SSM `curl -s -X POST localhost:8080/generate -d '{"prompt":"..."}'`.

## Capacity note (2026-07-06)
inf2.8xlarge was unfulfillable on **both spot and on-demand across all us-east-1 AZs** that evening.
For spot, launch via EC2 Fleet `gemma4-optb-lt` spanning AZs; if all fail, wait for capacity or retry later.
First boot from the AMI is slow (~20 min, cold-EBS lazy-load) — optionally enable Fast Snapshot Restore
on `snap-0149d318c5f36cfb9` to avoid it.

## Cleanup when fully done
```bash
aws ec2 deregister-image --region us-east-1 --image-id ami-0c13e7feb3fe2e01e
aws ec2 delete-snapshot  --region us-east-1 --snapshot-id snap-0149d318c5f36cfb9
```
