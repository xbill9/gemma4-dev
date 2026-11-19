#!/usr/bin/env python3
"""The EC2 instance that builds the lambda-*-g6-* images and pushes them to ECR.

The images are 15-40 GB, more than a laptop disk holds, and ECR is in the same region as the builder.
One CPU instance (no GPU, so the G-instance vCPU quota stays for Lambda) serves every rig; it is found
by its Name tag, so any rig's copy of this file starts, uses and stops the same instance. Shared by
the lambda-*-g6-* rigs (one copy per rig, kept identical); every value comes from the rig's tpu.env.

  builder.py up        role, instance profile, bucket, instance; waits for SSM
  builder.py status    the instance, if any
  builder.py run CMD   one shell command on the builder through SSM; prints its output
  builder.py down      terminate the instance (role, bucket and ECR repos kept)

lmi.py image calls build_image(): the rig's image/ directory goes to S3 as a tarball, the builder
downloads it, builds with buildx (Docker v2 media types, one platform, no attestations: what Lambda
accepts) and pushes. A Hugging Face token from Secrets Manager (HF_SECRET_ID) reaches the Dockerfile
as the build secret `hf`, never as a layer.
"""

import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import lmi

E = lmi.E
NAME = "lambda-image-builder"
TRUST_EC2 = json.dumps(
    {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}],
    }
)
POLICIES = (
    "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser",
    "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
)
USER_DATA = """#!/bin/bash
set -eux
dnf install -y docker git
systemctl enable --now docker
if ! docker buildx version; then
  mkdir -p /usr/libexec/docker/cli-plugins
  curl -fsSL https://github.com/docker/buildx/releases/download/v0.19.3/buildx-v0.19.3.linux-amd64 \\
    -o /usr/libexec/docker/cli-plugins/docker-buildx
  chmod +x /usr/libexec/docker/cli-plugins/docker-buildx
fi
docker buildx create --name lmi --driver docker-container --use
docker buildx inspect --bootstrap
touch /var/lib/builder-ready
"""


def bucket():
    return f"{NAME}-{lmi.account()}-{E['AWS_REGION']}"


def find():
    r = lmi.aws(
        "ec2",
        "describe-instances",
        "--filters",
        f"Name=tag:Name,Values={NAME}",
        "Name=instance-state-name,Values=pending,running",
    )[0]
    inst = [i for res in r["Reservations"] for i in res["Instances"]]
    return inst[0] if inst else None


def ensure_profile():
    if lmi.aws("iam", "get-role", "--role-name", NAME, check=False)[0] is None:
        lmi.aws("iam", "create-role", "--role-name", NAME, "--assume-role-policy-document", TRUST_EC2)
    for p in POLICIES:
        lmi.aws("iam", "attach-role-policy", "--role-name", NAME, "--policy-arn", p)
    if lmi.aws("iam", "get-instance-profile", "--instance-profile-name", NAME, check=False)[0] is None:
        lmi.aws("iam", "create-instance-profile", "--instance-profile-name", NAME)
        lmi.aws("iam", "add-role-to-instance-profile", "--instance-profile-name", NAME, "--role-name", NAME)
        time.sleep(15)  # a new instance profile is not usable by run-instances at once
    if not E.get("HF_SECRET_ID"):
        return
    secret = lmi.aws("secretsmanager", "describe-secret", "--secret-id", E["HF_SECRET_ID"])[0]["ARN"]
    lmi.aws(
        "iam",
        "put-role-policy",
        "--role-name",
        NAME,
        "--policy-name",
        "hf-token",
        "--policy-document",
        json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Action": "secretsmanager:GetSecretValue", "Resource": secret}],
            }
        ),
    )


def ensure_bucket():
    b = bucket()
    if lmi.aws("s3api", "head-bucket", "--bucket", b, check=False)[0] is None:
        lmi.aws("s3api", "create-bucket", "--bucket", b)
        lmi.aws(
            "s3api",
            "put-public-access-block",
            "--bucket",
            b,
            "--public-access-block-configuration",
            "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true",
        )
    return b


def cmd_up():
    ensure_profile()
    ensure_bucket()
    i = find()
    if i is None:
        ami = lmi.aws(
            "ssm",
            "get-parameter",
            "--name",
            "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64",
        )[0]["Parameter"]["Value"]
        bdm = [
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": int(E["BUILDER_DISK_GB"]),
                    "VolumeType": "gp3",
                    "Iops": 6000,
                    "Throughput": 1000,
                    "DeleteOnTermination": True,
                },
            }
        ]
        r = lmi.aws(
            "ec2",
            "run-instances",
            "--image-id",
            ami,
            "--instance-type",
            E["BUILDER_INSTANCE_TYPE"],
            "--iam-instance-profile",
            f"Name={NAME}",
            "--block-device-mappings",
            json.dumps(bdm),
            "--user-data",
            USER_DATA,
            "--metadata-options",
            "HttpTokens=required",
            "--tag-specifications",
            f"ResourceType=instance,Tags=[{{Key=Name,Value={NAME}}}]",
            "--count",
            "1",
        )[0]
        i = r["Instances"][0]
    iid = i["InstanceId"]
    print("instance", iid, i["InstanceType"], flush=True)
    t0 = time.time()
    while time.time() - t0 < 900:
        info = lmi.aws(
            "ssm", "describe-instance-information", "--filters", f"Key=InstanceIds,Values={iid}", check=False
        )[0]
        if info and info.get("InstanceInformationList"):
            out = run_remote("test -f /var/lib/builder-ready && docker buildx ls | head -5", timeout=120, check=False)
            if "lmi" in out:
                print(f"ready after {time.time() - t0:.0f} s")
                return iid
        time.sleep(20)
    sys.exit("builder not ready after 15 min")


def cmd_status():
    i = find()
    print(json.dumps(i and {k: i.get(k) for k in ("InstanceId", "InstanceType", "LaunchTime")}, default=str))


def cmd_down():
    i = find()
    if i:
        lmi.aws("ec2", "terminate-instances", "--instance-ids", i["InstanceId"])
        print("terminated", i["InstanceId"])
    else:
        print("no builder")


def run_remote(script, timeout=600, check=True, quiet=True):
    """Run a bash script on the builder as root; return its stdout (SSM keeps at most 24,000 characters)."""
    i = find()
    if i is None:
        sys.exit("no builder: builder.py up")
    params = {"commands": [script], "executionTimeout": [str(timeout)]}
    c = lmi.aws(
        "ssm",
        "send-command",
        "--instance-ids",
        i["InstanceId"],
        "--document-name",
        "AWS-RunShellScript",
        "--timeout-seconds",
        "600",
        "--parameters",
        json.dumps(params),
    )[0]["Command"]["CommandId"]
    while True:
        time.sleep(5)
        r, _ = lmi.aws(
            "ssm", "get-command-invocation", "--command-id", c, "--instance-id", i["InstanceId"], check=False
        )
        if r and r["Status"] not in ("Pending", "InProgress", "Delayed"):
            break
    out = r.get("StandardOutputContent", "")
    if r["Status"] != "Success" and check:
        sys.exit(f"remote command {r['Status']}:\n{out[-4000:]}\n{r.get('StandardErrorContent', '')[-4000:]}")
    return out


def _tarball(src: Path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(src, arcname=".", filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return buf.getvalue()


def build_image(rig: Path, repo, tag, build_args):
    """Build rig/image on the builder, push to ECR as repo:tag; return the repo@digest URI."""
    acct = lmi.account()
    region = E["AWS_REGION"]
    reg = f"{acct}.dkr.ecr.{region}.amazonaws.com"
    if lmi.aws("ecr", "describe-repositories", "--repository-names", repo, check=False)[0] is None:
        lmi.aws("ecr", "create-repository", "--repository-name", repo)
    b = ensure_bucket()
    key = f"ctx/{repo}-{tag}.tgz"
    tgz = rig / ".image-context.tgz"
    tgz.write_bytes(_tarball(rig / "image"))
    lmi.run(["aws", "--region", region, "s3", "cp", "--only-show-errors", str(tgz), f"s3://{b}/{key}"])
    tgz.unlink()
    bargs = " ".join(f"--build-arg {k}='{v}'" for k, v in build_args.items())
    log = f"/build/{repo}.log"
    w = f"/build/{repo}"
    token, secret = "", ""
    if E.get("HF_SECRET_ID"):
        token = (
            f"export HF_TOKEN=$(aws secretsmanager get-secret-value --secret-id {E['HF_SECRET_ID']} "
            "--query SecretString --output text | python3 -c \"import json,sys; s=sys.stdin.read().strip(); "
            "d=json.loads(s) if s.startswith('{') else None; "
            "print(s if d is None else d.get('HF_TOKEN', next(iter(d.values()))))\")\n"
        )
        secret = "--secret id=hf,env=HF_TOKEN "
    script = f"""set -euo pipefail
export AWS_DEFAULT_REGION={region}
W=/build/{repo}; rm -rf $W; mkdir -p $W
aws s3 cp --only-show-errors s3://{b}/{key} - | tar xz -C $W
aws ecr get-login-password | docker login --username AWS --password-stdin {reg} >/dev/null
{token}cat > $W.sh <<'BUILD'
docker buildx build --builder lmi --platform linux/amd64 --provenance=false --sbom=false \
  {secret}{bargs} -t {reg}/{repo}:{tag} \
  --output type=image,oci-mediatypes=false,push=true {w}
echo BUILD_EXIT=$?
BUILD
nohup bash $W.sh > {log} 2>&1 &
echo started
"""
    run_remote(script, timeout=300)
    print(f"building on the builder; log {log}", flush=True)
    t0 = time.time()
    last = ""
    while True:
        time.sleep(30)
        out = run_remote(f"grep -ao 'BUILD_EXIT=[0-9]*' {log} || echo RUNNING; tail -c 600 {log}", timeout=60)
        state, _, tail = out.partition("\n")
        line = (tail.strip().splitlines() or [""])[-1][:160]
        if line != last:
            print(f"  {time.time() - t0:5.0f} s  {line}", flush=True)
            last = line
        if state.startswith("BUILD_EXIT="):
            break
    if state.strip() != "BUILD_EXIT=0":
        sys.exit(f"build failed ({state.strip()}):\n{run_remote(f'tail -c 8000 {log}', timeout=60)}")
    d = lmi.aws("ecr", "describe-images", "--repository-name", repo, "--image-ids", f"imageTag={tag}")[0]
    det = d["imageDetails"][0]
    pushed = det.get("imagePushedAt", "")
    print(f"pushed {repo}:{tag} {det['imageDigest']} {det.get('imageSizeInBytes')} bytes at {pushed}")
    return f"{reg}/{repo}@{det['imageDigest']}"


def main(argv):
    if not argv:
        sys.exit(__doc__)
    if argv[0] == "run":
        print(run_remote(" ".join(argv[1:]), check=False))
        return
    {"up": cmd_up, "status": cmd_status, "down": cmd_down}[argv[0]]()


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main(sys.argv[1:])
