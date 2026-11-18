#!/usr/bin/env python3
"""Gemma 4 E2B as a Lambda GPU function (Lambda Managed Instances), driven through the AWS CLI.

boto3 1.43 has no AcceleratorConfig; aws-cli 2.41 does, so every AWS call is `aws ... --output json`
through subprocess (list args, no shell). tpu.env decides every value; a real environment variable wins.

  lmi.py model      download the checkpoint at the pinned revision into image/model
  lmi.py image      build image/Dockerfile and push it to ECR; records the pushed digest
  lmi.py roles      operator role (EC2 for the capacity provider) and execution role
  lmi.py provider   capacity provider (g6.2xlarge only) in the default VPC; waits for Active
  lmi.py function   create or update the function, set min/max environments, publish, wait for Active
  lmi.py deploy     all of the above in order
  lmi.py up|down    min/max environments to tpu.env's values | to 0 (instances released, version kept)
  lmi.py status     function version, state, scaling config, the capacity provider's instances
  lmi.py ask TEXT   one chat completion through `aws lambda invoke`; prints reply, tokens, seconds
  lmi.py destroy    delete the function and the capacity provider (roles and ECR repo kept)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATE = HERE / ".lmi-state.json"


def load_env(path=HERE / "tpu.env"):
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        env[k.strip()] = os.environ.get(k.strip(), v)
    return env


E = load_env()


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(**kw):
    s = state()
    s.update(kw)
    STATE.write_text(json.dumps(s, indent=1) + "\n")


def run(cmd, check=True, capture=True, **kw):
    print("+", " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""), file=sys.stderr)
    p = subprocess.run(cmd, text=True, capture_output=capture, **kw)
    if check and p.returncode:
        sys.exit(f"failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr}")
    return p


def aws(*args, check=True):
    p = run(["aws", "--region", E["AWS_REGION"], "--output", "json", *args], check=check)
    if p.returncode:
        return None, p.stderr
    return (json.loads(p.stdout) if p.stdout.strip() else {}), ""


def account():
    return aws("sts", "get-caller-identity")[0]["Account"]


# ── model and image ──────────────────────────────────────────────────────────
def cmd_model():
    from huggingface_hub import snapshot_download

    dest = HERE / "image" / "model"
    snapshot_download(
        E["MODEL_NAME"],
        revision=E["MODEL_REVISION"],
        local_dir=dest,
        allow_patterns=["*.json", "*.jinja", "*.safetensors"],
    )
    total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file() and ".cache" not in f.parts)
    print(f"model in {dest}: {total} bytes")


def cmd_image():
    acct = account()
    reg = f"{acct}.dkr.ecr.{E['AWS_REGION']}.amazonaws.com"
    repo = E["ECR_REPO"]
    if aws("ecr", "describe-repositories", "--repository-names", repo, check=False)[0] is None:
        aws("ecr", "create-repository", "--repository-name", repo)
    pw = run(["aws", "--region", E["AWS_REGION"], "ecr", "get-login-password"]).stdout
    run(["docker", "login", "--username", "AWS", "--password-stdin", reg], input=pw)
    tag = f"{reg}/{repo}:{E['MODEL_REVISION'][:12]}"
    # Lambda refuses an OCI image index (what Docker 29 pushes by default, with a provenance attestation):
    # one platform, Docker v2 media types, no attestations, pushed by buildx.
    t0 = time.time()
    run(
        [
            "docker",
            "buildx",
            "build",
            "--platform",
            "linux/amd64",
            "--provenance=false",
            "--sbom=false",
            "--build-arg",
            f"BASE_IMAGE={E['BASE_IMAGE']}",
            "-t",
            tag,
            "--output",
            "type=image,oci-mediatypes=false,push=true",
            str(HERE / "image"),
        ],
        capture=False,
    )
    digest = aws(
        "ecr", "describe-images", "--repository-name", repo, "--image-ids", f"imageTag={tag.rsplit(':', 1)[1]}"
    )[0]["imageDetails"][0]["imageDigest"]
    uri = f"{reg}/{repo}@{digest}"
    save_state(image_uri=uri, push_s=round(time.time() - t0, 1))
    print(uri)


# ── IAM ──────────────────────────────────────────────────────────────────────
TRUST = json.dumps(
    {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}
        ],
    }
)


def ensure_role(name, policy_arn):
    r, _ = aws("iam", "get-role", "--role-name", name, check=False)
    if r is None:
        r, _ = aws("iam", "create-role", "--role-name", name, "--assume-role-policy-document", TRUST)
        time.sleep(10)  # IAM propagation before Lambda may assume it
    aws("iam", "attach-role-policy", "--role-name", name, "--policy-arn", policy_arn)
    return r["Role"]["Arn"]


def cmd_roles():
    op = ensure_role(E["OPERATOR_ROLE"], "arn:aws:iam::aws:policy/AWSLambdaManagedEC2ResourceOperator")
    ex = ensure_role(E["EXECUTION_ROLE"], "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
    save_state(operator_role=op, execution_role=ex)
    print(op, ex, sep="\n")


# ── capacity provider ────────────────────────────────────────────────────────
def default_network():
    subnets = [s for s in E["SUBNET_IDS"].split(",") if s]
    sgs = [s for s in E["SECURITY_GROUP_IDS"].split(",") if s]
    if not subnets or not sgs:
        vpc = aws("ec2", "describe-vpcs", "--filters", "Name=isDefault,Values=true")[0]["Vpcs"][0]["VpcId"]
        if not subnets:
            subnets = [
                s["SubnetId"]
                for s in aws("ec2", "describe-subnets", "--filters", f"Name=vpc-id,Values={vpc}")[0]["Subnets"]
            ]
        if not sgs:
            sgs = [
                g["GroupId"]
                for g in aws(
                    "ec2",
                    "describe-security-groups",
                    "--filters",
                    f"Name=vpc-id,Values={vpc}",
                    "Name=group-name,Values=default",
                )[0]["SecurityGroups"]
            ]
    return subnets, sgs


def get_provider():
    r, _ = aws("lambda", "get-capacity-provider", "--capacity-provider-name", E["CAPACITY_PROVIDER"], check=False)
    return r


def cmd_provider():
    r = get_provider()
    if r is None:
        subnets, sgs = default_network()
        spec = {
            "CapacityProviderName": E["CAPACITY_PROVIDER"],
            "VpcConfig": {"SubnetIds": subnets, "SecurityGroupIds": sgs},
            "PermissionsConfig": {"CapacityProviderOperatorRoleArn": state()["operator_role"]},
            "InstanceRequirements": {"Architectures": ["x86_64"], "AllowedInstanceTypes": [E["INSTANCE_TYPE"]]},
            "CapacityProviderScalingConfig": {"ScalingMode": "Auto", "MaxVCpuCount": int(E["MAX_VCPU_COUNT"])},
        }
        aws("lambda", "create-capacity-provider", "--cli-input-json", json.dumps(spec))
    for _ in range(60):
        r = get_provider()
        cp = r.get("CapacityProvider", r)
        st = cp.get("State")
        if st in ("Active", "ACTIVE"):
            break
        if st and "FAIL" in st.upper():
            sys.exit(json.dumps(cp, indent=1, default=str))
        time.sleep(10)
    save_state(capacity_provider_arn=cp["CapacityProviderArn"])
    print(cp["CapacityProviderArn"], cp.get("State"))


# ── function ─────────────────────────────────────────────────────────────────
def server_args():
    a = ["--max-num-seqs", E["MAX_NUM_SEQS"]]
    if E.get("LANGUAGE_MODEL_ONLY", "").lower() == "true":
        a.append("--language-model-only")
    if E.get("TOOL_CALL_PARSER"):
        a += ["--enable-auto-tool-choice", "--tool-call-parser", E["TOOL_CALL_PARSER"]]
    if E.get("REASONING_PARSER"):
        a += ["--reasoning-parser", E["REASONING_PARSER"]]
    return " ".join(a)


def function_env():
    return {
        "Variables": {
            "MODEL_ID": "/opt/model",
            "SERVED_MODEL_NAME": E["MODEL_NAME"],
            "VLLM_GPU_MEM_UTIL": E["GPU_MEMORY_UTILIZATION"],
            "VLLM_MAX_MODEL_LEN": E["MAX_MODEL_LEN"],
            "VLLM_SERVER_TIMEOUT": E["VLLM_SERVER_TIMEOUT"],
            "VLLM_SERVER_ARGS": server_args(),
        }
    }


def wait_function(qualifier=None, want_update=True):
    args = ["lambda", "get-function-configuration", "--function-name", E["FUNCTION_NAME"]]
    if qualifier:
        args += ["--qualifier", qualifier]
    t0 = time.time()
    while True:
        c = aws(*args)[0]
        st, lu = c.get("State"), c.get("LastUpdateStatus")
        if st == "Failed" or lu == "Failed":
            sys.exit(f"{st} / {lu}: {c.get('StateReason')} {c.get('LastUpdateStatusReason')}")
        # An unpublished LMI function sits at ActiveNonInvocable; a published version turns Active.
        ready = ("Active",) if qualifier else ("Active", "ActiveNonInvocable")
        if st in ready and (not want_update or lu in (None, "Successful")):
            return c, round(time.time() - t0, 1)
        time.sleep(15)


def scaling(lo, hi):
    aws(
        "lambda",
        "put-function-scaling-config",
        "--function-name",
        E["FUNCTION_NAME"],
        "--qualifier",
        "$LATEST.PUBLISHED",
        "--function-scaling-config",
        json.dumps({"MinExecutionEnvironments": int(lo), "MaxExecutionEnvironments": int(hi)}),
    )


def cmd_function():
    s = state()
    cpc = {
        "LambdaManagedInstancesCapacityProviderConfig": {
            "CapacityProviderArn": s["capacity_provider_arn"],
            "PerExecutionEnvironmentMaxConcurrency": int(E["PER_ENV_MAX_CONCURRENCY"]),
            "ExecutionEnvironmentMemoryGiBPerVCpu": float(E["MEMORY_GIB_PER_VCPU"]),
        }
    }
    have, _ = aws("lambda", "get-function", "--function-name", E["FUNCTION_NAME"], check=False)
    if have is None:
        spec = {
            "FunctionName": E["FUNCTION_NAME"],
            "PackageType": "Image",
            "Code": {"ImageUri": s["image_uri"]},
            "Role": s["execution_role"],
            "Architectures": ["x86_64"],
            "Timeout": int(E["TIMEOUT_S"]),
            "MemorySize": int(E["MEMORY_MB"]),
            "EphemeralStorage": {"Size": int(E["EPHEMERAL_MB"])},
            "Environment": function_env(),
            "CapacityProviderConfig": cpc,
            "AcceleratorConfig": {"AcceleratorMemorySize": int(E["ACCELERATOR_MEMORY_GB"])},
        }
        aws("lambda", "create-function", "--cli-input-json", json.dumps(spec))
    else:
        wait_function()
        aws("lambda", "update-function-code", "--function-name", E["FUNCTION_NAME"], "--image-uri", s["image_uri"])
        wait_function()
        aws(
            "lambda",
            "update-function-configuration",
            "--cli-input-json",
            json.dumps(
                {
                    "FunctionName": E["FUNCTION_NAME"],
                    "Timeout": int(E["TIMEOUT_S"]),
                    "MemorySize": int(E["MEMORY_MB"]),
                    "EphemeralStorage": {"Size": int(E["EPHEMERAL_MB"])},
                    "Environment": function_env(),
                    "CapacityProviderConfig": cpc,
                    "AcceleratorConfig": {"AcceleratorMemorySize": int(E["ACCELERATOR_MEMORY_GB"])},
                }
            ),
        )
    wait_function()
    t0 = time.time()
    v = aws("lambda", "publish-version", "--function-name", E["FUNCTION_NAME"], "--publish-to", "LATEST_PUBLISHED")[0]
    ver = v["Version"]
    # Lambda's default is 3 environments across AZs; set ours on the version before it scales up.
    try_scaling = aws(
        "lambda",
        "put-function-scaling-config",
        "--function-name",
        E["FUNCTION_NAME"],
        "--qualifier",
        ver,
        "--function-scaling-config",
        json.dumps({"MinExecutionEnvironments": int(E["MIN_ENVS"]), "MaxExecutionEnvironments": int(E["MAX_ENVS"])}),
        check=False,
    )
    c, _ = wait_function(ver, want_update=False)
    active_s = round(time.time() - t0, 1)
    save_state(version=ver, publish_to_active_s=active_s, published_at=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)))
    print(f"version {ver} {c['State']} {active_s} s after publish; scaling set: {try_scaling[0] is not None}")


def cmd_up():
    scaling(E["MIN_ENVS"], E["MAX_ENVS"])
    cmd_status()


def cmd_down():
    scaling(0, 0)
    cmd_status()


def cmd_status():
    c = aws(
        "lambda",
        "get-function-configuration",
        "--function-name",
        E["FUNCTION_NAME"],
        "--qualifier",
        "$LATEST.PUBLISHED",
        check=False,
    )[0]
    if c is None:
        print("no function")
        return
    print(
        json.dumps(
            {
                k: c.get(k)
                for k in (
                    "Version",
                    "State",
                    "StateReason",
                    "LastUpdateStatus",
                    "MemorySize",
                    "AcceleratorConfig",
                    "CodeSha256",
                )
            },
            indent=1,
        )
    )
    sc = aws(
        "lambda",
        "get-function-scaling-config",
        "--function-name",
        E["FUNCTION_NAME"],
        "--qualifier",
        "$LATEST.PUBLISHED",
        check=False,
    )[0]
    print("scaling", json.dumps(sc))
    inst = aws(
        "ec2",
        "describe-instances",
        "--filters",
        "Name=instance-state-name,Values=pending,running",
        f"Name=instance-type,Values={E['INSTANCE_TYPE']}",
    )[0]
    for r in inst["Reservations"]:
        for i in r["Instances"]:
            print(
                "instance",
                i["InstanceId"],
                i["InstanceType"],
                i["Placement"]["AvailabilityZone"],
                i["State"]["Name"],
                i.get("LaunchTime"),
            )


# ── invoke ───────────────────────────────────────────────────────────────────
def invoke(body, out):
    t0 = time.time()
    p = run(
        [
            "aws",
            "--region",
            E["AWS_REGION"],
            "lambda",
            "invoke",
            "--function-name",
            f"{E['FUNCTION_NAME']}:$LATEST.PUBLISHED",
            "--cli-binary-format",
            "raw-in-base64-out",
            "--cli-read-timeout",
            "0",
            "--payload",
            json.dumps(body),
            str(out),
        ],
        check=False,
    )
    dt = time.time() - t0
    meta = json.loads(p.stdout) if p.stdout.strip().startswith("{") else {"stderr": p.stderr}
    reply = json.loads(Path(out).read_text()) if Path(out).exists() else None
    return dt, meta, reply


def cmd_ask(text, think=False, max_tokens=512):
    body = {
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    out = HERE / ".ask-reply.json"
    out.unlink(missing_ok=True)
    dt, meta, reply = invoke(body, out)
    if not reply or "choices" not in reply:
        sys.exit(json.dumps({"meta": meta, "reply": reply}, indent=1))
    m = reply["choices"][0]["message"]
    if m.get("reasoning") or m.get("reasoning_content"):
        print("[thinking]", (m.get("reasoning") or m.get("reasoning_content"))[:400], "...\n")
    print(m.get("content"))
    u = reply.get("usage", {})
    print(
        f"\n{u.get('prompt_tokens')} in / {u.get('completion_tokens')} out, {dt:.2f} s round trip "
        f"(aws cli start-up included), model {reply.get('model')}"
    )


def cmd_destroy():
    aws("lambda", "delete-function", "--function-name", E["FUNCTION_NAME"], check=False)
    for _ in range(40):
        if (
            aws("lambda", "delete-capacity-provider", "--capacity-provider-name", E["CAPACITY_PROVIDER"], check=False)[
                0
            ]
            is not None
            or get_provider() is None
        ):
            break
        time.sleep(15)
    cmd_status()


def main(argv):
    if not argv:
        sys.exit(__doc__)
    c, rest = argv[0], argv[1:]
    if c == "ask":
        think = "--think" in rest
        cmd_ask(" ".join(a for a in rest if a != "--think") or "Say hello in five words.", think=think)
    elif c == "deploy":
        for f in (cmd_roles, cmd_image, cmd_provider, cmd_function):
            f()
    else:
        {
            "model": cmd_model,
            "image": cmd_image,
            "roles": cmd_roles,
            "provider": cmd_provider,
            "function": cmd_function,
            "up": cmd_up,
            "down": cmd_down,
            "status": cmd_status,
            "destroy": cmd_destroy,
        }[c]()


if __name__ == "__main__":
    main(sys.argv[1:])
