#!/usr/bin/env python3
"""Gemma 4 E2B as a Lambda GPU function (Lambda Managed Instances), driven through the AWS CLI.

Shared by the lambda-*-g6-* rigs (one copy per rig, kept identical); ENGINE in tpu.env picks the serving
engine's settings: vllm or sglang (AWS's Lambda images and their engine extension) or cmd (AWS's base or
PyTorch Lambda image plus this rig's image/engine-ext, which runs ENGINE_CMD).

boto3 1.43 has no AcceleratorConfig; aws-cli 2.41 does, so every AWS call is `aws ... --output json`
through subprocess (list args, no shell). tpu.env decides every value; a real environment variable wins.

  lmi.py image      build image/Dockerfile on the EC2 builder (builder.py) and push it to ECR; records the digest
  lmi.py roles      operator role (EC2 for the capacity provider) and execution role
  lmi.py provider   capacity provider (g6.2xlarge only) in the default VPC; waits for Active
  lmi.py function   create or update the function, set min/max environments, publish, wait for Active
  lmi.py deploy     all of the above in order
  lmi.py up|down    min/max environments to tpu.env's values | to 0 (instances released, version kept)
  lmi.py status     function version, state, scaling config, the capacity provider's instances
  lmi.py ask TEXT   one chat completion through `aws lambda invoke`; prints reply, tokens, seconds
  lmi.py args       print the function's environment (the engine's flags) without calling AWS
  lmi.py destroy    delete the function (the capacity provider is shared with the other lambda-*-g6 rigs)
"""

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


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
# One state file per region (image digest, provider ARN, timings): AWS_REGION=us-west-2 make deploy keeps
# us-east-1's. us-east-1 keeps the original name.
STATE = HERE / (".lmi-state.json" if E["AWS_REGION"] == "us-east-1" else f".lmi-state.{E['AWS_REGION']}.json")


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


# ── image ────────────────────────────────────────────────────────────────────
def build_args():
    """--build-arg values for image/Dockerfile: the tpu.env keys named in BUILD_ARGS."""
    return {k: E[k] for k in E.get("BUILD_ARGS", "").split(",") if k}


def cmd_image():
    import builder

    t0 = time.time()
    uri = builder.build_image(HERE, E["ECR_REPO"], E["IMAGE_TAG"], build_args())
    save_state(image_uri=uri, build_and_push_s=round(time.time() - t0, 1))
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


def instance_types():
    return [t for t in E.get("ALLOWED_INSTANCE_TYPES", E["INSTANCE_TYPE"]).split(",") if t]


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
            "InstanceRequirements": {"Architectures": ["x86_64"], "AllowedInstanceTypes": instance_types()},
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
def flag(name):
    return E.get(name, "").lower() in ("1", "true", "yes", "on")


def vllm_args():
    a = ["--max-num-seqs", E["MAX_NUM_SEQS"]]
    if flag("LANGUAGE_MODEL_ONLY"):
        a.append("--language-model-only")
    if E.get("TOOL_CALL_PARSER"):
        a += ["--enable-auto-tool-choice", "--tool-call-parser", E["TOOL_CALL_PARSER"]]
    if E.get("REASONING_PARSER"):
        a += ["--reasoning-parser", E["REASONING_PARSER"]]
    if E.get("SPECULATIVE_MODEL"):
        spec = {"model": E["SPECULATIVE_MODEL"], "num_speculative_tokens": int(E["NUM_SPECULATIVE_TOKENS"])}
        a += ["--speculative-config", json.dumps(spec)]
    return a


def sglang_args():
    """gpu-sglang-g6-2b-q4w4a16emb4/serve.py's flags, less the ones AWS's engine extension already passes."""
    a = ["--context-length", E["MAX_MODEL_LEN"], "--max-running-requests", E["MAX_NUM_SEQS"]]
    if E.get("TOOL_CALL_PARSER"):
        a += ["--tool-call-parser", E["TOOL_CALL_PARSER"]]
    if E.get("REASONING_PARSER"):
        a += ["--reasoning-parser", E["REASONING_PARSER"]]
    if flag("DISABLE_RADIX_CACHE"):
        a.append("--disable-radix-cache")
    if flag("DISABLE_PREFILL_CUDA_GRAPH"):
        a.append("--disable-prefill-cuda-graph")
    if E.get("CUDA_GRAPH_MAX_BS_DECODE"):
        a += ["--cuda-graph-max-bs-decode", E["CUDA_GRAPH_MAX_BS_DECODE"]]
    if E.get("SPECULATIVE_MODEL"):
        n = int(E["SPECULATIVE_NUM_STEPS"])
        a += ["--speculative-algorithm", "NEXTN", "--speculative-draft-model-path", E["SPECULATIVE_MODEL"],
              "--speculative-num-steps", str(n), "--speculative-eagle-topk", "1",
              "--speculative-num-draft-tokens", str(n + 1)]
    return a


def server_args():
    """The engine's flags as one shell-quoted string (the extensions split it with shlex.split)."""
    return shlex.join({"vllm": vllm_args, "sglang": sglang_args}[E["ENGINE"]]())


def function_env():
    eng = E["ENGINE"]
    v = {"SERVED_MODEL_NAME": E["MODEL_NAME"]}
    if eng == "vllm":
        v.update(
            MODEL_ID="/opt/model",
            VLLM_GPU_MEM_UTIL=E["GPU_MEMORY_UTILIZATION"],
            VLLM_MAX_MODEL_LEN=E["MAX_MODEL_LEN"],
            VLLM_SERVER_TIMEOUT=E["SERVER_TIMEOUT"],
            VLLM_SERVER_ARGS=server_args(),
        )
    elif eng == "sglang":
        v.update(
            MODEL_ID="/opt/model",
            SGLANG_MEM_FRACTION=E["MEM_FRACTION_STATIC"],
            SGLANG_ATTENTION_BACKEND=E["ATTENTION_BACKEND"],
            SGLANG_SERVER_TIMEOUT=E["SERVER_TIMEOUT"],
            SGLANG_SERVER_ARGS=server_args(),
        )
    elif eng == "cmd":
        v.update(
            ENGINE_CMD=E["ENGINE_CMD"],
            ENGINE_PORT=E["ENGINE_PORT"],
            ENGINE_HEALTH_PATH=E.get("ENGINE_HEALTH_PATH", "/health"),
            ENGINE_SERVER_TIMEOUT=E["SERVER_TIMEOUT"],
        )
    else:
        sys.exit(f"ENGINE={eng}: expected vllm, sglang or cmd")
    # Extra variables for the engine process itself, as KEY=VALUE pairs separated by spaces.
    for kv in shlex.split(E.get("ENGINE_ENV", "")):
        k, val = kv.split("=", 1)
        v[k] = val
    return {"Variables": v}


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
    # Lambda refuses the call for a moment after publish; until it lands the version scales to 3 environments,
    # more than the vCPU quota allows (2026-11-19: the other rig's publish then failed for EC2 capacity).
    for _ in range(40):
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
        if try_scaling[0] is not None:
            break
        print("scaling not accepted yet:", try_scaling[1].strip()[-200:], file=sys.stderr)
        time.sleep(3)
    # A republish keeps the old environments serving while new ones start: the version reads Active at
    # once, and LastUpdateStatus stays InProgress until the new environments are up.
    c, _ = wait_function(ver)
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
        f"Name=instance-type,Values={','.join(instance_types())}",
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


def cmd_destroy(provider=False):
    """Delete the function; with --provider also the capacity provider, which Lambda refuses while
    another rig's function still uses it."""
    aws("lambda", "delete-function", "--function-name", E["FUNCTION_NAME"], check=False)
    for _ in range(40 if provider else 0):
        r, err = aws("lambda", "delete-capacity-provider", "--capacity-provider-name", E["CAPACITY_PROVIDER"],
                     check=False)
        if r is not None or get_provider() is None:
            break
        print(err.strip()[:200], file=sys.stderr)
        time.sleep(15)
    cmd_status()


def main(argv):
    if not argv:
        sys.exit(__doc__)
    c, rest = argv[0], argv[1:]
    if c == "ask":
        think = "--think" in rest
        cmd_ask(" ".join(a for a in rest if a != "--think") or "Say hello in five words.", think=think)
    elif c == "destroy":
        cmd_destroy(provider="--provider" in rest)
    elif c == "args":
        print(json.dumps(function_env(), indent=1))
    elif c == "deploy":
        for f in (cmd_roles, cmd_image, cmd_provider, cmd_function):
            f()
    else:
        {
            "image": cmd_image,
            "roles": cmd_roles,
            "provider": cmd_provider,
            "function": cmd_function,
            "up": cmd_up,
            "down": cmd_down,
            "status": cmd_status,
        }[c]()


if __name__ == "__main__":
    main(sys.argv[1:])
