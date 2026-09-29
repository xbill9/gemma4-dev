"""Serve this rig on one on-demand G6 and run sagemaker-gemma's compare.py against it.

    python3 ec2_measure.py <run-dir> <endpoint-name> [model-id]

The measurement runs ON the instance against localhost (the tree's convention:
no WAN in the numbers, and no inbound rule needed). The instance gets a security
group with no inbound rules; Systems Manager is the only way in. It is terminated
in a `finally`, whatever happens, and `watchdog` in the run dir covers a killed
driver. Every step is timestamped into <run-dir>/timeline.txt.
"""

import asyncio
import base64
import gzip
import io
import os
import re
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
for line in (HERE / "tpu.env").read_text().splitlines():
    if line and not line.startswith("#") and "=" in line:
        key, _, value = line.partition("=")
        os.environ.setdefault(key, value)

import server  # noqa: E402

COMPARE_DIR = Path(os.getenv("COMPARE_DIR", Path.home() / "sagemaker-gemma"))
PROFILE = os.getenv("INSTANCE_PROFILE", "g6-vllm-instance-profile")
SG_NAME = "gpu-vllm-g6-ssm-only"
READY_TIMEOUT_S = 45 * 60


def stamp(run: Path, msg: str) -> None:
    line = f"{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}"
    print(line, flush=True)
    with open(run / "timeline.txt", "a") as f:
        f.write(line + "\n")


def network(run: Path) -> tuple[str, str]:
    ec2 = server._client("ec2")
    vpc = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    subnet = ec2.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpc]}, {"Name": "default-for-az", "Values": ["true"]}]
    )["Subnets"][0]["SubnetId"]
    found = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [SG_NAME]}, {"Name": "vpc-id", "Values": [vpc]}]
    )["SecurityGroups"]
    if found:
        sg = found[0]["GroupId"]
        if found[0]["IpPermissions"]:
            raise SystemExit(f"{SG_NAME} ({sg}) has inbound rules; refusing to use it")
    else:
        sg = ec2.create_security_group(
            GroupName=SG_NAME, VpcId=vpc, Description="G6 vLLM rigs: no inbound, reached over SSM only"
        )["GroupId"]
        stamp(run, f"created security group {sg} ({SG_NAME}, no inbound rules)")
    return subnet, sg


async def wait_ready(run: Path, iid: str) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    last = ""
    while time.monotonic() < deadline:
        try:
            out = await server._ssm(
                iid,
                "grep -o '\\[stage\\] [^ ]*' /var/log/cloud-init-output.log | tail -1; "
                "curl -sf -o /dev/null localhost:8000/health && echo READY || true",
                timeout=60,
            )
        except Exception as exc:  # SSM agent not registered yet
            out = f"ssm: {type(exc).__name__}"
        if out != last:
            stamp(run, out.replace("\n", " | "))
            last = out
        if "READY" in out:
            return
        await asyncio.sleep(30)
    raise TimeoutError("vLLM not healthy within the ready timeout")


def bundle() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ("compare.py", "sm.py"):
            tar.add(COMPARE_DIR / name, arcname=name)
    return base64.b64encode(buf.getvalue()).decode()


def unpack(text: str) -> str:
    return gzip.decompress(base64.b64decode(text.strip())).decode()


async def main(run: Path, name: str, model: str) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "settings.txt").write_text(
        "".join(
            f"{k}={getattr(server, k)}\n"
            for k in (
                "RIG_NAME",
                "AWS_REGION",
                "INSTANCE_TYPE",
                "VLLM_IMAGE",
                "MAX_MODEL_LEN",
                "GPU_MEMORY_UTILIZATION",
                "MAX_NUM_SEQS",
                "EXTRA_VLLM_ARGS",
            )
        )
        + f"MODEL={model}\nSERVE_FLAGS={server._serve_flags(model, server.INSTANCE_TYPE)}\n"
    )
    subnet, sg = network(run)
    stamp(run, f"launch {server.INSTANCE_TYPE} on-demand, subnet {subnet}, sg {sg}, model {model}")
    out = await server.create_g6_instance(subnet, sg, PROFILE, name=name, model_name=model, spot=False)
    stamp(run, out.replace("\n", " | "))
    m = re.search(r"i-[0-9a-f]{8,}", out)
    if not m:
        raise SystemExit("launch failed")
    iid = m.group(0)
    (run / "instance-id.txt").write_text(iid + "\n")
    try:
        await wait_ready(run, iid)
        stamp(run, "vLLM healthy; copying compare.py")
        await server._ssm(iid, f"mkdir -p /opt/bench && echo {bundle()} | base64 -d | tar xz -C /opt/bench")
        py = await server._ssm(iid, "python3 --version")
        stamp(run, f"measuring on the instance ({py})")
        result = await server._ssm(
            iid,
            f"cd /opt/bench && COMPARE_DOCKER_CONTAINER={server.SERVICE_NAME} "
            f"python3 compare.py measure /opt/bench {name}@http://localhost:8000 > measure.log 2>&1; "
            f"gzip -c measure-{name}.json | base64 -w0",
            timeout=1800,
        )
        (run / f"measure-{name}.json").write_text(unpack(result))
        log = await server._ssm(iid, "gzip -c /opt/bench/measure.log | base64 -w0")
        (run / f"measure-{name}.log").write_text(unpack(log))
        vlog = await server._ssm(
            iid,
            f"docker logs {server.SERVICE_NAME} 2>&1 | grep -v '#015' | grep -vE 'Avg prompt throughput: 0.0 tokens/s, "
            f"Avg generation throughput: 0.0' | gzip -c | base64 -w0",
            timeout=120,
        )
        (run / "vllm-engine.log").write_text(unpack(vlog))
        boot = await server._ssm(iid, "gzip -c /var/log/cloud-init-output.log | base64 -w0")
        (run / "bootstrap.log").write_text(unpack(boot))
        stamp(run, "measured")
    finally:
        stamp(run, (await server.terminate_g6_instance(iid)).replace("\n", " | "))
        ec2 = server._client("ec2")
        ec2.get_waiter("instance_terminated").wait(InstanceIds=[iid])
        stamp(run, f"{iid} terminated")


if __name__ == "__main__":
    run_dir, endpoint = Path(sys.argv[1]), sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else server.MODEL_NAME
    asyncio.run(main(run_dir, endpoint, model))
