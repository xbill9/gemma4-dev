#!/usr/bin/env python3
"""Boot-time and warm-reload campaign across the three g5g runtimes.

Start line for a cold boot is the moment run_instances RETURNS an instance id --
capacity wait is excluded deliberately, it measures AWS not the rig. Stop line is
HTTP 200 on /health, then a first chat completion is timed separately because
"health 200" and "can actually serve" are not the same instant on every runtime.

One instance at a time: two rigs downloading 9.5 GB at once would contend for
network and corrupt exactly the number being measured.

Results append to boot_results.json after EVERY measurement, so a crash at hour
two does not lose hour one.
"""
import asyncio, json, os, subprocess, sys, time
import urllib.request, urllib.error
import logging
import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, "/home/xbill/gemma4-dev/gpu-pytorch-g5g-2b")
logging.disable(logging.ERROR)

OUT = "/tmp/claude-1000/-home-xbill-gemma4-dev-gpu-pytorch-g5g-2b/dc2377d3-309f-45f7-adc1-d87ec3922a6a/scratchpad/boot_results_v2.json"
SUBNETS = [("us-east-1a","subnet-061a363014b302012"),("us-east-1d","subnet-0d2b2038965e8d0c0"),
           ("us-east-1b","subnet-0740fc9b7195785c3"),("us-east-1c","subnet-0c2872fe4182b9ec1")]
SG, PROFILE = "sg-01ee54036d37aa770", "g5g-jax-instance-profile"
REPEATS = 3
ec2 = boto3.client("ec2", region_name="us-east-1")
results = []
if os.path.exists(OUT):
    results = json.load(open(OUT))
    print(f"resuming: {len(results)} measurement(s) already recorded", flush=True)


def already_done(rig, kind, rep):
    return any(r["rig"] == rig and r["kind"] == kind and r["rep"] == rep
               and r.get("t_health_s") for r in results)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save():
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2)


def ssm(iid, commands, timeout=120):
    r = subprocess.run(
        ["aws","ssm","send-command","--region","us-east-1","--instance-ids",iid,
         "--document-name","AWS-RunShellScript","--parameters",
         json.dumps({"commands": commands}), "--query","Command.CommandId","--output","text"],
        capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def ssm_wait(iid, cid, timeout=600):
    end = time.time() + timeout
    while time.time() < end:
        r = subprocess.run(["aws","ssm","get-command-invocation","--region","us-east-1",
                            "--command-id",cid,"--instance-id",iid,
                            "--query","[Status,StandardOutputContent]","--output","text"],
                           capture_output=True, text=True, timeout=90)
        parts = r.stdout.split("\t", 1)
        if parts and parts[0] not in ("Pending","InProgress",""):
            return parts[0], (parts[1] if len(parts) > 1 else "")
        time.sleep(5)
    return "Timeout", ""


def ssm_online(iid):
    r = subprocess.run(["aws","ssm","describe-instance-information","--region","us-east-1",
                        "--filters",f"Key=InstanceIds,Values={iid}","--query",
                        "InstanceInformationList[0].PingStatus","--output","text"],
                       capture_output=True, text=True, timeout=90)
    return r.stdout.strip() == "Online"


def public_ip(iid):
    r = subprocess.run(["aws","ec2","describe-instances","--region","us-east-1",
                        "--instance-ids",iid,"--query",
                        "Reservations[0].Instances[0].PublicIpAddress","--output","text"],
                       capture_output=True, text=True, timeout=90)
    ip = r.stdout.strip()
    return None if ip in ("None","") else ip


def health_200(ip, path="/health"):
    try:
        req = urllib.request.Request(f"http://{ip}:8000{path}")
        with urllib.request.urlopen(req, timeout=8) as fh:
            return fh.status == 200
    except Exception:
        return False


def first_completion(ip, model="google/gemma-4-E2B-it"):
    """Time one chat completion. 'health 200' and 'serves' differ per runtime."""
    body = json.dumps({"model": model, "messages":[{"role":"user","content":"Say ok."}],
                       "max_tokens": 8, "temperature": 0.0}).encode()
    req = urllib.request.Request(f"http://{ip}:8000/v1/chat/completions", data=body,
                                 headers={"Content-Type":"application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as fh:
        fh.read()
    return time.perf_counter() - t0


def launch(rig, name, ami=None, create_fn=None):
    """Returns (instance_id, t_launch, az, market) once run_instances returns an id.

    Falls back to on-demand after 3 fruitless rounds. Same hardware either way,
    so this cannot move a boot measurement -- it only stops a spot drought from
    adding hours. Market is recorded per rep so the choice stays auditable.
    """
    deadline = time.time() + 90*60
    rounds = 0
    while time.time() < deadline:
        rounds += 1
        spot = rounds <= 3
        market = "spot" if spot else "on-demand"
        if rounds == 4:
            log(f"  {rig}: 3 rounds without spot capacity -> falling back to on-demand")
        for az, subnet in SUBNETS:
            try:
                if ami:
                    r = ec2.run_instances(
                        ImageId=ami, InstanceType="g5g.2xlarge", MinCount=1, MaxCount=1,
                        SubnetId=subnet, SecurityGroupIds=[SG],
                        IamInstanceProfile={"Name": PROFILE},
                        BlockDeviceMappings=[{"DeviceName":"/dev/sda1","Ebs":{
                            "VolumeSize":80,"VolumeType":"gp3","DeleteOnTermination":True}}],
                        TagSpecifications=[{"ResourceType":"instance","Tags":[
                            {"Key":"Name","Value":name},{"Key":"ManagedBy","Value":rig}]}],
                        **({"InstanceMarketOptions":{"MarketType":"spot",
                             "SpotOptions":{"SpotInstanceType":"one-time"}}} if spot else {}))
                    t = time.perf_counter()
                    return r["Instances"][0]["InstanceId"], t, az, market
                out = asyncio.run(create_fn(subnet_id=subnet, security_group_id=SG,
                                            iam_instance_profile=PROFILE, name=name,
                                            instance_type="g5g.2xlarge", spot=spot))
                t = time.perf_counter()
                if out.startswith("✅"):
                    return out.split("`")[1], t, az, market
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in (
                        "InsufficientInstanceCapacity","MaxSpotInstanceCountExceeded"):
                    raise
            time.sleep(3)
        log(f"  {rig}: no capacity in any AZ, retrying")
        time.sleep(60)
    raise RuntimeError("no capacity within deadline")


def terminate(iid):
    try:
        ec2.terminate_instances(InstanceIds=[iid])
    except ClientError:
        pass
    for _ in range(60):
        try:
            s = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["State"]["Name"]
            if s == "terminated":
                return
        except Exception:
            return
        time.sleep(10)


def mask_apt(iid):
    ssm(iid, ["systemctl stop apt-daily-upgrade.timer apt-daily.timer unattended-upgrades || true",
              "systemctl mask apt-daily-upgrade.service apt-daily-upgrade.timer "
              "apt-daily.service apt-daily.timer || true"])


def wait_health(ip_getter, t0, budget=45*60):
    """Poll every 0.5 s. Returns seconds from t0 to the first 200.

    Resolution matters: at the original 5 s two genuinely similar boots landed on
    the same tick and reported byte-identical times, which reads as perfect
    reproducibility and is really just the measurement floor.
    """
    end = time.time() + budget
    ip = None
    while time.time() < end:
        if ip is None:
            ip = ip_getter()
        if ip and health_200(ip):
            return time.perf_counter() - t0, ip
        time.sleep(0.5)
        if ip is None:
            time.sleep(2)
    return None, None


def run_ours(rig_dir, rig_tag, svc, deploy_attr, rep, warm_leg):
    """Cold boot + optional warm redeploys for the JAX / PyTorch rigs."""
    sys.path.insert(0, f"/home/xbill/gemma4-dev/{rig_dir}")
    import importlib
    for m in list(sys.modules):
        if m == "server":
            del sys.modules[m]
    srv = importlib.import_module("server")
    deploy = getattr(srv, deploy_attr)

    name = f"{svc}-boot{rep}"
    iid, t0, az, market = launch(rig_tag, name, create_fn=srv.create_g5g_instance)
    log(f"  {rig_tag} rep{rep}: {iid} in {az}")
    rec = {"rig": rig_tag, "kind": "cold", "rep": rep, "instance": iid, "az": az, "market": market,
           "utc_launched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        masked = False
        while not masked and time.perf_counter() - t0 < 900:
            if ssm_online(iid):
                rec["t_ssm_online_s"] = round(time.perf_counter() - t0, 2)
                mask_apt(iid); masked = True
            else:
                time.sleep(3)

        while time.perf_counter() - t0 < 45*60:
            if "INSTALL COMPLETE" in asyncio.run(srv.get_install_progress(iid, tail=4)):
                rec["t_install_s"] = round(time.perf_counter() - t0, 2); break
            time.sleep(5)

        asyncio.run(deploy(iid, restart=True))
        rec["t_deploy_issued_s"] = round(time.perf_counter() - t0, 2)

        t_health, ip = wait_health(lambda: public_ip(iid), t0)
        rec["t_health_s"] = round(t_health, 2) if t_health else None
        rec["utc_healthy"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if ip:
            rec["t_first_completion_s"] = round(first_completion(ip), 2)
        log(f"  {rig_tag} rep{rep}: health {rec['t_health_s']}s "
            f"(install {rec.get('t_install_s')}s) first-completion {rec.get('t_first_completion_s')}s")
        results.append(rec); save()

        if warm_leg and ip:
            for w in range(1, REPEATS + 1):
                tw = time.perf_counter()
                asyncio.run(deploy(iid, restart=True))
                th, _ = wait_health(lambda: ip, tw, budget=20*60)
                fc = first_completion(ip) if th else None
                wrec = {"rig": rig_tag, "kind": "warm_redeploy", "rep": w, "instance": iid,
                        "t_health_s": round(th, 2) if th else None,
                        "t_first_completion_s": round(fc, 2) if fc else None}
                log(f"  {rig_tag} warm{w}: {wrec['t_health_s']}s")
                results.append(wrec); save()
    finally:
        terminate(iid)


def run_vllm(rep, warm_leg):
    AMI = "ami-0b44b90b3d02430ee"
    iid, t0, az, market = launch("gpu-vllm-g5g-2b", f"vllm-g5g-boot{rep}", ami=AMI)
    log(f"  vllm rep{rep}: {iid} in {az}")
    rec = {"rig": "gpu-vllm-g5g-2b", "kind": "cold", "rep": rep, "instance": iid, "az": az, "market": market,
           "utc_launched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        masked = False
        while not masked and time.perf_counter() - t0 < 900:
            if ssm_online(iid):
                rec["t_ssm_online_s"] = round(time.perf_counter() - t0, 2)
                mask_apt(iid); masked = True
            else:
                time.sleep(3)

        t_health, ip = wait_health(lambda: public_ip(iid), t0)
        rec["t_health_s"] = round(t_health, 2) if t_health else None
        rec["utc_healthy"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if ip:
            rec["t_first_completion_s"] = round(first_completion(ip), 2)
        # Pull the engine's own breakdown -- weight load and init are the two
        # terms that decide this rig's boot, and only its log reports them.
        cid = ssm(iid, ["grep -oE 'Loading weights took [0-9.]+ seconds' /var/log/vllm-serve.log | tail -1",
                        "grep -oE 'init engine .* took [0-9.]+ s' /var/log/vllm-serve.log | tail -1"])
        _, out = ssm_wait(iid, cid, timeout=180)
        rec["engine_log"] = out.strip()
        log(f"  vllm rep{rep}: health {rec['t_health_s']}s :: {rec['engine_log'][:90]}")
        results.append(rec); save()

        if warm_leg and ip:
            for w in range(1, REPEATS + 1):
                tw = time.perf_counter()
                cid = ssm(iid, ["systemctl restart vllm"])
                ssm_wait(iid, cid, timeout=300)
                time.sleep(5)
                th, _ = wait_health(lambda: ip, tw, budget=30*60)
                fc = first_completion(ip) if th else None
                wrec = {"rig": "gpu-vllm-g5g-2b", "kind": "warm_restart", "rep": w,
                        "instance": iid, "t_health_s": round(th, 2) if th else None,
                        "t_first_completion_s": round(fc, 2) if fc else None}
                log(f"  vllm warm{w}: {wrec['t_health_s']}s")
                results.append(wrec); save()
    finally:
        terminate(iid)


if __name__ == "__main__":
    # PyTorch and JAX first: each boot is minutes, so a failure surfaces early.
    for rep in range(1, REPEATS + 1):
        if already_done("gpu-pytorch-g5g-2b", "cold", rep):
            log(f"=== pytorch cold {rep}/{REPEATS} — already recorded, skipping ==="); continue
        log(f"=== pytorch cold {rep}/{REPEATS} ===")
        run_ours("gpu-pytorch-g5g-2b", "gpu-pytorch-g5g-2b", "torch-g5g",
                 "deploy_torch_server", rep, warm_leg=(rep == REPEATS))
    for rep in range(1, REPEATS + 1):
        if already_done("gpu-jax-g5g-2b", "cold", rep):
            log(f"=== jax cold {rep}/{REPEATS} — already recorded, skipping ==="); continue
        log(f"=== jax cold {rep}/{REPEATS} ===")
        run_ours("gpu-jax-g5g-2b", "gpu-jax-g5g-2b", "jax-g5g",
                 "deploy_jax_server", rep, warm_leg=(rep == REPEATS))
    for rep in range(1, REPEATS + 1):
        if already_done("gpu-vllm-g5g-2b", "cold", rep):
            log(f"=== vllm cold {rep}/{REPEATS} — already recorded, skipping ==="); continue
        log(f"=== vllm cold {rep}/{REPEATS} ===")
        run_vllm(rep, warm_leg=(rep == REPEATS))
    log("CAMPAIGN COMPLETE")
    save()
