#!/usr/bin/env python3
"""Within-box A/B of vLLM's --safetensors-load-strategy=prefetch.

The 2026-08-31 boots showed weight loading at ~468-546 s = ~21 MB/s off local
disk, which is neither RAM (a 32 GiB host did not fix it) nor gp3 bandwidth.
vLLM names the suspect in every boot log: auto-prefetch is disabled on EXT4.

Runs ON-DEMAND, not spot. This needs ~40 min of instance life (23 min boot plus
three timed restarts) and the first attempt was reclaimed 10 minutes in with
Server.SpotInstanceTermination. A reclamation mid-A/B would corrupt the
comparison rather than merely waste the run. ~$0.42 for the whole thing.

One instance, restarts only, so the comparison has no host variance in it:
  restart A  -- as shipped        (expect ~264 s, the campaign's warm median)
  restart B  -- with prefetch
  restart C  -- with prefetch, repeat
The flag may not exist in v0.27.2rc0; an argparse failure is a real outcome and
is reported rather than retried.
"""
import sys, time, json
sys.path.insert(0, "/tmp/claude-1000/-home-xbill-gemma4-dev-gpu-pytorch-g5g-2b/dc2377d3-309f-45f7-adc1-d87ec3922a6a/scratchpad")
from boot_campaign import (  # noqa: E402
    ec2, SUBNETS, SG, PROFILE, log, ssm, ssm_wait, ssm_online, public_ip,
    wait_health, first_completion, mask_apt, terminate)
from botocore.exceptions import ClientError  # noqa: E402

AMI, ITYPE = "ami-0b44b90b3d02430ee", "g5g.2xlarge"
OUT = ("/tmp/claude-1000/-home-xbill-gemma4-dev-gpu-pytorch-g5g-2b/"
       "dc2377d3-309f-45f7-adc1-d87ec3922a6a/scratchpad/prefetch_ab_result.json")
res = {"instance_type": ITYPE, "ami": AMI, "restarts": []}


def launch():
    deadline, rounds = time.time() + 60*60, 0
    while time.time() < deadline:
        rounds += 1
        spot = False  # on-demand: see the docstring
        for az, subnet in SUBNETS:
            try:
                r = ec2.run_instances(
                    ImageId=AMI, InstanceType=ITYPE, MinCount=1, MaxCount=1,
                    SubnetId=subnet, SecurityGroupIds=[SG],
                    IamInstanceProfile={"Name": PROFILE},
                    BlockDeviceMappings=[{"DeviceName":"/dev/sda1","Ebs":{
                        "VolumeSize":80,"VolumeType":"gp3","DeleteOnTermination":True}}],
                    TagSpecifications=[{"ResourceType":"instance","Tags":[
                        {"Key":"Name","Value":"vllm-g5g-prefetch"},
                        {"Key":"ManagedBy","Value":"gpu-vllm-g5g-2b"}]}],
                    **({"InstanceMarketOptions":{"MarketType":"spot",
                        "SpotOptions":{"SpotInstanceType":"one-time"}}} if spot else {}))
                return r["Instances"][0]["InstanceId"], az, ("spot" if spot else "on-demand")
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in (
                        "InsufficientInstanceCapacity","MaxSpotInstanceCountExceeded"):
                    raise
        log("  prefetch-ab: no capacity, retrying"); time.sleep(60)
    raise RuntimeError("no capacity")


def timed_restart(iid, ip, label):
    t = time.perf_counter()
    cid = ssm(iid, ["systemctl restart vllm"]); ssm_wait(iid, cid, timeout=300)
    time.sleep(5)
    th, _ = wait_health(lambda: ip, t, budget=30*60)
    cid = ssm(iid, [
        "grep -oE 'Loading weights took [0-9.]+ seconds' /var/log/vllm-serve.log | tail -1",
        "grep -oE 'error: unrecognized arguments.*|error: argument .*' /var/log/vllm-serve.log | tail -1",
        "systemctl is-active vllm"])
    _, out = ssm_wait(iid, cid, timeout=240)
    rec = {"label": label, "t_health_s": round(th, 2) if th else None, "log": out.strip()}
    if th:
        rec["t_first_completion_s"] = round(first_completion(ip), 2)
    log(f"  {label}: health={rec['t_health_s']}s :: {rec['log'][:110]}")
    res["restarts"].append(rec); json.dump(res, open(OUT, "w"), indent=2)
    return rec


iid, az, market = launch()
res.update(instance=iid, az=az, market=market)
log(f"prefetch-ab: {iid} ({market}) in {az}")
try:
    while not ssm_online(iid):
        time.sleep(3)
    mask_apt(iid)
    t0 = time.perf_counter()
    th, ip = wait_health(lambda: public_ip(iid), t0)
    res["cold_boot_s"] = round(th, 2) if th else None
    log(f"  cold boot: {res['cold_boot_s']}s -- now the A/B")

    timed_restart(iid, ip, "A: as shipped")

    cid = ssm(iid, [
        "cp /opt/serve.sh /opt/serve.sh.orig",
        "sed -i 's|--host 0.0.0.0|--safetensors-load-strategy=prefetch \\\\\\n  --host 0.0.0.0|' /opt/serve.sh",
        "grep -n 'safetensors-load-strategy' /opt/serve.sh || echo SED-FAILED"])
    _, out = ssm_wait(iid, cid, timeout=180)
    res["sed"] = out.strip()
    log(f"  patched serve.sh :: {res['sed'][:120]}")

    timed_restart(iid, ip, "B: prefetch")
    timed_restart(iid, ip, "C: prefetch repeat")
    json.dump(res, open(OUT, "w"), indent=2)
finally:
    terminate(iid); log("prefetch-ab: terminated")
