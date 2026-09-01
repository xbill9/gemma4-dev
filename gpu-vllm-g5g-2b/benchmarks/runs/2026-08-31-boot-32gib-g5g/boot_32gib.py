#!/usr/bin/env python3
"""One vLLM cold boot on g5g.4xlarge (32 GiB) to test the RAM-pressure hypothesis.

All three timed boots on 2026-08-31 were g5g.2xlarge (16 GiB), where weight
loading took 546 s against 11.19 GiB of available RAM. If that is page-cache
thrash then a 32 GiB host should collapse it. Same AMI, same AZ pool, same
harness and the same two stop lines as the 9-boot campaign.

n=1 is adequate ONLY because the predicted effect (546 s -> tens of seconds)
dwarfs the ~12% boot variance the campaign measured. It would not be adequate
for a small effect.
"""
import sys, time, json
sys.path.insert(0, "/tmp/claude-1000/-home-xbill-gemma4-dev-gpu-pytorch-g5g-2b/dc2377d3-309f-45f7-adc1-d87ec3922a6a/scratchpad")
from boot_campaign import (  # noqa: E402
    ec2, SUBNETS, SG, PROFILE, log, ssm, ssm_wait, ssm_online, public_ip,
    wait_health, first_completion, mask_apt, terminate)
from botocore.exceptions import ClientError  # noqa: E402

AMI = "ami-0b44b90b3d02430ee"
ITYPE = "g5g.4xlarge"
OUT = ("/tmp/claude-1000/-home-xbill-gemma4-dev-gpu-pytorch-g5g-2b/"
       "dc2377d3-309f-45f7-adc1-d87ec3922a6a/scratchpad/boot_32gib_result.json")


def launch():
    deadline, rounds = time.time() + 60*60, 0
    while time.time() < deadline:
        rounds += 1
        spot = rounds <= 3
        market = "spot" if spot else "on-demand"
        if rounds == 4:
            log(f"  32GiB: 3 rounds without spot -> on-demand")
        for az, subnet in SUBNETS:
            try:
                r = ec2.run_instances(
                    ImageId=AMI, InstanceType=ITYPE, MinCount=1, MaxCount=1,
                    SubnetId=subnet, SecurityGroupIds=[SG],
                    IamInstanceProfile={"Name": PROFILE},
                    BlockDeviceMappings=[{"DeviceName":"/dev/sda1","Ebs":{
                        "VolumeSize":80,"VolumeType":"gp3","DeleteOnTermination":True}}],
                    TagSpecifications=[{"ResourceType":"instance","Tags":[
                        {"Key":"Name","Value":"vllm-g5g-boot32"},
                        {"Key":"ManagedBy","Value":"gpu-vllm-g5g-2b"}]}],
                    **({"InstanceMarketOptions":{"MarketType":"spot",
                        "SpotOptions":{"SpotInstanceType":"one-time"}}} if spot else {}))
                return r["Instances"][0]["InstanceId"], time.perf_counter(), az, market
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in (
                        "InsufficientInstanceCapacity","MaxSpotInstanceCountExceeded"):
                    raise
        log("  32GiB: no capacity in any AZ, retrying")
        time.sleep(60)
    raise RuntimeError("no capacity")


iid, t0, az, market = launch()
log(f"32GiB boot: {iid} ({ITYPE}, {market}) in {az}")
rec = {"rig": "gpu-vllm-g5g-2b", "kind": "cold", "instance_type": ITYPE, "host_ram_gb": 32,
       "instance": iid, "az": az, "market": market,
       "utc_launched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
try:
    while time.perf_counter() - t0 < 900:
        if ssm_online(iid):
            rec["t_ssm_online_s"] = round(time.perf_counter() - t0, 2); mask_apt(iid); break
        time.sleep(3)
    th, ip = wait_health(lambda: public_ip(iid), t0)
    rec["t_health_s"] = round(th, 2) if th else None
    if ip:
        rec["t_first_completion_s"] = round(first_completion(ip), 2)
    cid = ssm(iid, [
        "grep -oE 'Loading weights took [0-9.]+ seconds' /var/log/vllm-serve.log | tail -1",
        "grep -oE 'init engine .* took [0-9.]+ s' /var/log/vllm-serve.log | tail -1",
        "grep -oE 'Available RAM: [0-9.]+ GiB' /var/log/vllm-serve.log | tail -1",
        "grep -oE 'GPU KV cache size: [0-9,]+ tokens' /var/log/vllm-serve.log | tail -1",
        "free -g | awk 'NR==2{print \"MemTotal_GiB\", $2}'"])
    _, out = ssm_wait(iid, cid, timeout=240)
    rec["engine_log"] = out.strip()
    log(f"32GiB: health {rec['t_health_s']}s\n{rec['engine_log']}")
    json.dump(rec, open(OUT, "w"), indent=2)
finally:
    terminate(iid)
    log("32GiB: terminated")
