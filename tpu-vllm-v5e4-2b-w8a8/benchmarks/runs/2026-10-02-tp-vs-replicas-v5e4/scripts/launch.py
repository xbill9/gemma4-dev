#!/usr/bin/env python3
"""On the VM: start a vLLM container with vllm-gemma4's exact image, env and args, changing only
--tensor-parallel-size, --port, TPU chip pinning and any extra args.

launch.py <name> <port> <tp> <visible_chips|all> <chips_per_process_bounds|-> <tpu_process_port|-> [CACHE] [ENV:K=V ...] [extra vllm args...]
"""
import json, subprocess, sys

name, port, tp, chips, bounds, pport, *rest = sys.argv[1:]
cache = "CACHE" in rest
extra = [a for a in rest if not a.startswith("ENV:") and a != "CACHE"]
xenv = [a[4:] for a in rest if a.startswith("ENV:")]
info = json.loads(subprocess.check_output(["sudo", "docker", "inspect", "vllm-gemma4"]))[0]
image = info["Config"]["Image"]
args = list(info["Args"])
i = args.index("--tensor-parallel-size")
args[i + 1] = tp
args += ["--port", port] + extra
env = [e for e in info["Config"]["Env"] if e.split("=", 1)[0] in ("HF_HOME", "MIN_TOKEN_BUCKET", "GMM_V2_TILE_VMEM_FRACTION")]
if chips != "all":
    env += [f"TPU_VISIBLE_CHIPS={chips}", f"TPU_CHIPS_PER_PROCESS_BOUNDS={bounds}", "TPU_PROCESS_BOUNDS=1,1,1",
            f"TPU_PROCESS_PORT={pport}", "CLOUD_TPU_TASK_ID=0", f"TPU_PROCESS_ADDRESSES=localhost:{pport}"]
tok = subprocess.check_output(["gcloud", "secrets", "versions", "access", "latest", "--secret=hf-token"], text=True).strip()
subprocess.run(["sudo", "docker", "rm", "-f", name], capture_output=True)
env += xenv
cmd = ["sudo", "docker", "run", "-d", "--name", name, "--privileged", "--net=host", "-v", "/dev/shm:/dev/shm", "--shm-size", "10gb"]
for e in env:
    cmd += ["-e", e]
if cache:  # persistent XLA compile cache shared by every container on this VM
    cmd += ["-e", "VLLM_XLA_CACHE_PATH=/xla-cache", "-v", "/opt/xla-cache:/xla-cache"]
cmd += ["-e", f"HF_TOKEN={tok}", image] + args
subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
print("started", name, "env", env, "args", " ".join(args))
