#!/usr/bin/env python3
"""relaunch.py <max_model_len>: re-create the vllm-gemma4 container with the same image, env and
args as the boot script's, changing only --max-model-len. Prints the args (env values masked)."""
import json, subprocess, sys
L = sys.argv[1]
c = json.loads(subprocess.check_output(["sudo", "docker", "inspect", "vllm-gemma4"]))[0]
# Persist the original spec once, so later relaunches start from the boot script's container.
import os
spec = "/tmp/vllm-gemma4.spec.json"
if os.path.exists(spec):
    c = json.load(open(spec))
else:
    json.dump(c, open(spec, "w"))
img = c["Config"]["Image"]
cmd = list(c["Config"]["Cmd"] or [])
i = cmd.index("--max-model-len")
cmd[i + 1] = L
base_env = set(json.loads(subprocess.check_output(["sudo", "docker", "inspect", img]))[0]["Config"]["Env"] or [])
env = [e for e in c["Config"]["Env"] if e not in base_env]
subprocess.run(["sudo", "docker", "rm", "-f", "vllm-gemma4"], check=False, capture_output=True)
args = ["sudo", "docker", "run", "--name", "vllm-gemma4", "--privileged", "--net=host", "-d",
        "-v", "/dev/shm:/dev/shm", "--shm-size", "10gb",
        "-e", "VLLM_XLA_CACHE_PATH=/xla-cache", "-v", "/opt/xla-cache:/xla-cache"]
subprocess.run(["sudo", "mkdir", "-p", "/opt/xla-cache"], check=True)
for e in env:
    args += ["-e", e]
args += [img] + cmd
print(" ".join(a if not a.startswith("HF_TOKEN=") else "HF_TOKEN=<masked>" for a in args))
subprocess.run(args, check=True, capture_output=True)
