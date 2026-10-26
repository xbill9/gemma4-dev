"""Shared pieces for speed.py and longctx.py on the L4: tpu.env, server lifecycle.

From local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-3050oem/common.py with the
thread counts read from tpu.env and a bounded cool-down (a datacenter L4 idles warmer than a desktop card).
"""
import json, os, signal, subprocess, time, urllib.request

PORT = 8099
URL = f"http://127.0.0.1:{PORT}"

def load_env(path):
    env = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip('"')
    return env

def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

def cool(limit=45, wait=90):
    end = time.time() + wait
    while int(gpu().split(",")[0]) > limit and time.time() < end:
        time.sleep(5)

def start(env, ctx, draft, n, log_path, parallel=1):
    cmd = [env["LLAMA_SERVER_BIN"], "-m", env["MODEL_PATH"], "--host", "127.0.0.1", "--port", str(PORT),
           "-ngl", "99", "-c", str(ctx), "-ctk", "f16", "-ctv", "f16", "-fa", "1",
           "-t", env["THREADS"], "-tb", env["THREADS_BATCH"], "--parallel", str(parallel),
           "--reasoning", "off", "-ot", env["OVERRIDE_TENSORS"]]
    if draft:
        cmd += ["-md", draft, "--spec-type", "draft-mtp", "-ngld", "all", "--spec-draft-n-max", n]
    cool()
    log = open(log_path, "w")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        if srv.poll() is not None:
            raise RuntimeError(f"server exited, see {log_path}")
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16})
    return srv, log

def stop(srv, log):
    srv.send_signal(signal.SIGTERM); srv.wait(); log.close()
