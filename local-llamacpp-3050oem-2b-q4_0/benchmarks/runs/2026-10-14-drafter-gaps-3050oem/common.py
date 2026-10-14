"""Shared pieces for speed.py and longctx.py: tpu.env, drafter paths, server lifecycle."""
import json, os, signal, subprocess, time, urllib.request

PORT = 8099
URL = f"http://127.0.0.1:{PORT}"

def load_env(rig):
    env = {}
    for line in open(os.path.join(rig, "tpu.env")):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip('"')
    return env

def drafts(env):
    q4x = env["SPEC_DRAFT_MODEL"]
    assert q4x.endswith("-it-qat-assistant-q4_0-exact.gguf"), q4x
    # Google's non-QAT drafter (google/gemma-4-E?B-it-assistant), Q8_0 by convert_hf_to_gguf.py
    nq = q4x.replace("-it-qat-assistant", "-it-assistant").replace("-q4_0-exact.gguf", "-q8_0.gguf")
    assert os.path.exists(nq), nq
    return {"q4x": q4x, "nq": nq}

def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

def cool(limit=40):
    while int(gpu().split(",")[0]) > limit:
        time.sleep(5)

def start(env, ctx, draft, n, log_path):
    cmd = [env["LLAMA_SERVER_BIN"], "-m", env["MODEL_PATH"], "--host", "127.0.0.1", "--port", str(PORT),
           "-ngl", "99", "-c", str(ctx), "-ctk", "f16", "-ctv", "f16", "-fa", "1", "-t", "6", "-tb", "12",
           "--parallel", "1", "--reasoning", "off", "-ot", env["OVERRIDE_TENSORS"]]
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
