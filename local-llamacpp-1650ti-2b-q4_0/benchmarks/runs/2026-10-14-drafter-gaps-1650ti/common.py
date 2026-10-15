"""Shared pieces for speed.py and longctx.py: tpu.env, drafter paths, server lifecycle.
Copied from local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-3050oem/common.py
for the GTX 1650 Ti: LLAMA_SERVER_BIN from the environment wins over tpu.env, -ot only when the rig
sets OVERRIDE_TENSORS (the 1650 Ti rigs keep per_layer_token_embd on the host), cooled to <=50 C."""
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
    if os.environ.get("LLAMA_SERVER_BIN"):
        env["LLAMA_SERVER_BIN"] = os.environ["LLAMA_SERVER_BIN"]
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

def cool(limit=50):
    while int(gpu().split(",")[0]) > limit:
        time.sleep(5)

def start(env, ctx, draft, n, log_path):
    cmd = [env["LLAMA_SERVER_BIN"], "-m", env["MODEL_PATH"], "--host", "127.0.0.1", "--port", str(PORT),
           "-ngl", "99", "-c", str(ctx), "-ctk", "f16", "-ctv", "f16", "-fa", "1", "-t", "6", "-tb", "12",
           "--parallel", "1", "--reasoning", "off"]
    if env.get("OVERRIDE_TENSORS"):
        cmd += ["-ot", env["OVERRIDE_TENSORS"]]
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
