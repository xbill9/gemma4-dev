"""Start and stop one vLLM server on the T4 with a rig's serving flags (its tpu.env, as
~/bin/vllm-t4 passes them), optionally with a Gemma 4 drafter and extra flags. Shared by the
overnight jobs in this directory; nothing outside it imports this.

Drafters: qat = tpu.env SPECULATIVE_MODEL (Google's QAT assistant), nq = Google's non-QAT
assistant, q4ct = the QAT assistant repacked to compressed-tensors W4A16 by
repack/repack_drafter.py (in /opt1/models). An arm is none[-tag] or DRAFTER-nN[-tag].
"""
import json, os, signal, subprocess, sys, time, urllib.request

def load_env(rig):
    env = {}
    for line in open(os.path.join(rig, "tpu.env")):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip('"')
    return env

def drafts(env):
    qat = env["SPECULATIVE_MODEL"]
    return {"qat": qat, "nq": qat.replace("-it-qat-q4_0-unquantized-assistant", "-it-assistant"),
            "q4ct": "/opt1/models/" + qat.split("/")[1].replace("-unquantized-assistant", "-w4a16-ct-assistant")}

def parse_arm(env, arm):
    p = arm.split("-")
    return (None, None) if p[0] == "none" else (drafts(env)[p[0]], int(p[1].lstrip("n")))

def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

def cool(limit=int(os.environ.get("COOL_C", "50"))):
    while int(gpu().split(",")[0]) > limit:
        time.sleep(5)

def start(env, logpath, draft=None, n=None, extra=()):
    cmd = [env["PYTHON_BIN"], "-m", "vllm.entrypoints.openai.api_server", "--model", env["MODEL_NAME"],
           "--host", env["HOST"], "--port", env["PORT"], "--dtype", env["DTYPE"],
           "--kv-cache-dtype", env["KV_CACHE_DTYPE"], "--tensor-parallel-size", env["TENSOR_PARALLEL_SIZE"],
           "--gpu-memory-utilization", env["GPU_MEMORY_UTILIZATION"], "--max-model-len", env["MAX_MODEL_LEN"],
           "--max-num-seqs", env["MAX_NUM_SEQS"]]
    if env.get("LANGUAGE_MODEL_ONLY") == "true":
        cmd.append("--language-model-only")
    if n:
        cmd += ["--speculative-config", json.dumps({"model": draft, "num_speculative_tokens": n})]
    cmd += list(extra)
    senv = dict(os.environ, PYTHONUSERBASE=env["PYTHONUSERBASE"], TMPDIR=env["TMPDIR"], HF_HUB_OFFLINE="1")
    cool()
    log = open(logpath, "w")
    log.write(" ".join(cmd) + "\n"); log.flush()
    t0 = time.time()
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=senv, start_new_session=True)
    url = env["ENDPOINT"]
    while True:
        try:
            urllib.request.urlopen(url + "/health", timeout=2); break
        except Exception:
            if srv.poll() is not None:
                raise RuntimeError(f"vLLM exited during startup, see {logpath}")
            if time.time() - t0 > 1800:
                stop(srv, log); raise RuntimeError(f"not healthy after 1800 s, see {logpath}")
            time.sleep(5)
    print(os.path.basename(logpath), "healthy after", round(time.time() - t0), "s", flush=True)
    return srv, log

def stop(srv, log):
    try:
        os.killpg(srv.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    srv.wait(); log.close(); time.sleep(10)

def spec_counters(url):
    out = {}
    with urllib.request.urlopen(url + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:spec_decode_num_") and "_total" in line:
                name, val = line.rsplit(" ", 1)
                k = name.split("{")[0]
                out[k] = out.get(k, 0.0) + float(val)
    return (out.get("vllm:spec_decode_num_draft_tokens_total", 0.0),
            out.get("vllm:spec_decode_num_accepted_tokens_total", 0.0))

def stream_chat(url, model, body, timeout=900):
    """One streamed chat completion: (text, completion tokens, ttft s, decode tok/s, e2e s)."""
    body = dict(body, model=model, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); times, text, n = [], [], 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("choices") and d["choices"][0]["delta"].get("content"):
                times.append(time.perf_counter()); text.append(d["choices"][0]["delta"]["content"])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
    t1 = time.perf_counter()
    dec = (n - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else None
    return "".join(text), n, (times[0] - t0) if times else None, dec, t1 - t0
