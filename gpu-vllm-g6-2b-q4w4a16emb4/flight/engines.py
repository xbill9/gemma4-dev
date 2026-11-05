"""Every engine on one NVIDIA L4 (EC2 G6) in one sitting, the same requests, measured one way.
Forked 2026-11-05 from local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-engines-3050oem/engines.py;
the requests, statistics and arm names are that file's, so the two flights' rows line up.

    python3 engines.py OUT_DIR ARM[,ARM...]

Run on the instance, with the monorepo at GEMMA4_DEV (default /opt/gemma4-dev). Arms (one server each,
started and stopped here, card cooled to <= COOL_C before each):
    vllm, vllm-d4       gpu-vllm-g6-2b-q4w4a16emb4 serve.py; -d4 = W4A16 drafter (its DRAFTER_DIR), 4 tokens
    llama, llama-d2     gpu-llamacpp-g6-2b-q4_0 tpu.env binary and model at the 3050 flight's flags
                        (-c 8192 --parallel 4), its THREADS/THREADS_BATCH; -d2 = exact Q4_0 drafter, 2 tokens
    mistralrs, mistralrs-dN
                        gpu-mistralrs-g6-2b-q4_0 serve.py (the 3050's 4 requests, 8,192-token cache);
                        -dN = Google's QAT assistant (bf16), N drafted tokens, 4,096-token cache
    mistralrs-card      the same at 64 requests, 65,536-token cache, 4,096 tokens per step (UNTESTED)
    sglang, sglang-dN   gpu-sglang-g6-2b-q4w4a16emb4 serve.py; -dN = W4A16 drafter, N tokens
    pytorch             gpu-pytorch-g6-2b-q4w4a16 openai_server.py on the DLAMI interpreter (greedy only,
                        no streaming)
    litert-q4, litert-int248, litert-int248-d
                        gpu-litertlm-g6-2b-q4w4a16emb4 and gpu-litertlm-g6-2b-int248; -d = built-in drafter on

Requests: the six prompts of the 3050 flight, greedy, thinking off, 256 tokens (max_completion_tokens as
well, which LiteRT needs), prompt cache off on llama.cpp.
 - e2e: non-streaming, 3 repeats per prompt; tok/s = completion_tokens / request wall time. Every engine.
 - stream: 3 repeats per prompt; decode tok/s = (completion_tokens - 1) / (last - first token chunk), TTFT.
   Every engine but pytorch.
 - batch: 4 requests at once (one of each of the first four prompts), non-streaming, 3 repeats.
Per arm: GPU memory after the warm-up. Appends to OUT_DIR/results.jsonl; server log per arm."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

OUT, ARMLIST = sys.argv[1:3]
COOL_C = int(os.environ.get("COOL_C", "50"))
ROOT = os.environ.get("GEMMA4_DEV", "/opt/gemma4-dev")
VRIG = os.path.join(ROOT, "gpu-vllm-g6-2b-q4w4a16emb4")
LRIG = os.path.join(ROOT, "gpu-llamacpp-g6-2b-q4_0")
PRIG = os.path.join(ROOT, "gpu-pytorch-g6-2b-q4w4a16")
MRIG = os.path.join(ROOT, "gpu-mistralrs-g6-2b-q4_0")
SRIG = os.path.join(ROOT, "gpu-sglang-g6-2b-q4w4a16emb4")
LIRIG = os.path.join(ROOT, "gpu-litertlm-g6-2b-int248")
LQRIG = os.path.join(ROOT, "gpu-litertlm-g6-2b-q4w4a16emb4")
SYS_PY = "/usr/bin/python3"
PROMPTS = {
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "chat": "I just moved to a new city and feel a bit lonely. Any advice?",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "explain": "Explain how a refrigerator works to a ten-year-old.",
    "facts": "List the planets of the solar system with one fact about each.",
    "summary": "Summarize the causes of the French Revolution in one paragraph.",
}
REPEATS = 3


def load_env(path):
    vals = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"')
    return vals


def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def mtp_dir():
    """Google's QAT assistant snapshot at the mistral.rs rig's pinned revision (downloads once)."""
    from huggingface_hub import snapshot_download
    m = load_env(os.path.join(MRIG, "tpu.env"))
    return snapshot_download(m["MTP_HF"], revision=m["MTP_HF_REVISION"])


def server(arm):
    """(cmd, env, url, model, health path, streams)"""
    env = dict(os.environ)
    if arm.startswith("vllm"):
        v = load_env(os.path.join(VRIG, "tpu.env"))
        env["SPECULATIVE_MODEL"] = ""
        if arm == "vllm-d4":
            env.update(SPECULATIVE_MODEL=v["DRAFTER_DIR"], NUM_SPECULATIVE_TOKENS="4")
        return [SYS_PY, os.path.join(VRIG, "serve.py")], env, v["ENDPOINT"], v["MODEL_NAME"], "/health", True
    if arm.startswith("llama"):
        v = load_env(os.path.join(LRIG, "tpu.env"))
        cmd = [v["LLAMA_SERVER_BIN"], "-m", v["MODEL_PATH"], "--host", "127.0.0.1", "--port", "8099", "-ngl", "99",
               "-c", "8192", "--parallel", "4", "-ctk", "f16", "-ctv", "f16", "-fa", "1", "-t", v["THREADS"], "-tb", v["THREADS_BATCH"],
               "--reasoning", "off", "-ot", v["OVERRIDE_TENSORS"], "--metrics"]
        if arm == "llama-d2":
            cmd += ["-md", v["SPEC_DRAFT_MODEL"], "--spec-type", "draft-mtp", "-ngld", "all", "--spec-draft-n-max", "2"]
        return cmd, env, "http://127.0.0.1:8099", v["MODEL_NAME"], "/health", True
    if arm.startswith("mistralrs"):
        v = load_env(os.path.join(MRIG, "tpu.env"))
        env["MTP_MODEL"] = ""
        if arm.startswith("mistralrs-d"):
            # the drafter fits beside 4 requests with a 4,096-token cache, not 8,192
            env.update(MTP_MODEL=mtp_dir(), MTP_N_PREDICT=arm[len("mistralrs-d"):], KV_CACHE="4096tokens",
                       MAX_MODEL_LEN="4096")
        if arm == "mistralrs-card":  # the whole L4, not the 3050's 8 GB limits (UNTESTED)
            env.update(MAX_SEQS="64", KV_CACHE="65536tokens", MAX_NUM_BATCHED_TOKENS="4096")
        return [SYS_PY, os.path.join(MRIG, "serve.py")], env, v["ENDPOINT"], "default", "/health", True
    if arm.startswith("sglang"):
        v = load_env(os.path.join(SRIG, "tpu.env"))
        env["SPECULATIVE_MODEL"] = ""
        if arm.startswith("sglang-d"):  # W4A16 drafter (vLLM's best), N drafted tokens
            drafter = load_env(os.path.join(VRIG, "tpu.env"))["DRAFTER_DIR"]
            env.update(SPECULATIVE_MODEL=drafter, SPECULATIVE_NUM_STEPS=arm[len("sglang-d"):])
        return [SYS_PY, os.path.join(SRIG, "serve.py")], env, v["ENDPOINT"], v["MODEL_NAME"], "/health", True
    if arm == "pytorch":
        py = open("/opt/torch-g6-q4/PYTHON_BIN").read().strip()  # the DLAMI's torch 2.13 interpreter
        return ([py, os.path.join(PRIG, "openai_server.py"), "--port", "8100", "--max-ctx", "8192"], env,
                "http://127.0.0.1:8100", "google/gemma-4-E2B-it", "/health", False)
    if arm.startswith("litert"):
        rig = {"litert-q4": LQRIG, "litert-int248": LIRIG, "litert-int248-d": LIRIG}[arm]
        r = load_env(os.path.join(rig, "tpu.env"))
        mid = r["MODEL_ID"]
        cfg = os.path.join(OUT, f"config-{arm}.json")
        spec = "on" if arm.endswith("-d") else "off"
        json.dump({"models": {mid: {"backend": "gpu", "cache": "disk", "speculative_decoding": spec}}}, open(cfg, "w"))
        env["LITERT_LM_DIR"] = r["LITERT_LM_DIR"]
        return (["litert-lm", "serve", "--config", cfg, "--host", "127.0.0.1", "--port", "9399"], env,
                "http://127.0.0.1:9399", mid, "/v1/models", True)
    raise SystemExit(f"unknown arm {arm}")


def body(model, prompt, stream):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 256,
         "max_completion_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
         "cache_prompt": False}
    if model == "default":  # mistral.rs: max_completion_tokens is an alias of max_tokens and both is a 400
        del b["max_completion_tokens"]
    if stream:
        b.update(stream=True, stream_options={"include_usage": True})
    return json.dumps(b).encode()


def post(url, data):
    return urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", data,
                                                         {"Content-Type": "application/json"}), timeout=900)


def e2e(url, model, prompt):
    t0 = time.perf_counter()
    with post(url, body(model, prompt, False)) as r:
        d = json.load(r)
    w = time.perf_counter() - t0
    n = d["usage"]["completion_tokens"]
    text = d["choices"][0]["message"].get("content") or ""
    return n, w, text


def stream(url, model, prompt):
    stamps, text, n = [], [], 0
    t0 = time.perf_counter()
    with post(url, body(model, prompt, True)) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
                if piece:
                    stamps.append(time.perf_counter())
                    text.append(piece)
            if d.get("usage"):
                n = d["usage"].get("completion_tokens", n)
    tps = (n - 1) / (stamps[-1] - stamps[0]) if n > 1 and len(stamps) > 1 else 0.0
    return n, tps, (stamps[0] - t0) if stamps else 0.0, "".join(text)


os.makedirs(OUT, exist_ok=True)
out = open(os.path.join(OUT, "results.jsonl"), "a")


def emit(rec):
    out.write(json.dumps(rec) + "\n")
    out.flush()
    print(json.dumps(rec), flush=True)


for arm in ARMLIST.split(","):
    cmd, env, url, model, health, streams = server(arm)
    while int(gpu().split(",")[0]) > COOL_C:
        time.sleep(5)
    log = open(os.path.join(OUT, f"server-{arm}.log"), "w")
    t_start = time.time()
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    while srv.poll() is None:
        try:
            urllib.request.urlopen(url + health, timeout=2)
            break
        except Exception:
            time.sleep(2)
    if srv.poll() is not None:
        print(f"{arm}: server exited, see server-{arm}.log", flush=True)
        continue
    # mistral.rs answers /health before it admits requests (503 service_unavailable): ready = first reply
    for _ in range(300):
        try:
            e2e(url, model, "Hello")  # warm-up
            break
        except Exception:
            time.sleep(2)
    ready_s = round(time.time() - t_start, 1)
    if streams:
        stream(url, model, "Hello")
    mem = gpu()
    base = {"arm": arm, "gpu": mem, "ready_s": ready_s}
    for name, prompt in PROMPTS.items():
        for rep in range(REPEATS):
            n, w, text = e2e(url, model, prompt)
            emit(dict(base, kind="e2e", prompt=name, rep=rep, tokens=n, wall_s=w, tps=n / w,
                      sha=hashlib.sha256(text.encode()).hexdigest()[:16]))
    if streams:
        for name, prompt in PROMPTS.items():
            for rep in range(REPEATS):
                n, tps, ttft, text = stream(url, model, prompt)
                emit(dict(base, kind="stream", prompt=name, rep=rep, tokens=n, tps=tps, ttft_s=ttft,
                          sha=hashlib.sha256(text.encode()).hexdigest()[:16]))
    four = list(PROMPTS.values())[:4]
    for rep in range(REPEATS):
        res = [None] * 4

        def go(i):
            res[i] = e2e(url, model, four[i])

        ts = [threading.Thread(target=go, args=(i,)) for i in range(4)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        w = time.perf_counter() - t0
        tok = sum(r[0] for r in res if r)
        emit(dict(base, kind="batch4", rep=rep, tokens=tok, wall_s=w, tps=tok / w, ok=sum(r is not None for r in res)))
    os.killpg(srv.pid, signal.SIGTERM)
    srv.wait()
    log.close()
    while int(gpu().split(",")[2].split()[0]) > 100:
        time.sleep(2)
