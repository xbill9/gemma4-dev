"""Gemma 4 drafters on the Tesla T4 under vLLM: Google's QAT assistant (the rigs' SPECULATIVE_MODEL)
and the non-QAT `google/gemma-4-E?B-it-assistant`, at several draft lengths, against no drafter.
The vLLM counterpart of local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-1650ti/spec_ab.py:
same six prompts, greedy x3 plus one sampled pass (temperature 0.8, seed 1234), 256 tokens.

    python3 spec_ab.py RIG_DIR OUT_DIR ARM[,ARM...]

An arm is none[-tag] or DRAFTER-nN[-tag], DRAFTER one of qat (tpu.env SPECULATIVE_MODEL) or nq
(the same repo id without -qat-q4_0-unquantized). One vLLM server per arm on the rig's port with the
rig's serving flags from its tpu.env (as ~/bin/vllm-t4 passes them), started from the HF cache
offline. The card is cooled to <= COOL_C (default 50 C) before each arm. Decode tok/s per request
is (tokens - 1) / (last chunk - first chunk) of a streamed reply; acceptance is the difference of
the server's spec-decode counters across that one request (requests run one at a time).
Writes OUT_DIR/results.jsonl and OUT_DIR/server-ARM.log.
"""

import hashlib, json, os, signal, subprocess, sys, time, urllib.request

RIG, OUT, ARMLIST = sys.argv[1:4]
COOL_C = int(os.environ.get("COOL_C", "50"))
env = {}
for line in open(os.path.join(RIG, "tpu.env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k] = v.strip('"')
# A rig with no drafter in its tpu.env (the bf16 gpu-vllm-t4-2b) gets one from the environment.
if os.environ.get("SPECULATIVE_MODEL"):
    env["SPECULATIVE_MODEL"] = os.environ["SPECULATIVE_MODEL"]
PROMPTS = {
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "chat": "I just moved to a new city and feel a bit lonely. Any advice?",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "explain": "Explain how a refrigerator works to a ten-year-old.",
    "facts": "List the planets of the solar system with one fact about each.",
    "summary": "Summarize the causes of the French Revolution in one paragraph.",
}
QAT = env["SPECULATIVE_MODEL"]
assert QAT.endswith("-it-qat-q4_0-unquantized-assistant"), QAT
DRAFTS = {
    "qat": QAT,
    "nq": QAT.replace("-it-qat-q4_0-unquantized-assistant", "-it-assistant"),
    # the QAT drafter repacked to compressed-tensors W4A16 by repack/repack_drafter.py
    "q4ct": "/opt1/models/" + QAT.split("/")[1].replace("-unquantized-assistant", "-w4a16-ct-assistant"),
}
ARMS = []
for arm in ARMLIST.split(","):
    p = arm.split("-")
    ARMS.append((arm, None, None) if p[0] == "none" else (arm, DRAFTS[p[0]], int(p[1].lstrip("n"))))
REPEATS = 3
URL = env["ENDPOINT"]


def post_stream(body):
    body = dict(body, model=MODEL, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    times, text, n = [], [], 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("choices") and d["choices"][0]["delta"].get("content"):
                times.append(time.perf_counter())
                text.append(d["choices"][0]["delta"]["content"])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
    return "".join(text), n, (n - 1) / (times[-1] - times[0])


def spec_counters():
    out = {}
    with urllib.request.urlopen(URL + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:spec_decode_num_") and "_total" in line:
                name, val = line.rsplit(" ", 1)
                k = name.split("{")[0]
                out[k] = out.get(k, 0.0) + float(val)
    return (
        out.get("vllm:spec_decode_num_draft_tokens_total", 0.0),
        out.get("vllm:spec_decode_num_accepted_tokens_total", 0.0),
    )


def gpu():
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def cool():
    while int(gpu().split(",")[0]) > COOL_C:
        time.sleep(5)


MODEL = env["MODEL_NAME"]
senv = dict(os.environ, PYTHONUSERBASE=env["PYTHONUSERBASE"], TMPDIR=env["TMPDIR"], HF_HUB_OFFLINE="1")
os.makedirs(OUT, exist_ok=True)
out = open(os.path.join(OUT, "results.jsonl"), "a")
for arm, draft, n in ARMS:
    cmd = [
        env["PYTHON_BIN"],
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--host",
        env["HOST"],
        "--port",
        env["PORT"],
        "--dtype",
        env["DTYPE"],
        "--kv-cache-dtype",
        env["KV_CACHE_DTYPE"],
        "--tensor-parallel-size",
        env["TENSOR_PARALLEL_SIZE"],
        "--gpu-memory-utilization",
        env["GPU_MEMORY_UTILIZATION"],
        "--max-model-len",
        env["MAX_MODEL_LEN"],
        "--max-num-seqs",
        env["MAX_NUM_SEQS"],
    ]
    if env.get("LANGUAGE_MODEL_ONLY") == "true":
        cmd.append("--language-model-only")
    if n:
        cmd += ["--speculative-config", json.dumps({"model": draft, "num_speculative_tokens": n})]
    cool()
    log = open(os.path.join(OUT, f"server-{arm}.log"), "w")
    t0 = time.time()
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=senv, start_new_session=True)
    while True:
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            break
        except Exception:
            if srv.poll() is not None:
                sys.exit(f"{arm}: vLLM exited, see server-{arm}.log")
            if time.time() - t0 > 1800:
                os.killpg(srv.pid, signal.SIGTERM)
                sys.exit(f"{arm}: not healthy after 1800 s")
            time.sleep(5)
    print(arm, "healthy after", round(time.time() - t0), "s", flush=True)
    post_stream({"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16})  # warm-up
    vram = gpu()
    for mode in ("greedy", "sampled"):
        reps = REPEATS if mode == "greedy" else 1
        for name, prompt in PROMPTS.items():
            for r in range(reps):
                body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 256}
                body.update({"temperature": 0} if mode == "greedy" else {"temperature": 0.8, "seed": 1234})
                d0, a0 = spec_counters()
                text, ntok, tps = post_stream(body)
                d1, a1 = spec_counters()
                rec = {
                    "arm": arm,
                    "mode": mode,
                    "prompt": name,
                    "rep": r,
                    "decode_tps": tps,
                    "n": ntok,
                    "draft_n": d1 - d0,
                    "draft_accepted": a1 - a0,
                    "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
                    "text": text,
                    "gpu": vram,
                    "gpu_after": gpu(),
                    "startup_s": round(time.time() - t0),
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(
                    arm,
                    mode,
                    name,
                    r,
                    round(tps, 1),
                    ntok,
                    int(a1 - a0),
                    "/",
                    int(d1 - d0),
                    rec["gpu_after"],
                    flush=True,
                )
    os.killpg(srv.pid, signal.SIGTERM)
    srv.wait()
    log.close()
    time.sleep(10)
