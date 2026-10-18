"""Tuning the q4_0-cache serving candidates on the 12B, RTX 3050 OEM, after kv_sweep.py.

    python3 tune.py RIG_DIR OUT_DIR

1) The most context the q4_0 cache with the drafter loads, in 4,096 steps between kv_sweep's
81,920 (loads) and 98,304 (fails). 2) Prompt batch size (-ub, with -b at least -ub): 512 is
llama.cpp's default; 1024 and 2048 read long prompts faster if the compute buffer fits.
3) Drafted tokens 1 and 3 against 2 with a long conversation held. Same prompts, depths and
measurements as kv_sweep.py (prompt cache on, greedy, 256 tokens). Writes tune.jsonl, tune-fit.jsonl.
"""
import hashlib, json, os, signal, subprocess, sys, time, urllib.request

RIG, OUT = sys.argv[1], sys.argv[2]
PORT = 8099
URL = f"http://127.0.0.1:{PORT}"
env = {}
for line in open(os.path.join(RIG, "tpu.env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k] = v.strip('"')
DRAFT = env["SPEC_DRAFT_MODEL"]
DOC = open("/usr/share/doc/NVIDIA_GLX-1.0/README.txt").read()
DEPTHS = [0, 8192, 16384, 32768, 49152, 65536, 98304, 131072]
FIT_Q4 = [131072, 114688, 98304, 81920, 65536]
QUESTIONS = {
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "summary": "Summarize the document above in one paragraph.",
}
# (arm, cache, drafter, -c candidates largest first, drafted tokens, -ub, depths to measure)
ARMS = [("q4-draft-fit", "q4_0", True, [94208, 90112, 86016], "2", 512, [0, 65536]),
        ("q4-draft-ub1024", "q4_0", True, [81920, 65536], "2", 1024, [0, 32768, 65536]),
        ("q4-draft-ub2048", "q4_0", True, [81920, 65536], "2", 2048, [0, 32768, 65536]),
        ("q4-none-ub1024", "q4_0", False, [131072, 114688, 98304], "2", 1024, [32768, 98304]),
        ("q4-draft-n1", "q4_0", True, [81920], "1", 512, [0, 32768, 65536]),
        ("q4-draft-n3", "q4_0", True, [81920], "3", 512, [0, 32768, 65536])]


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def cool(limit=40):
    while int(gpu().split(",")[0]) > limit:
        time.sleep(5)


def start(ctx, ctk, draft, n, ub, log_path):
    cmd = [env["LLAMA_SERVER_BIN"], "-m", env["MODEL_PATH"], "--host", "127.0.0.1", "--port", str(PORT),
           "-ngl", "99", "-c", str(ctx), "-ctk", ctk, "-ctv", ctk, "-fa", "1", "-t", "6", "-tb", "12",
           "--parallel", "1", "--reasoning", "off", "-b", str(max(2048, ub)), "-ub", str(ub)]
    if draft:
        cmd += ["-md", DRAFT, "--spec-type", "draft-mtp", "-ngld", "all", "--spec-draft-n-max", n]
    cool()
    log = open(log_path, "w")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        if srv.poll() is not None:
            log.close()
            return None, None
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16})
            return srv, log
        except Exception:
            time.sleep(1)
    srv.kill(); srv.wait(); log.close()
    return None, None


def stop(srv, log):
    srv.send_signal(signal.SIGTERM); srv.wait(); log.close()


out = open(os.path.join(OUT, "tune.jsonl"), "w")
fit = open(os.path.join(OUT, "tune-fit.jsonl"), "w")
texts = {}
for arm, ctk, draft, ctxs, n, ub, want in ARMS:
    srv = None
    for ctx in ctxs:
        srv, log = start(ctx, ctk, draft, n, ub, os.path.join(OUT, f"server-{arm}-{ctx}.log"))
        fit.write(json.dumps({"arm": arm, "ub": ub, "ctx": ctx, "loaded": srv is not None, "gpu": gpu()}) + "\n"); fit.flush()
        print("fit", arm, ctx, srv is not None, flush=True)
        if srv:
            break
    if not srv:
        continue
    if not texts:
        ids = post("/tokenize", {"content": DOC})["tokens"]
        print("doc tokens", len(ids), flush=True)
        for depth in DEPTHS:
            if depth <= len(ids):
                texts[depth] = post("/detokenize", {"tokens": ids[:depth]})["content"] if depth else ""
    for depth in [d for d in want if d in texts and d + 1024 <= ctx]:
        for q, question in QUESTIONS.items():
            if depth == 0 and q == "summary":
                continue
            content = f"Here is a document:\n\n{texts[depth]}\n\n{question}" if depth else question
            d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": content}],
                                              "max_tokens": 256, "temperature": 0, "cache_prompt": True})
            t = d["timings"]; text = d["choices"][0]["message"]["content"]
            rec = {"arm": arm, "ctk": ctk, "draft": draft, "n_max": n, "ub": ub, "ctx": ctx, "depth": depth, "question": q,
                   "prompt_n": d["usage"]["prompt_tokens"], "prompt_new": t["prompt_n"],
                   "prompt_tps": t["prompt_per_second"], "decode_tps": t["predicted_per_second"],
                   "n": t["predicted_n"], "draft_n": t.get("draft_n", 0),
                   "draft_accepted": t.get("draft_n_accepted", 0),
                   "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "gpu": gpu()}
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(arm, depth, q, rec["prompt_n"], round(rec["decode_tps"], 1), rec["n"],
                  rec["draft_accepted"], "/", rec["draft_n"], flush=True)
    stop(srv, log)
