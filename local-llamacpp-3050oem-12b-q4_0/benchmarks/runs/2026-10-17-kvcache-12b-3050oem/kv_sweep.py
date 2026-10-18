"""KV cache type x drafter on the 12B, RTX 3050 OEM: the most context each setting loads, and
generation speed with a long conversation already in context.

    python3 kv_sweep.py RIG_DIR OUT_DIR

Arms: cache f16 / q8_0 / q4_0 (K and V alike), each with the rig's exact Q4_0 drafter at 2 tokens
and without. f16 and q8_0 use the largest -c found in ../2026-10-16-first-light-3050oem/fit.md;
q4_0 has no fit yet, so its -c is the largest of FIT_Q4 that loads. One llama-server per arm on
port 8099, --parallel 1, the rig's serving flags otherwise.

Context: the first D tokens of the NVIDIA driver README (English technical prose, SHA-256 in
doc.sha256), at every D in DEPTHS that leaves 1,024 tokens free, followed by one of three
questions: a story and a code request unrelated to the document, and a summary of it. Greedy,
max_tokens 256, prompt cache on so each depth's document is read once per arm; decode_tps
(timings.predicted_per_second) excludes reading the prompt. Writes results.jsonl and fit.jsonl.
Method of local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-3050oem/longctx.py.
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
ARMS = [("f16-draft", "f16", True, [24576]), ("f16-none", "f16", False, [40960]),
        ("q8-draft", "q8_0", True, [49152]), ("q8-none", "q8_0", False, [65536]),
        ("q4-draft", "q4_0", True, FIT_Q4), ("q4-none", "q4_0", False, FIT_Q4)]


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def cool(limit=40):
    while int(gpu().split(",")[0]) > limit:
        time.sleep(5)


def start(ctx, ctk, draft, log_path):
    cmd = [env["LLAMA_SERVER_BIN"], "-m", env["MODEL_PATH"], "--host", "127.0.0.1", "--port", str(PORT),
           "-ngl", "99", "-c", str(ctx), "-ctk", ctk, "-ctv", ctk, "-fa", "1", "-t", "6", "-tb", "12",
           "--parallel", "1", "--reasoning", "off"]
    if draft:
        cmd += ["-md", DRAFT, "--spec-type", "draft-mtp", "-ngld", "all", "--spec-draft-n-max", "2"]
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


out = open(os.path.join(OUT, "results.jsonl"), "w")
fit = open(os.path.join(OUT, "fit.jsonl"), "w")
texts = {}
for arm, ctk, draft, ctxs in ARMS:
    srv = None
    for ctx in ctxs:
        srv, log = start(ctx, ctk, draft, os.path.join(OUT, f"server-{arm}-{ctx}.log"))
        fit.write(json.dumps({"arm": arm, "ctx": ctx, "loaded": srv is not None, "gpu": gpu()}) + "\n"); fit.flush()
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
    for depth in [d for d in DEPTHS if d in texts and d + 1024 <= ctx]:
        for q, question in QUESTIONS.items():
            if depth == 0 and q == "summary":
                continue
            content = f"Here is a document:\n\n{texts[depth]}\n\n{question}" if depth else question
            d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": content}],
                                              "max_tokens": 256, "temperature": 0, "cache_prompt": True})
            t = d["timings"]; text = d["choices"][0]["message"]["content"]
            rec = {"arm": arm, "ctk": ctk, "draft": draft, "ctx": ctx, "depth": depth, "question": q,
                   "prompt_n": d["usage"]["prompt_tokens"], "prompt_new": t["prompt_n"],
                   "prompt_tps": t["prompt_per_second"], "decode_tps": t["predicted_per_second"],
                   "n": t["predicted_n"], "draft_n": t.get("draft_n", 0),
                   "draft_accepted": t.get("draft_n_accepted", 0),
                   "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "gpu": gpu()}
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(arm, depth, q, rec["prompt_n"], round(rec["decode_tps"], 1), rec["n"],
                  rec["draft_accepted"], "/", rec["draft_n"], flush=True)
    stop(srv, log)
