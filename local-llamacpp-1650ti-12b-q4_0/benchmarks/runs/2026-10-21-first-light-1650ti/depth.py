"""Generation and prompt reading with a long conversation held, at one serving setting, on the
GTX 1650 Ti. The 3050 12B sister's kv_sweep/tune method (same document, same three questions,
greedy, prompt cache on) cut to 128 tokens and fewer depths, because prompt reading here is
about 60 tok/s.

    python3 depth.py OUT_DIR NGL CTX KV DRAFT_N [DEPTH ...]     (DRAFT_N 0 = no drafter)

Document: NVIDIA_GLX-1.0/README.txt, SHA-256 c5161cdc... (the sister's doc.sha256), read from
~/models/longctx-doc. Appends to depth.jsonl; each record carries the load average.
"""
import hashlib, json, os, signal, subprocess, sys, time, urllib.request

OUT, NGL, CTX, KV, N = sys.argv[1:6]
DEPTHS = [int(d) for d in sys.argv[6:]] or [0, 4096, 16384]
PORT = 8099
URL = f"http://127.0.0.1:{PORT}"
BIN = "/home/xbill/llama.cpp/build/bin/llama-server"
MODEL = "/home/xbill/models/gemma-4-12B-it-qat-q4_0-exact/gemma-4-12B-it-q4_0-exact.gguf"
DRAFT = "/home/xbill/models/gemma-4-12B-it-qat-assistant/gemma-4-12B-it-qat-assistant-q4_0-exact.gguf"
DOC = open("/home/xbill/models/longctx-doc/NVIDIA_GLX-README.txt").read()
QUESTIONS = {
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "summary": "Summarize the document above in one paragraph.",
}
ARM = f"ngl{NGL}-c{CTX}-{KV}-" + (f"draft{N}" if N != "0" else "none")


def post(path, body, timeout=7200):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


cmd = [BIN, "-m", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "-ngl", NGL, "-c", CTX,
       "-ctk", KV, "-ctv", KV, "-fa", "1", "-t", "6", "-tb", "12", "--parallel", "1", "--reasoning", "off"]
if N != "0":
    cmd += ["-md", DRAFT, "--spec-type", "draft-mtp", "-ngld", "all", "--spec-draft-n-max", N]
log = open(os.path.join(OUT, f"server-depth-{ARM}.log"), "w")
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
for _ in range(300):
    if srv.poll() is not None:
        sys.exit(f"{ARM} failed to load")
    try:
        urllib.request.urlopen(URL + "/health", timeout=2); break
    except Exception:
        time.sleep(1)
post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16})
ids = post("/tokenize", {"content": DOC})["tokens"]
out = open(os.path.join(OUT, "depth.jsonl"), "a")
for depth in DEPTHS:
    text = post("/detokenize", {"tokens": ids[:depth]})["content"] if depth else ""
    for q, question in QUESTIONS.items():
        if depth == 0 and q == "summary":
            continue
        content = f"Here is a document:\n\n{text}\n\n{question}" if depth else question
        t0 = time.time()
        d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": content}],
                                          "max_tokens": 128, "temperature": 0, "cache_prompt": True})
        wall = time.time() - t0
        t = d["timings"]; reply = d["choices"][0]["message"]["content"]
        rec = {"arm": ARM, "ngl": int(NGL), "ctx": int(CTX), "kv": KV, "draft_n": int(N), "depth": depth,
               "question": q, "prompt_n": d["usage"]["prompt_tokens"], "prompt_new": t["prompt_n"],
               "prompt_tps": t["prompt_per_second"], "decode_tps": t["predicted_per_second"],
               "n": t["predicted_n"], "drafted": t.get("draft_n", 0), "accepted": t.get("draft_n_accepted", 0),
               "wall_s": round(wall, 1), "load": os.getloadavg()[0],
               "sha": hashlib.sha256(reply.encode()).hexdigest()[:16], "gpu": gpu()}
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(ARM, depth, q, rec["prompt_n"], round(rec["prompt_tps"], 1), round(rec["decode_tps"], 2),
              rec["n"], rec["accepted"], "/", rec["drafted"], rec["wall_s"], "s load", round(rec["load"], 2), flush=True)
srv.send_signal(signal.SIGTERM); srv.wait(); log.close()
