"""Drafter A/B for the 12B on the GTX 1650 Ti Max-Q (4 GiB), where most layers run on the CPU.
From local-llamacpp-3050oem-31b-q4_0/benchmarks/runs/2026-10-18-first-light-3050oem/spec_ab.py
(the 3050's partly-offloaded 31B), with this rig's model and drafter, -t 6, and -ngl / -ub / -c /
cache type as arguments.

    python3 spec_ab.py RIG_DIR OUT_DIR NGL UB CTX KV [NGL_NONE]

One llama-server per arm on port 8099. Six prompts, greedy, one run each, 128 tokens, prompt
cache off. Arms: none, exact Q4_0 drafter at 1 / 2 / 3 drafted tokens, none at NGL_NONE layers
(the most that load without the drafter, if given), none again (drift check). Writes results.jsonl;
each record carries the load average, because the host is a laptop in use.
"""
import hashlib, json, os, signal, subprocess, sys, time, urllib.request

RIG, OUT, NGL, UB, CTX, KV = sys.argv[1:7]
NGL_NONE = sys.argv[7] if len(sys.argv) > 7 else None
env = {}
for line in open(os.path.join(RIG, "tpu.env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k] = v.strip('"')
MODEL = "/home/xbill/models/gemma-4-12B-it-qat-q4_0-exact/gemma-4-12B-it-q4_0-exact.gguf"
DRAFT = "/home/xbill/models/gemma-4-12B-it-qat-assistant/gemma-4-12B-it-qat-assistant-q4_0-exact.gguf"
PROMPTS = {
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "chat": "I just moved to a new city and feel a bit lonely. Any advice?",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "explain": "Explain how a refrigerator works to a ten-year-old.",
    "facts": "List the planets of the solar system with one fact about each.",
    "summary": "Summarize the causes of the French Revolution in one paragraph.",
}
ARMS = [("none", None, NGL), ("q4x-n1", "1", NGL), ("q4x-n2", "2", NGL), ("q4x-n3", "3", NGL)]
if NGL_NONE:
    ARMS.append((f"none-ngl{NGL_NONE}", None, NGL_NONE))
ARMS.append(("none-again", None, NGL))


def post(body):
    req = urllib.request.Request("http://127.0.0.1:8099/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1200))


def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def cool():
    while int(gpu().split(",")[0]) > 50:
        time.sleep(5)


os.makedirs(OUT, exist_ok=True)
out = open(os.path.join(OUT, "results.jsonl"), "w")
for arm, n, ngl in ARMS:
    cmd = [env["LLAMA_SERVER_BIN"], "-m", MODEL, "--host", "127.0.0.1", "--port", "8099",
           "-ngl", ngl, "-ub", UB, "-c", CTX, "-ctk", KV, "-ctv", KV, "-fa", "1", "-t", "6", "-tb", "12",
           "--parallel", "1", "--reasoning", "off"]
    if n:
        cmd += ["-md", DRAFT, "--spec-type", "draft-mtp", "-ngld", "all", "--spec-draft-n-max", n]
    cool()
    log = open(os.path.join(OUT, f"server-{arm}.log"), "w")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    up = False
    for _ in range(240):
        if srv.poll() is not None:
            break
        try:
            urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=2); up = True; break
        except Exception:
            time.sleep(1)
    if not up:
        print(arm, "FAILED TO LOAD", flush=True)
        srv.kill(); srv.wait(); log.close()
        continue
    post({"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16})  # warm-up
    vram = gpu()
    for name, prompt in PROMPTS.items():
        d = post({"messages": [{"role": "user", "content": prompt}], "max_tokens": 128,
                  "cache_prompt": False, "temperature": 0})
        t = d["timings"]
        text = d["choices"][0]["message"]["content"]
        rec = {"arm": arm, "mode": "greedy", "prompt": name, "rep": 0,
               "decode_tps": t["predicted_per_second"], "prompt_tps": t["prompt_per_second"],
               "n": t["predicted_n"], "draft_n": t.get("draft_n", 0),
               "draft_accepted": t.get("draft_n_accepted", 0),
               "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "gpu": vram,
               "load": os.getloadavg()[0]}
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(arm, name, round(rec["decode_tps"], 2), rec["n"], rec["draft_accepted"], "/", rec["draft_n"], vram, round(rec["load"], 2), flush=True)
    srv.send_signal(signal.SIGTERM); srv.wait(); log.close()
