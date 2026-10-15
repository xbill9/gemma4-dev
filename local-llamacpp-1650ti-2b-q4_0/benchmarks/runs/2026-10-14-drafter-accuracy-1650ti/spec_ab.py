"""Drafter precision on the GTX 1650 Ti Max-Q: Google's QAT assistant drafter as Q8_0 (make draft),
exact Q4_0 (gguf_exact.py) and bf16, the three files built and hashed in
local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-q4exact-3050oem/. Copied from
2026-10-14-drafter-1650ti/spec_ab.py with the arms changed: same six prompts, greedy x3 plus one
sampled pass (temperature 0.8, seed 1234), 256 tokens, prompt cache off.

    python3 spec_ab.py RIG_DIR OUT_DIR ARM[,ARM...]

An arm is none[-tag] or FORMAT-nN[-tag], FORMAT one of q8, q4x, bf16. The drafter files sit beside
the rig's SPEC_DRAFT_MODEL (the exact Q4_0 one). LLAMA_SERVER_BIN in the environment overrides
tpu.env's, to measure another llama.cpp build with everything else the same.

One llama-server per arm on port 8099 at -c 8192 --parallel 1 with the rig's serving flags
(binary and model from its tpu.env). The card is cooled to <= COOL_C (default 50 °C) before
each arm. The no-drafter arm runs first and last to expose drift. Writes results.jsonl.
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
PROMPTS = {
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "chat": "I just moved to a new city and feel a bit lonely. Any advice?",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "explain": "Explain how a refrigerator works to a ten-year-old.",
    "facts": "List the planets of the solar system with one fact about each.",
    "summary": "Summarize the causes of the French Revolution in one paragraph.",
}
for k in ("LLAMA_SERVER_BIN",):  # the environment wins over tpu.env, as everywhere in this tree
    if os.environ.get(k):
        env[k] = os.environ[k]
Q4X = env["SPEC_DRAFT_MODEL"]
assert Q4X.endswith("-q4_0-exact.gguf"), Q4X
DRAFTS = {"q4x": Q4X, "q8": Q4X.replace("-q4_0-exact.gguf", "-q8_0.gguf"),
          "bf16": Q4X.replace("-q4_0-exact.gguf", "-bf16.gguf")}
ARMS = []
for arm in ARMLIST.split(","):
    p = arm.split("-")
    ARMS.append((arm, None, None) if p[0] == "none" else (arm, DRAFTS[p[0]], p[1].lstrip("n")))
REPEATS = 3

def post(body):
    req = urllib.request.Request("http://127.0.0.1:8099/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))

def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

def cool():
    while int(gpu().split(",")[0]) > COOL_C:
        time.sleep(5)

os.makedirs(OUT, exist_ok=True)
out = open(os.path.join(OUT, "results.jsonl"), "w")
for arm, draft, n in ARMS:
    cmd = [env["LLAMA_SERVER_BIN"], "-m", env["MODEL_PATH"], "--host", "127.0.0.1", "--port", "8099",
           "-ngl", env["N_GPU_LAYERS"], "-c", "8192", "-ctk", env["KV_CACHE_TYPE"], "-ctv", env["KV_CACHE_TYPE"],
           "-fa", env["FLASH_ATTENTION"], "-t", env["THREADS"], "-tb", env["THREADS_BATCH"],
           "--parallel", "1", "--reasoning", env["REASONING"]]
    if env.get("OVERRIDE_TENSORS"):
        cmd += ["-ot", env["OVERRIDE_TENSORS"]]
    if n:
        cmd += ["-md", draft, "--spec-type", "draft-mtp", "-ngld", "all", "--spec-draft-n-max", n]
    cool()
    log = open(os.path.join(OUT, f"server-{arm}.log"), "w")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(180):
        try:
            urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=2); break
        except Exception:
            if srv.poll() is not None:
                sys.exit(f"{arm}: llama-server exited, see server-{arm}.log")
            time.sleep(1)
    post({"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16})  # warm-up
    vram = gpu()
    for mode in ("greedy", "sampled"):
        reps = REPEATS if mode == "greedy" else 1
        for name, prompt in PROMPTS.items():
            for r in range(reps):
                body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 256,
                        "cache_prompt": False}
                body.update({"temperature": 0} if mode == "greedy" else {"temperature": 0.8, "seed": 1234})
                d = post(body)
                t = d["timings"]
                text = d["choices"][0]["message"]["content"]
                rec = {"arm": arm, "mode": mode, "prompt": name, "rep": r,
                       "decode_tps": t["predicted_per_second"], "n": t["predicted_n"],
                       "draft_n": t.get("draft_n", 0), "draft_accepted": t.get("draft_n_accepted", 0),
                       "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "gpu": vram,
                       "gpu_after": gpu()}
                out.write(json.dumps(rec) + "\n"); out.flush()
                print(arm, mode, name, r, round(rec["decode_tps"], 1), rec["n"],
                      rec["draft_accepted"], "/", rec["draft_n"], rec["gpu_after"], flush=True)
    srv.send_signal(signal.SIGTERM); srv.wait(); log.close()
