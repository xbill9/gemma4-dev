"""Job E: why E4B's greedy replies change between identical requests on the GTX 1650 Ti.

    python3 repeat.py OUT_DIR

The 2026-10-14 1650 Ti drafter runs found E4B giving 2-3 different texts over 3 repeats of a
prompt at temperature 0, with or without the drafter; E2B on this card and E4B on the RTX 3050
repeat exactly. Each arm here is one llama-server at -c 8192 --parallel 1 with the rig's flags
(no -ot), the prompt cache off, and 10 repeats each of a story and a code prompt, 256 tokens.
One setting changes per arm. Writes results.jsonl (texts included).
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

OUT = sys.argv[1]
M = os.path.expanduser("~/models")
E4B = f"{M}/gemma-4-E4B-it-qat-q4_0-exact/gemma-4-E4B-it-q4_0-exact.gguf"
E2B = f"{M}/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf"
D4 = f"{M}/gemma-4-E4B-it-qat-assistant/gemma-4-E4B-it-qat-assistant-q4_0-exact.gguf"
F95 = os.path.expanduser("~/llama.cpp/build/bin/llama-server")          # the rigs' build, f95b0d9
B77 = os.path.expanduser("~/llama.cpp-77c4540ce/build/bin/llama-server")  # the 3050's build
BASE = ["-ngl", "99", "-c", "8192", "-ctk", "f16", "-ctv", "f16", "--parallel", "1", "--reasoning", "off"]
ARMS = [  # name, binary, model, extra flags, extra env
    ("e4b-f95", F95, E4B, ["-fa", "1", "-t", "6", "-tb", "12"], {}),
    ("e4b-f95-fa0", F95, E4B, ["-fa", "0", "-t", "6", "-tb", "12"], {}),
    ("e4b-f95-t1", F95, E4B, ["-fa", "1", "-t", "1", "-tb", "1"], {}),
    ("e4b-f95-nographs", F95, E4B, ["-fa", "1", "-t", "6", "-tb", "12"], {"GGML_CUDA_DISABLE_GRAPHS": "1"}),
    ("e4b-f95-drafter2", F95, E4B, ["-fa", "1", "-t", "6", "-tb", "12", "-md", D4, "--spec-type", "draft-mtp",
                                     "-ngld", "all", "--spec-draft-n-max", "2"], {}),
    ("e4b-77c", B77, E4B, ["-fa", "1", "-t", "6", "-tb", "12"], {}),
    ("e2b-f95", F95, E2B, ["-fa", "1", "-t", "6", "-tb", "12"], {}),
]
PROMPTS = {
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
}
URL = "http://127.0.0.1:8099"

def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

def post(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))

out = open(os.path.join(OUT, "results.jsonl"), "w")
for name, binary, model, extra, envx in ARMS:
    if not os.path.exists(binary):
        print(name, "SKIP: no", binary, flush=True)
        continue
    while int(gpu().split(",")[0]) > 50:
        time.sleep(5)
    log = open(os.path.join(OUT, f"server-{name}.log"), "w")
    srv = subprocess.Popen([binary, "-m", model, "--host", "127.0.0.1", "--port", "8099"] + BASE + extra,
                           stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **envx})
    for _ in range(300):
        if srv.poll() is not None:
            break
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    if srv.poll() is not None:
        print(name, "SKIP: server exited", flush=True)
        continue
    post({"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16})
    for p, prompt in PROMPTS.items():
        for r in range(10):
            d = post({"messages": [{"role": "user", "content": prompt}], "max_tokens": 256,
                      "temperature": 0, "cache_prompt": False})
            text = d["choices"][0]["message"]["content"]
            rec = {"arm": name, "prompt": p, "rep": r, "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
                   "decode_tps": d["timings"]["predicted_per_second"], "n": d["timings"]["predicted_n"],
                   "gpu": gpu(), "text": text}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(name, p, r, rec["sha"], round(rec["decode_tps"], 1), flush=True)
    srv.send_signal(signal.SIGTERM)
    srv.wait()
    log.close()
