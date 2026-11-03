"""Drafters on both engines on the RTX 3050 OEM, one request at a time, measured one way.

    python3 drafters.py ENGINE OUT_DIR ARM[,ARM...]

ENGINE is `vllm` (local-vllm-3050oem-2b-q4w4a16emb4, its serve.py and tpu.env) or `llama`
(local-llamacpp-3050oem-2b-q4_0, its tpu.env serving flags at -c 8192 --parallel 1, as its
2026-10-14 drafter runs). An arm is `none[-tag]` or `DRAFTER-nN[-tag]`:

    vllm   bf16  google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant, snapshot ebc7e1a2 (path, so no newer revision)
           w4a16 the same drafter repacked to compressed-tensors W4A16 (repack/repack_drafter.py)
    llama  q4x   exact Q4_0 GGUF (the rig's SPEC_DRAFT_MODEL)
           q8    llama.cpp's Q8_0 conversion
           bf16  the source weights unchanged (gguf_exact.py --bf16)

The six prompts of the 2026-10-13/14/15 drafter runs, greedy x3 plus one sampled pass
(temperature 0.8, seed 1234), 256 tokens, thinking off (chat_template_kwargs), prompt cache off
on llama.cpp. Both engines are timed the same way, client-side on the stream: decode tok/s is
(completion_tokens - 1) / (last token chunk - first token chunk), counting content and reasoning
chunks. llama.cpp's own `timings.predicted_per_second` is recorded beside it. Acceptance: llama.cpp
from the reply's `timings` (draft_n, draft_n_accepted); vLLM from the difference of its
spec-decode counters across the request. The card is cooled to <= COOL_C (default 50 C) before
each arm. Appends to OUT_DIR/results.jsonl; one server log per arm.
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ENGINE, OUT, ARMLIST = sys.argv[1:4]
COOL_C = int(os.environ.get("COOL_C", "50"))
ROOT = os.path.expanduser("~/gemma4-dev")
VRIG = os.path.join(ROOT, "local-vllm-3050oem-2b-q4w4a16emb4")
LRIG = os.path.join(ROOT, "local-llamacpp-3050oem-2b-q4_0")
VLLM_DRAFTS = {
    "bf16": "/home/xbill/.cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-q4_0-unquantized-assistant/snapshots/ebc7e1a211354561464cb82ed6d886792138dcb6",
    "w4a16": "/home/xbill/models/gemma-4-E2B-it-qat-q4_0-w4a16-ct-assistant",
}
LDIR = "/home/xbill/models/gemma-4-E2B-it-qat-assistant/gemma-4-E2B-it-qat-assistant-"
LLAMA_DRAFTS = {"q4x": LDIR + "q4_0-exact.gguf", "q8": LDIR + "q8_0.gguf", "bf16": LDIR + "bf16.gguf"}
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
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def cool():
    while int(gpu().split(",")[0]) > COOL_C:
        time.sleep(5)


def parse_arm(arm):
    p = arm.split("-")
    return (None, None) if p[0] == "none" else (p[0], int(p[1].lstrip("n")))


if ENGINE == "vllm":
    env = load_env(os.path.join(VRIG, "tpu.env"))
    URL, MODEL = env["ENDPOINT"], env["MODEL_NAME"]
else:
    env = load_env(os.path.join(LRIG, "tpu.env"))
    URL, MODEL = "http://127.0.0.1:8099", env["MODEL_NAME"]


def server_cmd(draft, n):
    if ENGINE == "vllm":
        cmd = ["python3", os.path.join(VRIG, "serve.py")]
        senv = dict(os.environ, SPECULATIVE_MODEL="")
        if draft:
            senv["SPECULATIVE_MODEL"] = VLLM_DRAFTS[draft]
            senv["NUM_SPECULATIVE_TOKENS"] = str(n)
        return cmd, senv
    cmd = [
        env["LLAMA_SERVER_BIN"],
        "-m",
        env["MODEL_PATH"],
        "--host",
        "127.0.0.1",
        "--port",
        "8099",
        "-ngl",
        "99",
        "-c",
        "8192",
        "-ctk",
        "f16",
        "-ctv",
        "f16",
        "-fa",
        "1",
        "-t",
        "6",
        "-tb",
        "12",
        "--parallel",
        "1",
        "--reasoning",
        "off",
        "-ot",
        env["OVERRIDE_TENSORS"],
        "--metrics",
    ]
    if draft:
        cmd += ["-md", LLAMA_DRAFTS[draft], "--spec-type", "draft-mtp", "-ngld", "all", "--spec-draft-n-max", str(n)]
    return cmd, dict(os.environ)


def vllm_counters():
    out = {}
    with urllib.request.urlopen(URL + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:spec_decode_num_") and "_total" in line:
                name, val = line.rsplit(" ", 1)
                k = name.split("{")[0]
                out[k] = out.get(k, 0.0) + float(val)
    return out.get("vllm:spec_decode_num_draft_tokens_total", 0.0), out.get(
        "vllm:spec_decode_num_accepted_tokens_total", 0.0
    )


def stream(body):
    body = dict(body, model=MODEL, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    stamps, text, n, timings = [], [], 0, {}
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
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
            if d.get("timings"):
                timings = d["timings"]
    tps = (n - 1) / (stamps[-1] - stamps[0]) if n > 1 and len(stamps) > 1 else 0.0
    return "".join(text), n, tps, (stamps[0] - t0) if stamps else 0.0, timings


os.makedirs(OUT, exist_ok=True)
out = open(os.path.join(OUT, "results.jsonl"), "a")
for arm in ARMLIST.split(","):
    draft, n = parse_arm(arm)
    cmd, senv = server_cmd(draft, n)
    cool()
    log = open(os.path.join(OUT, f"server-{ENGINE}-{arm}.log"), "w")
    t_start = time.time()
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=senv, start_new_session=True)
    while True:
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            break
        except Exception:
            if srv.poll() is not None:
                sys.exit(f"{arm}: server exited, see server-{ENGINE}-{arm}.log")
            time.sleep(2)
    ready_s = round(time.time() - t_start, 1)
    stream({"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16})  # warm-up
    vram = gpu()
    for mode in ("greedy", "sampled"):
        for name, prompt in PROMPTS.items():
            for rep in range(REPEATS if mode == "greedy" else 1):
                body = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                body.update({"temperature": 0} if mode == "greedy" else {"temperature": 0.8, "seed": 1234})
                if ENGINE == "llama":
                    body["cache_prompt"] = False
                c0 = vllm_counters() if ENGINE == "vllm" else None
                text, ntok, tps, ttft, timings = stream(body)
                if ENGINE == "vllm":
                    c1 = vllm_counters()
                    dn, da = c1[0] - c0[0], c1[1] - c0[1]
                else:
                    dn, da = timings.get("draft_n", 0), timings.get("draft_n_accepted", 0)
                rec = {
                    "engine": ENGINE,
                    "arm": arm,
                    "draft": draft,
                    "n_draft": n,
                    "mode": mode,
                    "prompt": name,
                    "rep": rep,
                    "decode_tps": tps,
                    "server_tps": timings.get("predicted_per_second"),
                    "ttft_s": ttft,
                    "tokens": ntok,
                    "draft_n": dn,
                    "draft_accepted": da,
                    "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
                    "gpu": vram,
                    "ready_s": ready_s,
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(ENGINE, arm, mode, name, rep, round(tps, 1), ntok, da, "/", dn, flush=True)
    os.killpg(srv.pid, signal.SIGTERM)
    srv.wait()
    log.close()
