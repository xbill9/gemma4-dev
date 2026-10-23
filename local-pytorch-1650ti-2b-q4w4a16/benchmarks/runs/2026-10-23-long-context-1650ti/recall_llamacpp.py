"""Ask llama.cpp the same recall prompts (prompts-*.json from dump_prompts.py), greedy, one at a time.

    python3 recall_llamacpp.py PROMPTS.json OUT.jsonl CTX

Starts llama-server on port 8099 with the llama.cpp 1650 Ti rig's model and flags, stops it by PID afterwards.
"""

import json
import subprocess
import sys
import time
import urllib.request

MODEL = "/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf"
SERVER = "/home/xbill/llama.cpp/build/bin/llama-server"
prompts, out_path, ctx = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
log = open(out_path.replace(".jsonl", ".server.log"), "w")
proc = subprocess.Popen(
    [
        SERVER,
        "-m",
        MODEL,
        "-ngl",
        "99",
        "-fa",
        "1",
        "-c",
        ctx,
        "--parallel",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        "8099",
    ],
    stdout=log,
    stderr=subprocess.STDOUT,
)
try:
    for _ in range(600):
        try:
            if b"ok" in urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=2).read():
                break
        except Exception:
            time.sleep(1)
    with open(out_path, "a") as out:
        for p in prompts:
            body = {
                "messages": [{"role": "user", "content": p["content"]}],
                "temperature": 0,
                "max_tokens": 64,
                "cache_prompt": False,
                # llama-server turns Gemma 4 thinking on by default; the PyTorch run used the template default (off)
                "chat_template_kwargs": {"enable_thinking": False},
            }
            req = urllib.request.Request(
                "http://127.0.0.1:8099/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            r = json.load(urllib.request.urlopen(req, timeout=7200))
            msg = r["choices"][0]["message"]
            text = (msg.get("content") or "").strip()
            row = {
                "engine": "llama.cpp",
                "depth": p["depth"],
                "prompt_tokens": r["usage"]["prompt_tokens"],
                "code": p["code"],
                "answer": text[:80],
                "reasoning": (msg.get("reasoning_content") or "")[:120],
                "finish": r["choices"][0]["finish_reason"],
                "recalled": str(p["code"]) in text,
            }
            print(json.dumps(row), flush=True)
            out.write(json.dumps(row) + "\n")
finally:
    proc.terminate()
    proc.wait(timeout=30)
