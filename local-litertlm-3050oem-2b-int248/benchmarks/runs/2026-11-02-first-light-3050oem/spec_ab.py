"""Generation speed with and without LiteRT-LM's built-in MTP drafter.

One `litert-lm serve` per arm on port 9399, the rig's config with only
speculative_decoding changed. The six prompts of
local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-13-spec-mtp-3050oem,
greedy (temperature 0), max 256 tokens (sent as max_completion_tokens: the
server ignores max_tokens), 3 repeats, GPU cooled to <= 40 C
before each arm.

The server reports no timings, so decode speed is measured on the stream:
(completion_tokens - 1) / (time of last content chunk - time of first),
completion_tokens from the server's usage chunk. Prompt speed is not
measured here.

    python3 spec_ab.py RIG_DIR OUT_DIR
"""

import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROMPTS = {
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "chat": "I just moved to a new city and feel a bit lonely. Any advice?",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "explain": "Explain how a refrigerator works to a ten-year-old.",
    "facts": "List the planets of the solar system with one fact about each.",
    "summary": "Summarize the causes of the French Revolution in one paragraph.",
}
ARMS = ["off", "on", "adaptive"]
REPEATS = 3
PORT = 9399
URL = f"http://127.0.0.1:{PORT}"


def gpu():
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()


def cool():
    while int(gpu().split(",")[0]) > 40:
        time.sleep(5)


def stream(model, prompt):
    body = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_completion_tokens": 256,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    first = last = None
    chunks = 0
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                piece = c.get("delta", {}).get("content")
                if piece:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    chunks += 1
                    text.append(piece)
    n = usage["completion_tokens"]
    return {
        "ttft_s": first - t0,
        "n": n,
        "chunks": chunks,
        "decode_tps": (n - 1) / (last - first) if last > first else None,
        "text": "".join(text),
    }


def main():
    rig, out_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(rig))
    import render_config

    env = render_config.read_env(rig / "tpu.env")
    model = env["MODEL_ID"]
    os.environ["LITERT_LM_DIR"] = env["LITERT_LM_DIR"]
    results = open(out_dir / "results.jsonl", "a")
    texts = {}
    for arm in ARMS:
        os.environ["SPECULATIVE_DECODING"] = arm
        cfg_path = out_dir / f"config-{arm}.json"
        cfg_path.write_text(json.dumps(render_config.build_config(env), indent=2) + "\n")
        cool()
        log = open(out_dir / f"server-{arm}.log", "w")
        srv = subprocess.Popen(
            ["litert-lm", "serve", "--config", str(cfg_path), "--port", str(PORT)], stdout=log, stderr=subprocess.STDOUT
        )
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{URL}/v1/models", timeout=2)
                break
            except OSError:
                time.sleep(1)
        stream(model, "Say hello.")  # loads the engine; not recorded
        vram = gpu()
        for name, prompt in PROMPTS.items():
            for rep in range(REPEATS):
                r = stream(model, prompt)
                sha = hashlib.sha256(r["text"].encode()).hexdigest()[:16]
                texts.setdefault(arm, {}).setdefault(name, []).append(r["text"])
                rec = {
                    "arm": arm,
                    "prompt": name,
                    "rep": rep,
                    "decode_tps": r["decode_tps"],
                    "n": r["n"],
                    "chunks": r["chunks"],
                    "ttft_s": r["ttft_s"],
                    "sha": sha,
                    "gpu_after_load": vram,
                    "gpu_now": gpu(),
                }
                results.write(json.dumps(rec) + "\n")
                results.flush()
                print(arm, name, rep, round(r["decode_tps"], 1), r["n"], r["chunks"], sha, flush=True)
        srv.send_signal(signal.SIGTERM)
        srv.wait()
        log.close()
    (out_dir / "texts.json").write_text(json.dumps(texts, indent=1) + "\n")
    summarize(out_dir)


def summarize(out_dir):
    rows = [json.loads(x) for x in open(out_dir / "results.jsonl")]
    med = {}
    for arm in ARMS:
        for name in PROMPTS:
            v = [r["decode_tps"] for r in rows if r["arm"] == arm and r["prompt"] == name]
            if v:
                med[(arm, name)] = (statistics.median(v), (max(v) - min(v)) / statistics.median(v))
    lines = ["prompt   " + "".join(f"{a:>20s}" for a in ARMS)]
    for name in PROMPTS:
        base = med[("off", name)][0]
        cells = []
        for arm in ARMS:
            m, spread = med[(arm, name)]
            cells.append(f"{m:8.1f} ({m / base:4.2f}x) {spread:4.1%}")
        lines.append(f"{name:8s} " + "".join(f"{c:>20s}" for c in cells))
    for arm in ARMS:
        ratios = [med[(arm, n)][0] / med[("off", n)][0] for n in PROMPTS]
        lines.append(f"geomean {arm}: {statistics.geometric_mean(ratios):.3f}")
    shas = {}
    for r in rows:
        shas.setdefault((r["arm"], r["prompt"]), set()).add(r["sha"])
    lines.append(
        "distinct replies per (arm, prompt) over repeats: "
        + json.dumps({f"{a}/{p}": len(s) for (a, p), s in shas.items()})
    )
    same = sum(1 for n in PROMPTS if shas[("on", n)] == shas[("off", n)])
    lines.append(f"drafter on vs off, identical greedy reply: {same} of {len(PROMPTS)} prompts")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--summarize":
        summarize(Path(".").resolve())
    else:
        main()
