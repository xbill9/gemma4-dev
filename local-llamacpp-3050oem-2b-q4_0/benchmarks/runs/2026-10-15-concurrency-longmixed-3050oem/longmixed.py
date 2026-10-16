"""Concurrency with long, varied prompts: c requests at once, each one of six prompts built from
different parts of the NVIDIA driver README (1,000-3,500 tokens of it, SHA-256 in
../2026-10-14-drafter-gaps-3050oem/doc.sha256) with a different task each. 256 tokens out,
greedy, prompt cache off, thinking off, streamed. Prompt texts are written to prompts.json on the
first run and reused, so every arm and pass sends identical text.

    python3 longmixed.py --model NAME --out FILE.jsonl --levels 1,2,4 --repeats 6 [--pass N]
"""
import argparse, concurrent.futures, json, os, time, urllib.request

DOC = "/usr/share/doc/NVIDIA_GLX-1.0/README.txt"
PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")
# (offset in tokens, length in tokens, task)
SPECS = [
    (0, 1000, "Summarize the document above in one paragraph."),
    (20000, 3500, "List every command-line option or configuration setting the document above mentions, one per line with a short description."),
    (40000, 1500, "Write a bash script that checks the things the document above says to check, with comments."),
    (60000, 3000, "Explain the document above to someone who has never used Linux."),
    (80000, 2000, "Write five quiz questions about the document above, with answers."),
    (100000, 2500, "Rewrite the document above as a short FAQ."),
]

def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as fh:
        return json.load(fh)

def prompts(base):
    if os.path.exists(PROMPTS_FILE):
        return json.load(open(PROMPTS_FILE))
    ids = post(base, "/tokenize", {"content": open(DOC).read()})["tokens"]
    out = []
    for off, n, task in SPECS:
        text = post(base, "/detokenize", {"tokens": ids[off:off + n]})["content"]
        out.append(f"Here is a document:\n\n{text}\n\n{task}")
    json.dump(out, open(PROMPTS_FILE, "w"), indent=1)
    return out

def one(base, model, prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 256,
            "temperature": 0, "cache_prompt": False, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; timings = {}
    with urllib.request.urlopen(req, timeout=3600) as fh:
        for raw in fh:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ch = json.loads(line[5:])
            if ttft is None and any((c.get("delta") or {}).get("content") for c in ch.get("choices", [])):
                ttft = time.perf_counter() - t0
            if ch.get("timings"):
                timings = ch["timings"]
    return {"ttft_ms": round((ttft or 0) * 1000, 1), "wall_s": round(time.perf_counter() - t0, 3),
            "prompt_n": timings.get("prompt_n", 0), "n": timings.get("predicted_n", 0),
            "decode_tps": timings.get("predicted_per_second", 0),
            "draft_n": timings.get("draft_n", 0), "draft_accepted": timings.get("draft_n_accepted", 0)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="1,2,4,8,16,32")
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--pass", dest="pas", type=int, default=1)
    a = ap.parse_args()
    P = prompts(a.base)
    one(a.base, a.model, P[0])                                     # warm-up, not recorded
    with open(a.out, "a") as out:
        for c in map(int, a.levels.split(",")):
            for r in range(a.repeats):
                ps = [(i + r) % len(P) for i in range(c)]
                t0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                    runs = list(pool.map(lambda k: dict(one(a.base, a.model, P[k]), prompt=k), ps))
                wall = time.perf_counter() - t0
                rec = {"pass": a.pas, "c": c, "rep": r, "wall_s": round(wall, 3), "runs": runs}
                out.write(json.dumps(rec) + "\n"); out.flush()
                print(a.pas, c, r, round(sum(x["n"] for x in runs) / wall, 1), flush=True)

main()
