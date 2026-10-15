"""Gap 2 on the RTX 3050 OEM: does drafting keep paying with a long conversation in context?

    python3 longctx.py RIG_DIR OUT_DIR [ARM,ARM,...]

Naming arms runs only those and appends to results.jsonl (used to finish E4B after an interruption).

One llama-server per arm at -c 131072 (the rigs' serving context) --parallel 1, serving flags.
Context: the first D tokens of the NVIDIA driver README (/usr/share/doc/NVIDIA_GLX-1.0/README.txt,
English technical prose; SHA-256 in doc.sha256), D = 0, 8192, 32768, 65536, 120000, followed by
one of three questions: a story and a code request unrelated to the document, and a summary of it.
Greedy, max_tokens 256, prompt cache ON so each depth's document is read once per arm; the
decode rate (timings.predicted_per_second) excludes reading the prompt. Writes results.jsonl.
"""
import hashlib, json, os, sys
from common import load_env, drafts, start, stop, post, gpu

RIG, OUT = sys.argv[1], sys.argv[2]
env = load_env(RIG); D = drafts(env)
DOC = open("/usr/share/doc/NVIDIA_GLX-1.0/README.txt").read()
DEPTHS = [0, 8192, 32768, 65536, 120000]
QUESTIONS = {
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "summary": "Summarize the document above in one paragraph.",
}
ARMS = [("none", None, None), ("q4x-n1", "q4x", "1"), ("q4x-n2", "q4x", "2"), ("nq-n2", "nq", "2")]
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
if ONLY:
    ARMS = [a for a in ARMS if a[0] in ONLY]
out = open(os.path.join(OUT, "results.jsonl"), "a" if ONLY else "w")
texts = {}
for arm, draft, n in ARMS:
    srv, log = start(env, 131072, D[draft] if draft else None, n, os.path.join(OUT, f"server-{arm}.log"))
    if not texts:
        ids = post("/tokenize", {"content": DOC})["tokens"]
        assert len(ids) >= DEPTHS[-1], len(ids)
        for depth in DEPTHS:
            texts[depth] = post("/detokenize", {"tokens": ids[:depth]})["content"] if depth else ""
    for depth in DEPTHS:
        for q, question in QUESTIONS.items():
            if depth == 0 and q == "summary":
                continue
            content = (f"Here is a document:\n\n{texts[depth]}\n\n{question}" if depth else question)
            d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": content}],
                                              "max_tokens": 256, "temperature": 0, "cache_prompt": True})
            t = d["timings"]; text = d["choices"][0]["message"]["content"]
            rec = {"arm": arm, "depth": depth, "question": q, "prompt_n": d["usage"]["prompt_tokens"],
                   "prompt_new": t["prompt_n"], "prompt_tps": t["prompt_per_second"],
                   "decode_tps": t["predicted_per_second"], "n": t["predicted_n"],
                   "draft_n": t.get("draft_n", 0), "draft_accepted": t.get("draft_n_accepted", 0),
                   "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "gpu": gpu()}
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(arm, depth, q, rec["prompt_n"], round(rec["decode_tps"], 1), rec["n"],
                  rec["draft_accepted"], "/", rec["draft_n"], flush=True)
    stop(srv, log)
