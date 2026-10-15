"""Why greedy text with a drafter matches no drafter at -c 8192 but mostly differs at -c 131072.

    python3 div.py RIG_DIR OUT_DIR

In 2026-10-14-drafter-gaps-3050oem, E4B's replies matched no drafter with no text ahead of the
question even at -c 131072, and differed once 8k+ tokens of text came first. Three candidate
causes, varied one at a time here: prompt length (0 / 1k / 2k / 4k / 8k tokens of the NVIDIA
README ahead of the question), the prompt cache (off, then on), and the server context
(-c 16384 and -c 131072). Arms per context: no drafter, exact Q4_0 drafter at 2 tokens, and no
drafter again (determinism). Greedy, 256 tokens. Writes results.jsonl; common.py is copied from
2026-10-14-drafter-gaps-3050oem.
"""
import hashlib
import json
import os
import sys
from common import load_env, drafts, start, stop, post

RIG, OUT = sys.argv[1], sys.argv[2]
env = load_env(RIG)
D = drafts(env)
DOC = open("/usr/share/doc/NVIDIA_GLX-1.0/README.txt").read()
DEPTHS = [0, 1024, 2048, 4096, 8192]
QUESTIONS = {
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "summary": "Summarize the document above in one paragraph.",
}
ARMS = [("none", None), ("q4x-n2", "2"), ("none-again", None)]
out = open(os.path.join(OUT, "results.jsonl"), "w")
texts = {}
for ctx in (16384, 131072):
    for arm, n in ARMS:
        srv, log = start(env, ctx, D["q4x"] if n else None, n, os.path.join(OUT, f"server-c{ctx}-{arm}.log"))
        if not texts:
            ids = post("/tokenize", {"content": DOC})["tokens"]
            for depth in DEPTHS:
                texts[depth] = post("/detokenize", {"tokens": ids[:depth]})["content"] if depth else ""
        for cache in (False, True):
            for depth in DEPTHS:
                for q, question in QUESTIONS.items():
                    if depth == 0 and q == "summary":
                        continue
                    content = f"Here is a document:\n\n{texts[depth]}\n\n{question}" if depth else question
                    d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": content}],
                                                      "max_tokens": 256, "temperature": 0, "cache_prompt": cache})
                    t = d["timings"]
                    text = d["choices"][0]["message"]["content"]
                    rec = {"ctx": ctx, "arm": arm, "cache": cache, "depth": depth, "question": q,
                           "prompt_n": d["usage"]["prompt_tokens"], "prompt_new": t["prompt_n"],
                           "decode_tps": t["predicted_per_second"], "n": t["predicted_n"],
                           "draft_n": t.get("draft_n", 0), "draft_accepted": t.get("draft_n_accepted", 0),
                           "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(ctx, arm, cache, depth, q, rec["prompt_new"], round(rec["decode_tps"], 1), rec["n"], flush=True)
        stop(srv, log)
