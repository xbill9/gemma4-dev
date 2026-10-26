"""Does drafting keep paying with a long conversation in context, on the L4?

    python3 longctx.py TPU_ENV OUT_DIR DOC

The 3050 run's method (2026-10-14-drafter-gaps-3050oem/longctx.py): -c 131072 --parallel 1, the first D tokens of
DOC (D = 0, 8192, 32768, 65536, 120000) followed by a story request, a code request and a request to summarize the
document; greedy, 256 tokens, prompt cache on, so each depth's text is read once per arm and the decode rate excludes
the prompt. DOC is this instance's NVIDIA driver README (SHA-256 in doc.sha256), a different driver version from the
3050 host's, so depth-for-depth texts differ from that run. Arms: none, drafter at 1 and 2 tokens.
"""
import hashlib, json, os, sys
from common import load_env, start, stop, post, gpu

env, OUT, DOCPATH = load_env(sys.argv[1]), sys.argv[2], sys.argv[3]
DOC = open(DOCPATH).read()
DEPTHS = [0, 8192, 32768, 65536, 120000]
QUESTIONS = {
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "summary": "Summarize the document above in one paragraph.",
}
ARMS = [("none", None), ("q4x-n1", "1"), ("q4x-n2", "2")]
out = open(os.path.join(OUT, "results.jsonl"), "w")
texts = {}
for arm, n in ARMS:
    srv, log = start(env, 131072, env["SPEC_DRAFT_MODEL"] if n else None, n, os.path.join(OUT, f"server-{arm}.log"))
    if not texts:
        ids = post("/tokenize", {"content": DOC})["tokens"]
        assert len(ids) >= DEPTHS[-1], len(ids)
        for depth in DEPTHS:
            texts[depth] = post("/detokenize", {"tokens": ids[:depth]})["content"] if depth else ""
    for depth in DEPTHS:
        for q, question in QUESTIONS.items():
            if depth == 0 and q == "summary":
                continue
            content = f"Here is a document:\n\n{texts[depth]}\n\n{question}" if depth else question
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
