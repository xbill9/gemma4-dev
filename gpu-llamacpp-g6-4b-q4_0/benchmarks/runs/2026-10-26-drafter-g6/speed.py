"""Drafter speed on the L4, -c 8192: the six prompts of the 3050 runs, greedy x3 plus one sampled pass
(temperature 0.8, seed 1234), 256 tokens, prompt cache off.

    python3 speed.py TPU_ENV OUT_DIR

Arms: no drafter; the exact Q4_0 QAT drafter at 1, 2, 3 and 4 drafted tokens; no drafter again last.
Writes results.jsonl.
"""
import hashlib, json, os, sys
from common import load_env, start, stop, post, gpu

env, OUT = load_env(sys.argv[1]), sys.argv[2]
DRAFT = env["SPEC_DRAFT_MODEL"]
PROMPTS = {
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "chat": "I just moved to a new city and feel a bit lonely. Any advice?",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "explain": "Explain how a refrigerator works to a ten-year-old.",
    "facts": "List the planets of the solar system with one fact about each.",
    "summary": "Summarize the causes of the French Revolution in one paragraph.",
}
ARMS = [("none", None), ("q4x-n1", "1"), ("q4x-n2", "2"), ("q4x-n3", "3"), ("q4x-n4", "4"), ("none-again", None)]
out = open(os.path.join(OUT, "results.jsonl"), "w")
for arm, n in ARMS:
    srv, log = start(env, 8192, DRAFT if n else None, n, os.path.join(OUT, f"server-{arm}.log"))
    vram = gpu()
    for mode in ("greedy", "sampled"):
        for name, prompt in PROMPTS.items():
            for r in range(3 if mode == "greedy" else 1):
                body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "cache_prompt": False}
                body.update({"temperature": 0} if mode == "greedy" else {"temperature": 0.8, "seed": 1234})
                d = post("/v1/chat/completions", body)
                t = d["timings"]; text = d["choices"][0]["message"]["content"]
                rec = {"arm": arm, "mode": mode, "prompt": name, "rep": r, "decode_tps": t["predicted_per_second"],
                       "n": t["predicted_n"], "draft_n": t.get("draft_n", 0),
                       "draft_accepted": t.get("draft_n_accepted", 0),
                       "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "gpu": vram}
                out.write(json.dumps(rec) + "\n"); out.flush()
                print(arm, mode, name, r, round(rec["decode_tps"], 1), rec["n"],
                      rec["draft_accepted"], "/", rec["draft_n"], flush=True)
    stop(srv, log)
