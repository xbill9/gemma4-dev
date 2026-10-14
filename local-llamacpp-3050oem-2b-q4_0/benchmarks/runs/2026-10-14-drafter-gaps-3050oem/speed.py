"""Gaps 1 and 4 on the RTX 3050 OEM: 1 drafted token, and Google's non-QAT drafter.

    python3 speed.py RIG_DIR OUT_DIR

Same method as 2026-10-14-drafter-q4exact-3050oem/spec_ab.py: one llama-server per arm at
-c 8192 --parallel 1 with the rig's serving flags, six prompts, greedy x3 plus one sampled pass
(temperature 0.8, seed 1234), max_tokens 256, prompt cache off, GPU cooled to <=40 C per arm.
Arms: no drafter; the exact Q4_0 QAT drafter (serving now) at 1/2/3 tokens; Google's non-QAT
drafter (Q8_0) at 1/2/3 tokens; q4x-n2 again last to expose drift. Writes results.jsonl.
"""
import hashlib, json, os, sys
from common import load_env, drafts, start, stop, post, gpu

RIG, OUT = sys.argv[1], sys.argv[2]
env = load_env(RIG); D = drafts(env)
PROMPTS = {
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "chat": "I just moved to a new city and feel a bit lonely. Any advice?",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "explain": "Explain how a refrigerator works to a ten-year-old.",
    "facts": "List the planets of the solar system with one fact about each.",
    "summary": "Summarize the causes of the French Revolution in one paragraph.",
}
ARMS = [("none", None, None), ("q4x-n1", "q4x", "1"), ("q4x-n2", "q4x", "2"), ("q4x-n3", "q4x", "3"),
        ("nq-n1", "nq", "1"), ("nq-n2", "nq", "2"), ("nq-n3", "nq", "3"), ("q4x-n2-again", "q4x", "2")]
out = open(os.path.join(OUT, "results.jsonl"), "w")
for arm, draft, n in ARMS:
    srv, log = start(env, 8192, D[draft] if draft else None, n, os.path.join(OUT, f"server-{arm}.log"))
    vram = gpu()
    for mode in ("greedy", "sampled"):
        for name, prompt in PROMPTS.items():
            for r in range(3 if mode == "greedy" else 1):
                body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "cache_prompt": False}
                body.update({"temperature": 0} if mode == "greedy" else {"temperature": 0.8, "seed": 1234})
                d = post("/v1/chat/completions", body)
                t = d["timings"]; text = d["choices"][0]["message"]["content"]
                rec = {"arm": arm, "mode": mode, "prompt": name, "rep": r,
                       "decode_tps": t["predicted_per_second"], "n": t["predicted_n"],
                       "draft_n": t.get("draft_n", 0), "draft_accepted": t.get("draft_n_accepted", 0),
                       "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "gpu": vram}
                out.write(json.dumps(rec) + "\n"); out.flush()
                print(arm, mode, name, r, round(rec["decode_tps"], 1), rec["n"],
                      rec["draft_accepted"], "/", rec["draft_n"], flush=True)
    stop(srv, log)
