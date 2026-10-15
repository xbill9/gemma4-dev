"""Does the drafter still pay when several requests share the T4? The rigs serve with
--max-num-seqs 8 and the QAT drafter at 3 tokens; every drafter run so far was one request at a time.

    python3 conc.py RIG_DIR OUT_DIR ARM[,ARM...]

Per arm one vLLM server (t4serve.py, the rig's serving flags). Per concurrency C in 1, 2, 4, 8:
max(16, 4C) greedy chat requests of 256 tokens drawn in a fixed order from 26 real prompts (the six
of the drafter sweep and the twenty of 2026-09-29-specdec-c1-t4), C in flight at a time; the cell
is run twice. Records per request decode tok/s, TTFT and tokens; per cell wall time, output
tokens and the spec-decode counters' difference. Writes OUT_DIR/results.jsonl.
"""
import hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t4serve import load_env, parse_arm, start, stop, spec_counters, stream_chat, gpu

RIG, OUT, ARMLIST = sys.argv[1:4]
env = load_env(RIG); URL = env["ENDPOINT"]; MODEL = env["MODEL_NAME"]
PROMPTS = [
    "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "I just moved to a new city and feel a bit lonely. Any advice?",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how a refrigerator works to a ten-year-old.",
    "List the planets of the solar system with one fact about each.",
    "Summarize the causes of the French Revolution in one paragraph.",
    "Why is the sky blue? Answer in two short paragraphs.",
    "Write a Python function that checks whether a string is a palindrome, with a docstring.",
    "Explain what a TPU is to a high-school student.",
    "Give me five name ideas for a coffee shop run by robots.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "What are the pros and cons of running a language model locally instead of in the cloud?",
    "Translate into French: 'The meeting has been moved to Thursday afternoon.'",
    "Write a haiku about a GPU that is running out of memory.",
    "How do I reverse a linked list? Explain, then show code in C.",
    "List the planets of the solar system with one interesting fact each.",
    "What is the difference between a process and a thread?",
    "Draft a polite email declining a meeting invitation.",
    "Explain quantization of neural network weights in simple terms.",
    "Write a SQL query that returns the top 3 customers by total order value.",
    "What should I pack for a three-day hiking trip?",
    "Explain the Monty Hall problem and why switching is better.",
    "Write a short bash script that backs up a directory with a timestamp.",
    "What is Kubernetes and why would someone use it?",
    "Give a one-paragraph history of the Internet.",
    "Tell me a short story about a cat who learns to code.",
]
CONC = [int(c) for c in os.environ.get("CONC", "1,2,4,8").split(",")]
REPEATS = int(os.environ.get("REPEATS", "2"))
os.makedirs(OUT, exist_ok=True)
out = open(os.path.join(OUT, "results.jsonl"), "a")
cells = open(os.path.join(OUT, "cells.jsonl"), "a")
for arm in ARMLIST.split(","):
    draft, n = parse_arm(env, arm)
    srv, log = start(env, os.path.join(OUT, f"server-{arm}.log"), draft, n)
    try:
        stream_chat(URL, MODEL, {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16})
        for c in CONC:
            nreq = max(16, 4 * c)
            for rep in range(REPEATS):
                order = [PROMPTS[(i + 7 * rep) % len(PROMPTS)] for i in range(nreq)]
                d0, a0 = spec_counters(URL)
                t0 = time.perf_counter()
                def one(i):
                    text, ntok, ttft, dec, e2e = stream_chat(URL, MODEL, {
                        "messages": [{"role": "user", "content": order[i]}], "max_tokens": 256, "temperature": 0})
                    return {"arm": arm, "conc": c, "rep": rep, "i": i, "n": ntok, "ttft_s": ttft,
                            "decode_tps": dec, "e2e_s": e2e, "sha": hashlib.sha256(text.encode()).hexdigest()[:16]}
                with ThreadPoolExecutor(c) as pool:
                    rows = list(pool.map(one, range(nreq)))
                wall = time.perf_counter() - t0
                d1, a1 = spec_counters(URL)
                for r in rows:
                    out.write(json.dumps(r) + "\n")
                cell = {"arm": arm, "conc": c, "rep": rep, "requests": nreq, "wall_s": wall,
                        "out_tokens": sum(r["n"] for r in rows), "out_tps": sum(r["n"] for r in rows) / wall,
                        "draft_n": d1 - d0, "draft_accepted": a1 - a0, "gpu": gpu()}
                cells.write(json.dumps(cell) + "\n"); out.flush(); cells.flush()
                print(arm, "c", c, "rep", rep, round(cell["out_tps"], 1), "tok/s",
                      int(a1 - a0), "/", int(d1 - d0), cell["gpu"], flush=True)
    finally:
        stop(srv, log)
