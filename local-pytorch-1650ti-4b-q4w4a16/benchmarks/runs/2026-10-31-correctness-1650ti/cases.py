"""Prompts for the correctness check: short, one past the sliding ring, and across prompt-chunk boundaries."""
import json, os
def cases():
    d = os.path.dirname(os.path.abspath(__file__))
    p = json.load(open(os.path.join(d, "../2026-10-31-first-light-1650ti/prompts-eight.json")))
    out = [(f"short{i}", [{"role": "user", "content": t}], 24) for i, t in enumerate(p)]
    doc = "".join(f"Record {i:06d}: the code for record {i:06d} is {(i * 7919 + 13) % 900000 + 100000}.\n" for i in range(1, 80))
    for name, n in (("doc-20", 20), ("doc-79", 79)):
        d2 = doc[: doc.find(f"Record {n:06d}")] + doc[doc.find(f"Record {n:06d}"):].split("\n")[0]
        out.append((name, [{"role": "user", "content": d2 + f"\n\nQuestion: what is the code for record 000007? Answer with the code only."}], 24))
    return out
