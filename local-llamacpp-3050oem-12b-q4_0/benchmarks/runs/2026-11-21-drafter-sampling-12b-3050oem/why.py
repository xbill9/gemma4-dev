"""Why the 12B's greedy thinking-on replies loop: what they repeat, where it starts, and how the
same GSM8K problems went for E2B and E4B (repack, no drafter, thinking on, greedy, f16 cache)."""
import json, re, statistics as st, collections
R = "/home/xbill/gemma4-dev"
def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)}
w12 = load(f"{R}/local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-11-17-thinking-12b-3050oem/gsm8k-repack-none-thinkon.jsonl.partial")
e4 = load(f"{R}/local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-17-thinking-e4b-3050oem/gsm8k-repack-none-thinkon.jsonl")
e2 = load(f"{R}/local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem/gsm8k-repack-none-thinkon.jsonl")
def full(r): return (r.get("reasoning") or "") + (r.get("text") or "")
def cycle(t, w=200):
    """first char offset where a 200-char block recurs later, and the distance to that recurrence"""
    for p in range(0, len(t) - 2 * w, 100):
        q = t.find(t[p:p + w], p + w)
        if q != -1: return p, q - p
    return None, None
cut = [r for r in w12.values() if r["finish"] == "length"]
fin = [r for r in w12.values() if r["finish"] != "length"]
print(f"12B greedy thinking on, first {len(w12)} GSM8K: {len(cut)} hit 32,768 tokens, {len(fin)} finished ({sum(r['right'] is True for r in fin)} right)")
on, per = [], []
for r in cut:
    t = full(r); p, d = cycle(t)
    if p is None: continue
    on.append(round(p / len(t) * r["completion_tokens"])); per.append(round(d / len(t) * r["completion_tokens"]))
print(f"repetition starts at token ~{st.median(on)} (median; range {min(on)}-{max(on)}); repeating block ~{st.median(per)} tokens (median; range {min(per)}-{max(per)})")
print(f"finished replies: median {st.median(r['completion_tokens'] for r in fin)} tokens, max {max(r['completion_tokens'] for r in fin)}")
# what the repeating block says: phrases counted once per reply in the last 3,000 characters
pats = {"re-reads the wording": r"wording|phrase|phrasing|interpret|meaning|could it mean|does it mean|re-?read",
        "says 'again'": r"\bagain\b", "re-calculates": r"re-?calculat|let's check|double.check|verify",
        "states a final answer line": r"Answer:\s*\$?\d"}
for k, v in pats.items():
    n = sum(1 for r in cut if re.search(v, full(r)[-3000:], re.I))
    print(f"  last 3,000 chars of a cut reply {k}: {n} of {len(cut)}")
# did the reply already compute the right answer ("= <answer>" in the thinking), and where
hit = []
for r in cut:
    t = (r.get("reasoning") or "").replace(",", ""); a = str(r["answer"]).replace(",", "")
    m = re.search(r"=\s*\$?" + re.escape(a) + r"(?![\d.])", t)
    if m: hit.append(round(m.start() / len(t) * r["completion_tokens"]))
print(f"cut replies whose thinking already reads '= <right answer>': {len(hit)} of {len(cut)}, first at token ~{st.median(hit)} (median)")
ids = [r["id"] for r in cut]
for name, m in [("E4B", e4), ("E2B", e2)]:
    rs = [m[i] for i in ids]
    print(f"{name} on the same {len(ids)} problems: finished {sum(r['finish']!='length' for r in rs)}, right {sum(r['right'] is True for r in rs)}, "
          f"median {st.median(r['completion_tokens'] for r in rs)} tokens, max {max(r['completion_tokens'] for r in rs)}")
for name, m in [("12B finished", {r['id']: r for r in fin}), ("E4B all 1,319", e4), ("E2B all 1,319", e2)]:
    v = [r["completion_tokens"] for r in m.values()]
    print(f"{name}: median {st.median(v)} tokens, share over 4,000: {sum(x > 4000 for x in v)}/{len(v)}")
