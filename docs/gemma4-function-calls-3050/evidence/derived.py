#!/usr/bin/env python3
"""derived.py -- every figure the function-calls article quotes, computed from the run's replies.
Run from this directory; reads ../../../local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-23-functions-e2b-e4b-12b-3050oem.
Writes derived.txt."""
import json, os, statistics
from scipy.stats import binomtest, norm

RUN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
      "../../../local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-23-functions-e2b-e4b-12b-3050oem"))
OCT = os.path.normpath(os.path.join(RUN, "../../../../"))
MODELS, TASKS = ["e2b", "e4b", "12b"], ["bfcl_simple", "bfcl_multiple", "bfcl_irrelevance"]
SETS = [("greedy", "thinkoff"), ("sampled", "thinkoff"), ("sampled", "thinkon"), ("greedy", "thinkon")]
out = []
def p(*a): out.append(" ".join(str(x) for x in a))


def load(m, t, mode, th):
    return {f"{t}:{r['id']}": r for r in map(json.loads, open(f"{RUN}/{t}-{m}-{mode}-{th}.jsonl"))}


def pooled(m, mode, th, tasks=TASKS):
    d = {}
    for t in tasks:
        d.update(load(m, t, mode, th))
    return d


def wilson(k, n, z=norm.ppf(0.975)):
    q = k / n; d = 1 + z * z / n; c = q + z * z / (2 * n); h = z * (q * (1 - q) / n + z * z / (4 * n * n)) ** 0.5
    return (c - h) / d, (c + h) / d


def mc(a, b):
    x = sum(a[i]["right"] and not b[i]["right"] for i in a); y = sum(b[i]["right"] and not a[i]["right"] for i in a)
    return x, y, binomtest(x, x + y).pvalue if x + y else 1.0


def hours(d): return sum(r["seconds"] for r in d.values()) / 3600
def right(d): return sum(r["right"] for r in d.values())
def nocall(d): return sum(not r.get("calls") for r in d.values())
def thought(d): return sum(bool(r.get("reasoning")) for r in d.values())
def cut(d): return sum(r.get("finish") == "length" for r in d.values())


p("== pooled 840: right, score, 95% range, hours, seconds/problem, thought, cut")
for m in MODELS:
    for mode, th in SETS:
        d = pooled(m, mode, th); n = len(d); k = right(d); lo, hi = wilson(k, n)
        p(f"{m} {mode} {th}: {k}/{n} {k/n:.4f} [{lo:.3f}, {hi:.3f}] hours {hours(d):.2f} s/problem {hours(d)*3600/n:.2f} thought {thought(d)}/{n} cut {cut(d)}")

p("\n== per category right / n")
for m in MODELS:
    for mode, th in SETS:
        p(f"{m} {mode} {th}: " + "  ".join(f"{t[5:]} {right(load(m,t,mode,th))}/{len(load(m,t,mode,th))}" for t in TASKS))

p("\n== calls offered a fitting tool (simple+multiple, 600): replies with no tool call")
for m in MODELS:
    for mode, th in SETS:
        d = pooled(m, mode, th, TASKS[:2]); p(f"{m} {mode} {th}: no call {nocall(d)}/{len(d)} ({nocall(d)/len(d):.3f}, 1 in {len(d)/max(nocall(d),1):.1f})")
p("\n== irrelevance (240): replies that called the tool anyway")
for m in MODELS:
    for mode, th in SETS:
        d = load(m, "bfcl_irrelevance", mode, th); p(f"{m} {mode} {th}: called {len(d)-nocall(d)}/240 cut {cut(d)}")

p("\n== paired, pooled 840 (A right only, B right only, p)")
def pr(name, a, b):
    x, y, q = mc(a, b); p(f"{name}: +{x} / -{y} p={q:.3g}")
for m in MODELS:
    pr(f"{m} thinking off, greedy vs Google's", pooled(m, "greedy", "thinkoff"), pooled(m, "sampled", "thinkoff"))
    pr(f"{m} Google's, thinking on vs off", pooled(m, "sampled", "thinkon"), pooled(m, "sampled", "thinkoff"))
    pr(f"{m} greedy, thinking on vs off", pooled(m, "greedy", "thinkon"), pooled(m, "greedy", "thinkoff"))
    pr(f"{m} thinking on, greedy vs Google's", pooled(m, "greedy", "thinkon"), pooled(m, "sampled", "thinkon"))
pr("e4b Google's thinking on vs 12b greedy thinking off", pooled("e4b", "sampled", "thinkon"), pooled("12b", "greedy", "thinkoff"))
pr("e4b Google's thinking on vs 12b Google's thinking on", pooled("e4b", "sampled", "thinkon"), pooled("12b", "sampled", "thinkon"))
pr("e4b Google's thinking on vs e2b Google's thinking on", pooled("e4b", "sampled", "thinkon"), pooled("e2b", "sampled", "thinkon"))
pr("e4b greedy thinking on vs e4b Google's thinking on", pooled("e4b", "greedy", "thinkon"), pooled("e4b", "sampled", "thinkon"))
pr("e4b greedy thinking on vs 12b greedy thinking off", pooled("e4b", "greedy", "thinkon"), pooled("12b", "greedy", "thinkoff"))
for t in TASKS:
    pr(f"12b vs e4b greedy thinking off, {t[5:]}", load("12b", t, "greedy", "thinkoff"), load("e4b", t, "greedy", "thinkoff"))
pr("12b vs e4b greedy thinking off, all 840", pooled("12b", "greedy", "thinkoff"), pooled("e4b", "greedy", "thinkoff"))
for t in TASKS:
    pr(f"e4b vs 12b Google's thinking on, {t[5:]}", load("e4b", t, "sampled", "thinkon"), load("12b", t, "sampled", "thinkon"))
pr("12b irrelevance greedy thinking on vs off", load("12b", "bfcl_irrelevance", "greedy", "thinkon"), load("12b", "bfcl_irrelevance", "greedy", "thinkoff"))

p("\n== time ratios (hours)")
def ratio(name, a, b): p(f"{name}: {hours(a)/hours(b):.2f}x")
ratio("12b Google's on / e4b Google's on", pooled("12b", "sampled", "thinkon"), pooled("e4b", "sampled", "thinkon"))
ratio("12b greedy off / e4b Google's on", pooled("12b", "greedy", "thinkoff"), pooled("e4b", "sampled", "thinkon"))
ratio("e4b greedy on / e4b Google's on", pooled("e4b", "greedy", "thinkon"), pooled("e4b", "sampled", "thinkon"))
ratio("e4b Google's on / e4b greedy off", pooled("e4b", "sampled", "thinkon"), pooled("e4b", "greedy", "thinkoff"))
ratio("e4b Google's on / e2b Google's on", pooled("e4b", "sampled", "thinkon"), pooled("e2b", "sampled", "thinkon"))
ratio("12b greedy on / 12b greedy off", pooled("12b", "greedy", "thinkon"), pooled("12b", "greedy", "thinkoff"))

p("\n== Google's settings, thinking off: replies that carried thinking anyway, and right rate with/without it")
for m in MODELS:
    d = pooled(m, "sampled", "thinkoff"); a = [r for r in d.values() if r.get("reasoning")]; b = [r for r in d.values() if not r.get("reasoning")]
    p(f"{m}: thought {len(a)}/840; right with thought {sum(r['right'] for r in a)}/{len(a)}; without {sum(r['right'] for r in b)}/{len(b)}")
    d = pooled(m, "sampled", "thinkoff", TASKS[:2]); a = [r for r in d.values() if r.get("reasoning")]; b = [r for r in d.values() if not r.get("reasoning")]
    p(f"   simple+multiple: right with thought {sum(r['right'] for r in a)}/{len(a)}; without {sum(r['right'] for r in b)}/{len(b)}; no call without thought {sum(not r.get('calls') for r in b)}/{len(b)}")
p("e4b Google's thinking on: replies with thinking", thought(pooled("e4b", "sampled", "thinkon")), "/840;", "without thinking right",
  sum(r["right"] for r in pooled("e4b", "sampled", "thinkon").values() if not r.get("reasoning")), "/",
  sum(1 for r in pooled("e4b", "sampled", "thinkon").values() if not r.get("reasoning")))

p("\n== mean completion tokens per reply, pooled 840")
for m in MODELS:
    p(m + ": " + "  ".join(f"{mo} {th} {statistics.mean(r['completion_tokens'] for r in pooled(m,mo,th).values()):.0f}" for mo, th in SETS))

p("\n== prose instead of a call: E4B Google's thinking off, simple, first replies with no call and no thinking")
d = load("e4b", "bfcl_simple", "sampled", "thinkoff")
for r in [r for r in d.values() if not r.get("calls") and not r.get("reasoning")][:4]:
    p(f"{r['id']}: {json.dumps((r.get('text') or '')[:160])}")
p("same ids, E4B greedy thinking off:")
g = load("e4b", "bfcl_simple", "greedy", "thinkoff")
for r in [r for r in d.values() if not r.get("calls") and not r.get("reasoning")][:4]:
    k = f"bfcl_simple:{r['id']}"; p(f"{r['id']}: right={g[k]['right']} calls={json.dumps(g[k].get('calls'))[:160]}")

p("\n== simple+multiple only (600): a fitting tool is offered")
for m, mode, th in (("12b","greedy","thinkoff"),("e4b","sampled","thinkon"),("e4b","greedy","thinkoff"),("e2b","greedy","thinkoff")):
    d = pooled(m, mode, th, TASKS[:2]); p(f"{m} {mode} {th}: {right(d)}/600")
pr("12b greedy off vs e4b Google's on, simple+multiple", pooled("12b","greedy","thinkoff",TASKS[:2]), pooled("e4b","sampled","thinkon",TASKS[:2]))
pr("12b greedy off vs e4b greedy off, simple+multiple", pooled("12b","greedy","thinkoff",TASKS[:2]), pooled("e4b","greedy","thinkoff",TASKS[:2]))
pr("e4b Google's on vs e2b greedy off, all 840", pooled("e4b","sampled","thinkon"), pooled("e2b","greedy","thinkoff"))
ratio("e4b Google's on / e2b greedy off", pooled("e4b","sampled","thinkon"), pooled("e2b","greedy","thinkoff"))

pr("e4b vs e2b greedy thinking off, all 840", pooled("e4b","greedy","thinkoff"), pooled("e2b","greedy","thinkoff"))
pr("12b vs e2b greedy thinking off, all 840", pooled("12b","greedy","thinkoff"), pooled("e2b","greedy","thinkoff"))
pr("e4b vs e2b greedy thinking on, all 840", pooled("e4b","greedy","thinkon"), pooled("e2b","greedy","thinkon"))
pr("12b vs e4b Google's thinking off, all 840", pooled("12b","sampled","thinkoff"), pooled("e4b","sampled","thinkoff"))
ratio("12b greedy off / e2b greedy off", pooled("12b","greedy","thinkoff"), pooled("e2b","greedy","thinkoff"))
ratio("e4b greedy off / e2b greedy off", pooled("e4b","greedy","thinkoff"), pooled("e2b","greedy","thinkoff"))
ratio("e2b Google's on / e2b greedy off", pooled("e2b","sampled","thinkon"), pooled("e2b","greedy","thinkoff"))

p("\n== October repeat (greedy thinking off, simple, no drafter)")
octf = {"e2b": "local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-fullctx-3050oem/bfcl_simple-e2b-none.jsonl",
        "e4b": "local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-fullctx-3050oem/bfcl_simple-e4b-none.jsonl",
        "12b": "local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-16-accuracy-12b-3050oem/bfcl_simple-none.jsonl"}
for m, f in octf.items():
    o = [json.loads(l) for l in open(os.path.join(OCT, f))]
    p(f"{m}: October {sum(r['right'] for r in o)}/400, now {right(load(m,'bfcl_simple','greedy','thinkoff'))}/400")
p("\n== differences and inverse ratios quoted in the text")
def lost(name, a, b): p(f"{name}: {right(a)-right(b)}")
lost("e4b lost, greedy off - Google's off", pooled("e4b","greedy","thinkoff"), pooled("e4b","sampled","thinkoff"))
lost("e2b lost, greedy off - Google's off", pooled("e2b","greedy","thinkoff"), pooled("e2b","sampled","thinkoff"))
lost("12b lost, greedy off - greedy on", pooled("12b","greedy","thinkoff"), pooled("12b","greedy","thinkon"))
for name, a, b in (("e2b greedy off as share of 12b greedy off", pooled("e2b","greedy","thinkoff"), pooled("12b","greedy","thinkoff")),
                   ("e2b greedy off as share of e4b greedy off", pooled("e2b","greedy","thinkoff"), pooled("e4b","greedy","thinkoff")),
                   ("e4b Google's on as share of 12b Google's on", pooled("e4b","sampled","thinkon"), pooled("12b","sampled","thinkon"))):
    p(f"{name}: {hours(a)/hours(b):.2f}x")
p("\n== totals")
p(f"cells {len(MODELS)*len(SETS)*len(TASKS)}, replies {sum(len(pooled(m,mo,th)) for m in MODELS for mo,th in SETS)}, hours {sum(hours(pooled(m,mo,th)) for m in MODELS for mo,th in SETS):.1f}")
p(f"cut at 8,192 across all runs: {sum(cut(pooled(m,mo,th)) for m in MODELS for mo,th in SETS)}")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "derived.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
