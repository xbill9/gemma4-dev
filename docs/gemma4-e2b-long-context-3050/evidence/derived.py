#!/usr/bin/env python3
"""derived.py -- every figure the E2B long-context article quotes, computed from the run's replies.
Run from this directory. Reads the run directory below and the 12B's 2026-10-17 recall file. Writes derived.txt."""
import json, os, random, re, statistics as st
from scipy.stats import binomtest, norm

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.normpath(os.path.join(HERE, "../../../local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-25-longctx-e2b-3050oem"))
B12 = os.path.normpath(os.path.join(HERE, "../../../local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-17-kvcache-12b-3050oem/recall.jsonl"))
src = open(f"{RUN}/longctx.py").read()
ns = {}
exec(src[src.index("PROJECTS ="):src.index("TRIALS, N")], ns)
PROJECTS, PEOPLE, CITIES, N = ns["PROJECTS"], ns["PEOPLE"], ns["CITIES"], 8
out = []
def p(*a): out.append(" ".join(str(x) for x in a))
def fp(x): return f"{x:.3g}" if x >= 0.001 else f"{x:.2e}"
def wilson(k, n, z=norm.ppf(0.975)):
    q = k / n; d = 1 + z * z / n; c = q + z * z / (2 * n); h = z * (q * (1 - q) / n + z * z / (4 * n * n)) ** 0.5
    return (c - h) / d, (c + h) / d
def mcnemar(b, c): return 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue


def replay(task, depth, trial):
    """Re-draw the trial's facts and the place of each sentence (fraction of the document) exactly as longctx.py did."""
    rng = random.Random(f"{task}-{depth}-{trial}")
    names = rng.sample(PROJECTS, N)
    if task == "recall":
        truth = {n: f"{rng.randrange(100000, 1000000)}" for n in names}
        sents = [("code", n) for n in names]
    else:
        people, cities = rng.sample(PEOPLE, N), rng.sample(CITIES, N)
        truth = dict(zip(names, cities))
        sents = [("lead", n) for n in names] + [("office", n) for n in names]
        rng.shuffle(sents)
    k = len(sents)
    cuts = sorted(rng.sample(range(1, k * 4), k))
    where = {}
    for i, s in enumerate(sents):
        where[s] = cuts[i] / (k * 4)
    return truth, where


def facts(r):
    pat = (r'"?(?:[Pp]roject )?([A-Z][a-z]+)"?\s*:\s*"?(\d{6})' if r["task"] == "recall"
           else r'"(?:[Pp]roject )?([A-Z][a-z]+)"\s*:\s*"(?:the )?([A-Z][a-z]+)')
    got = dict(re.findall(pat, r["reply"]))
    truth, where = replay(r["task"], r["depth"], r["trial"])
    res = []
    for n, v in truth.items():
        g = got.get(n)
        kind = "right" if g == v else ("missing" if g is None else
               ("other fact" if g in truth.values() else "invented"))
        pos = where[("code", n)] if r["task"] == "recall" else max(where[("lead", n)], where[("office", n)])
        gap = None if r["task"] == "recall" else abs(where[("lead", n)] - where[("office", n)])
        res.append({"name": n, "kind": kind, "pos": pos, "gap": gap})
    return res


R = [json.loads(l) for l in open(f"{RUN}/longctx.jsonl")]
for r in R:
    r["facts"] = facts(r)
    assert sum(f["kind"] == "right" for f in r["facts"]) == r["right"]
ARMS = [a for a in ["f16-draft", "f16-none", "q8-draft", "q4-draft", "f16-sampled-off", "f16-sampled-on"] if any(r["arm"] == a for r in R)]
DEPTHS = sorted({r["depth"] for r in R})
SHARED = [d for d in DEPTHS if d <= 98304]
idx = {(r["arm"], r["task"], r["depth"], r["trial"]): r for r in R}
p(f"run {RUN}")
p(f"replies {len(R)}; arms {', '.join(ARMS)}; depths {DEPTHS}")
p(f"prompt tokens actually sent, by depth (min-max): " + "; ".join(
    f"{d}: {min(r['prompt_n'] for r in R if r['depth']==d)}-{max(r['prompt_n'] for r in R if r['depth']==d)}" for d in DEPTHS))

p("\n== Facts right of 80 per cell (10 trials x 8 facts), Wilson 95% range")
for task in ("recall", "link"):
    p(f"-- {task}")
    for a in ARMS:
        row = []
        for d in DEPTHS:
            rs = [r for r in R if r["arm"] == a and r["task"] == task and r["depth"] == d]
            if not rs: row.append("-"); continue
            k = sum(r["right"] for r in rs); lo, hi = wilson(k, 8 * len(rs))
            row.append(f"{d}: {k}/{8*len(rs)} ({lo:.2f}-{hi:.2f})")
        p(f"  {a}: " + "; ".join(row))
    for a in ARMS:
        rs = [r for r in R if r["arm"] == a and r["task"] == task and r["depth"] in SHARED]
        k = sum(r["right"] for r in rs); n = 8 * len(rs); lo, hi = wilson(k, n)
        p(f"  {a} {task} 16,384-98,304: {k} of {n} = {k/n:.3f} ({lo:.3f}-{hi:.3f})")

p("\n== Paired tests against f16-draft (the rig's serving setting), depths 16,384-98,304, both tasks")
p("   per fact: +only this arm right / -only f16-draft right, exact McNemar; per trial: trials this arm found more / fewer facts, exact sign test")
tests = []
def compare(a, base="f16-draft", depths=SHARED, tasks=("recall", "link"), label=None):
    b = c = up = down = 0
    for t in tasks:
        for d in depths:
            for i in range(10):
                x, y = idx.get((a, t, d, i)), idx.get((base, t, d, i))
                if not x or not y: continue
                for fx, fy in zip(x["facts"], y["facts"]):
                    b += fx["kind"] == "right" and fy["kind"] != "right"; c += fy["kind"] == "right" and fx["kind"] != "right"
                up += x["right"] > y["right"]; down += x["right"] < y["right"]
    pf, pt = mcnemar(b, c), mcnemar(up, down)
    lab = label or f"{a} vs {base}"
    tests.append((lab, pt))
    p(f"  {lab}: facts +{b} / -{c} p={fp(pf)}; trials +{up} / -{down} p={fp(pt)}")
for a in ARMS:
    if a != "f16-draft": compare(a)
for a in ("q8-draft", "q4-draft"):
    compare(a, depths=DEPTHS, label=f"{a} vs f16-draft, all five depths")
compare("f16-sampled-on", base="f16-sampled-off", label="thinking on vs off, both sampled")
if "f16-none" in ARMS:
    compare("f16-sampled-off", base="f16-none", depths=DEPTHS, label="sampled vs temperature 0, both no drafter, thinking off, all depths")
    compare("f16-sampled-on", base="f16-none", label="sampled+thinking vs temperature 0, both no drafter")
    compare("f16-none", depths=DEPTHS, label="no drafter vs drafter, temperature 0, all depths")
for task in ("recall", "link"):
    compare("f16-sampled-on", tasks=(task,), label=f"sampled+thinking vs f16-draft, {task} only")
p("\n  Holm correction over the trial-level tests above:")
m = len(tests); order = sorted(range(m), key=lambda i: tests[i][1]); run = 0.0
for rank, i in enumerate(order):
    adj = min(1.0, max(run, (m - rank) * tests[i][1])); run = adj
    p(f"    {tests[i][0]}: p={fp(tests[i][1])} Holm {fp(adj)} {'holds' if adj < 0.05 else 'does not hold'}")

p("\n== What the missed facts were: per fact, right / other fact from the same document / invented / missing")
for a in ARMS:
    for task in ("recall", "link"):
        fs = [f for r in R if r["arm"] == a and r["task"] == task for f in r["facts"]]
        cnt = {k: sum(f["kind"] == k for f in fs) for k in ("right", "other fact", "invented", "missing")}
        p(f"  {a} {task} (all depths, {len(fs)} facts): " + ", ".join(f"{k} {v}" for k, v in cnt.items()))

p("\n== Replies that ran to the reply limit (256 tokens; 16,384 with thinking)")
def names_in(r):
    truth, _ = replay(r["task"], r["depth"], r["trial"])
    return sum(n in r["reply"] for n in truth)
def repeats(text):
    from collections import Counter
    c = Counter(l.strip() for l in text.splitlines() if l.strip().startswith('"'))
    return max(c.values()) if c else 0
for a in ARMS:
    rs = [r for r in R if r["arm"] == a]
    cut = [r for r in rs if r["finish"] == "length"]
    p(f"  {a}: {len(cut)} of {len(rs)} cut off (by depth " + ", ".join(f"{d}: {sum(r['depth']==d for r in cut)}" for d in DEPTHS)
      + f"); facts right in them {sum(r['right'] for r in cut)}; naming none of the trial's 8 projects {sum(names_in(r)==0 for r in cut)}; "
      f"one key line 5+ times {sum(repeats(r['reply'])>=5 for r in cut)}")
ex = next(r for r in R if r["arm"] == "f16-draft" and r["finish"] == "length" and repeats(r["reply"]) >= 5)
p("  example, f16-draft " + f"{ex['task']} {ex['depth']} trial {ex['trial']}: " + repr(ex["reply"][:420]))
p("  recall by third of the document, replies that ended on their own vs cut off, 16,384-98,304:")
for a in ("f16-draft", "f16-sampled-off"):
    for fin in ("stop", "length"):
        fs = [f for r in R if r["arm"] == a and r["task"] == "recall" and r["finish"] == fin and r["depth"] in SHARED for f in r["facts"]]
        th = [f"{sum(f['kind']=='right' for f in fs if lo<=f['pos']<hi)}/{sum(1 for f in fs if lo<=f['pos']<hi)}" for lo, hi in ((0,1/3),(1/3,2/3),(2/3,1.1))]
        p(f"    {a} {'ended' if fin=='stop' else 'cut off'}: first {th[0]}, middle {th[1]}, last {th[2]}")

p("\n== Where the fact sat in the document (recall, by third), facts right / facts")
for a in ARMS:
    row = []
    for lo, hi, lab in ((0, 1/3, "first third"), (1/3, 2/3, "middle third"), (2/3, 1.01, "last third")):
        fs = [f for r in R if r["arm"] == a and r["task"] == "recall" and r["depth"] in SHARED for f in r["facts"] if lo <= f["pos"] < hi]
        row.append(f"{lab} {sum(f['kind']=='right' for f in fs)}/{len(fs)}")
    p(f"  {a} (16,384-98,304): " + "; ".join(row))
p("   link, by distance between the two sentences (fraction of the document), facts right / facts")
for a in ARMS:
    row = []
    for lo, hi, lab in ((0, 0.25, "under a quarter apart"), (0.25, 0.5, "a quarter to half"), (0.5, 1.01, "over half")):
        fs = [f for r in R if r["arm"] == a and r["task"] == "link" and r["depth"] in SHARED for f in r["facts"] if lo <= f["gap"] < hi]
        row.append(f"{lab} {sum(f['kind']=='right' for f in fs)}/{len(fs)}")
    p(f"  {a}: " + "; ".join(row))

p("\n== Time: reading the prompt (prompt cache off), generating, whole request; medians over the 20 requests per depth")
for a in ARMS:
    for d in DEPTHS:
        rs = [r for r in R if r["arm"] == a and r["depth"] == d]
        if not rs: continue
        p(f"  {a} {d}: read {st.median(r['prompt_ms'] for r in rs)/1000:.1f} s at {st.median(r['prompt_tps'] for r in rs):.0f} tok/s; "
          f"generate {st.median(r['decode_tps'] for r in rs):.1f} tok/s; reply {st.median(r['completion_n'] for r in rs):.0f} tok; "
          f"request {st.median(r['wall_s'] for r in rs):.1f} s")
for a in ("f16-sampled-off", "f16-sampled-on"):
    rs = [r for r in R if r["arm"] == a and r["depth"] in SHARED]
    p(f"  {a} 16,384-98,304: reply tokens median {st.median(r['completion_n'] for r in rs):.0f}, max {max(r['completion_n'] for r in rs)}, "
      f"cut off {sum(r['finish']=='length' for r in rs)}; thinking text in {sum(r['reasoning_chars']>0 for r in rs)} of {len(rs)}")
for d in SHARED:
    on = st.median(r["wall_s"] for r in R if r["arm"] == "f16-sampled-on" and r["depth"] == d)
    off = st.median(r["wall_s"] for r in R if r["arm"] == "f16-sampled-off" and r["depth"] == d)
    p(f"  thinking adds at {d}: {on - off:.1f} s per request ({on:.1f} against {off:.1f})")
p("  GPU memory and power at each arm's last request (temp, clock, memory, power): "
  + "; ".join(f"{a}: {[r for r in R if r['arm']==a][-1]['gpu']}" for a in ARMS))
for a in ARMS:
    srv = next((l.strip() for l in open(f"{RUN}/longctx.out") if f"server up {a}:" in l), None)
    p(f"  {a} server: {srv}")

p("\n== 12B recall, 2026-10-17 (same document, same sentence design, its own seeds; greedy, prompt cache on, 8 codes x 10 trials)")
B = [json.loads(l) for l in open(B12)]
for arm in sorted({r["arm"] for r in B}):
    p(f"  12B {arm}: " + "; ".join(f"{d}: {sum(r['right'] for r in B if r['arm']==arm and r['depth']==d)}/{8*sum(1 for r in B if r['arm']==arm and r['depth']==d)}"
                                    for d in sorted({r['depth'] for r in B if r['arm'] == arm})))
open(os.path.join(HERE, "derived.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))

ex = []
rng = random.Random("link-32768-0"); names = rng.sample(PROJECTS, N); people = rng.sample(PEOPLE, N); cities = rng.sample(CITIES, N)
ex.append(f"example link trial (32,768, trial 0): " + "; ".join(f"Project {n} is led by {pp}. {pp} works from the {c} office." for n, pp, c in list(zip(names, people, cities))[:2]))
t, _ = replay("recall", 32768, 0)
ex.append("example recall trial (32,768, trial 0): " + "; ".join(f"The access code for project {n} is {v}." for n, v in list(t.items())[:3]))
for a, task in (("f16-draft", "recall"), ("f16-draft", "link"), ("f16-sampled-on", "link")):
    r = idx[(a, task, 32768, 0)]
    ex.append(f"reply {a} {task} 32,768 trial 0 ({r['right']} of 8): {r['reply']!r}")
open(os.path.join(HERE, "derived.txt"), "a").write("\n== Examples\n" + "\n".join(ex) + "\n")
print("\n".join(ex))
