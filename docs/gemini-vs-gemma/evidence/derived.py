"""Every figure the gemini-vs-gemma articles quote, computed from rows on disk.

The comparison is one run, 2026-11-14-accuracy-api: seven models through the same Gemini API
endpoint, started together, with the same scoring script, questions, token limits, concurrency,
timeout and retries, each at the lowest thinking level it accepts. Prices are Google's list prices
as saved in pricing-2026-11-14.txt (paid tier, Standard, per 1M tokens; output includes thinking);
Gemma 4 is free-tier only. Writes derived.txt beside this file.
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUN = "benchmarks/runs/2026-11-14-accuracy-api"

# (rig slot, model id, thinking level sent, $ in / 1M, $ out / 1M; None = free tier only)
MODELS = [
    ("31b", "gemma-4-31b-it", "minimal", None, None),
    ("26b", "gemma-4-26b-a4b-it", "minimal", None, None),
    ("flashlite35", "gemini-3.5-flash-lite", "minimal", 0.30, 2.50),
    ("flash38", "gemini-3.8-flash", "none", 0.75, 3.75),
    ("hydrogen4", "gemini-4-hydrogen", "none", 1.50, 7.50),
    ("carbon4", "gemini-4-carbon", "none", 2.00, 10.00),
    ("argon4", "gemini-4-argon", "none", 2.00, 10.00),
]
TASKS = ("gsm8k", "bfcl_simple")


def load(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return {}
    return {r["id"]: r for r in map(json.loads, open(p)) if "id" in r}


def boot(d, n=2000):
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return b[int(n * 0.025)], b[int(n * 0.975) - 1]


def tokens(rows):
    return {k: sum(r.get(k) or 0 for r in rows.values()) for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")}


def cost(tk, pin, pout):
    if pin is None:
        return None
    return (tk["prompt_tokens"] * pin + (tk["completion_tokens"] + tk["reasoning_tokens"]) * pout) / 1e6


def main():
    out = []
    p = out.append
    data = {(s, t): load(f"gemini-api-google-{s}/{RUN}/{t}.jsonl") for s, *_ in MODELS for t in TASKS}

    p("== one table: same endpoint, same settings, lowest thinking level each model accepts")
    for slot, mid, eff, pin, pout in MODELS:
        g, b = data[(slot, "gsm8k")], data[(slot, "bfcl_simple")]
        if not g or not b:
            p(f"  {mid:22s} {eff:8s} NOT COMPLETE")
            continue
        vg = [int(bool(r.get("right"))) for r in g.values()]
        vb = [int(bool(r.get("right"))) for r in b.values()]
        lg, hg = boot(vg)
        lb, hb = boot(vb)
        tg, tb = tokens(g), tokens(b)
        cg, cb = cost(tg, pin, pout), cost(tb, pin, pout)
        c = "free tier" if cg is None else f"${cg + cb:.2f}"
        p(f"  {mid:22s} {eff:8s} GSM8K {sum(vg):,} of {len(vg):,} {sum(vg) / len(vg):.4f} [{lg:.4f}, {hg:.4f}]"
          f"  BFCL {sum(vb)} of {len(vb)} {sum(vb) / len(vb):.4f} [{lb:.4f}, {hb:.4f}]"
          f"  thinking per GSM8K problem {tg['reasoning_tokens'] / len(g):.1f}  both tasks cost {c}")
    p("")

    p("== details per task")
    for t in TASKS:
        for slot, mid, eff, pin, pout in MODELS:
            rows = data[(slot, t)]
            if not rows:
                continue
            tk = tokens(rows)
            c = cost(tk, pin, pout)
            share = f"  thinking share {tk['reasoning_tokens'] * pout / 1e6 / c:.0%}" if c else ""
            p(f"  {t:12s} {mid:22s} errors {sum('error' in r for r in rows.values())}"
              f"  cut {sum(r.get('finish') == 'length' for r in rows.values())}"
              f"  in {tk['prompt_tokens']:,}  answer {tk['completion_tokens']:,}  thinking {tk['reasoning_tokens']:,}"
              f"  problems with thinking {sum((r.get('reasoning_tokens') or 0) > 0 for r in rows.values())} of {len(rows):,}"
              f"  cost {'free tier' if c is None else f'${c:.2f}'}{share}")
        p("")

    p("== each model against Gemma 4 31B, same problems (difference, 95% range, +right where 31B wrong / -)")
    for t in TASKS:
        ref = data[("31b", t)]
        for slot, mid, *_ in MODELS[1:]:
            rows = data[(slot, t)]
            if not rows or not ref:
                continue
            ids = sorted(set(rows) & set(ref))
            d = [int(bool(rows[i].get("right"))) - int(bool(ref[i].get("right"))) for i in ids]
            lo, hi = boot(d)
            verdict = "ahead" if lo > 0 else "behind" if hi < 0 else "level"
            p(f"  {t:12s} {mid:22s} {sum(d) / len(d):+.4f} [{lo:+.4f}, {hi:+.4f}] {verdict:6s} +{d.count(1)} / -{d.count(-1)} of {len(ids):,}")
        p("")
    text = "\n".join(out) + "\n"
    open(os.path.join(HERE, "derived.txt"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
