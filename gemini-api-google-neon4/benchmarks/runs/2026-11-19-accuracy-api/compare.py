"""Gemini 4 Neon (2026-11-19) beside the seven models of the 2026-11-14 run, computed from rows on disk.

Same scoring script, questions, token limits (GSM8K 2048, BFCL 512), concurrency 8, timeout and
retries, each model at the lowest thinking level it accepts. Neon ran alone five days later, so its
seconds per request are not comparable with the 11-14 models, which shared one connection seven ways.
Prices per 1M tokens, output including thinking: Neon $3 / $15 (../../../pricing-2026-11-19.txt),
the rest from docs/gemini-vs-gemma/evidence/pricing-2026-11-14.txt. Writes compare.txt beside this file.
"""

import json
import os
import random
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
OLD = "benchmarks/runs/2026-11-14-accuracy-api"

MODELS = [
    ("31b", "gemma-4-31b-it", "minimal", None, None, OLD),
    ("26b", "gemma-4-26b-a4b-it", "minimal", None, None, OLD),
    ("flashlite35", "gemini-3.5-flash-lite", "minimal", 0.30, 2.50, OLD),
    ("flash38", "gemini-3.8-flash", "none", 0.75, 3.75, OLD),
    ("hydrogen4", "gemini-4-hydrogen", "none", 1.50, 7.50, OLD),
    ("carbon4", "gemini-4-carbon", "none", 2.00, 10.00, OLD),
    ("argon4", "gemini-4-argon", "none", 2.00, 10.00, OLD),
    ("neon4", "gemini-4-neon", "none", 3.00, 15.00, "benchmarks/runs/2026-11-19-accuracy-api"),
]
TASKS = ("gsm8k", "bfcl_simple")
LATER = {"carbon4": (4.00, 20.00), "argon4": (4.00, 20.00)}  # from 2026-12-01


def load(slot, run, t):
    p = os.path.join(ROOT, f"gemini-api-google-{slot}", run, f"{t}.jsonl")
    return {r["id"]: r for r in map(json.loads, open(p)) if "id" in r}


def tokens(rows):
    return {
        k: sum(r.get(k) or 0 for r in rows.values()) for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
    }


def cost(tk, pin, pout):
    if pin is None:
        return None
    return (tk["prompt_tokens"] * pin + (tk["completion_tokens"] + tk["reasoning_tokens"]) * pout) / 1e6


def boot(d, n=2000):
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return b[int(n * 0.025)], b[int(n * 0.975) - 1]


def main():
    out = []
    p = out.append
    data = {(m[0], t): load(m[0], m[5], t) for m in MODELS for t in TASKS}

    p("== scores and cost, lowest thinking level each model accepts")
    costs = {}
    for slot, mid, eff, pin, pout, run in MODELS:
        g, b = data[(slot, "gsm8k")], data[(slot, "bfcl_simple")]
        vg = [int(bool(r.get("right"))) for r in g.values()]
        vb = [int(bool(r.get("right"))) for r in b.values()]
        tg, tb = tokens(g), tokens(b)
        c = None if pin is None else cost(tg, pin, pout) + cost(tb, pin, pout)
        costs[slot] = (tg, tb)
        lg, hg = boot(vg)
        lb, hb = boot(vb)
        p(
            f"  {mid:22s} {eff:8s} GSM8K {sum(vg):,}/{len(vg):,} {sum(vg) / len(vg):.4f} [{lg:.4f}, {hg:.4f}]"
            f"  BFCL {sum(vb)}/{len(vb)} {sum(vb) / len(vb):.4f} [{lb:.4f}, {hb:.4f}]"
            f"  errors {sum('error' in r for r in list(g.values()) + list(b.values()))}"
            f"  thinking/GSM8K problem {tg['reasoning_tokens'] / len(g):.1f}"
            f"  GSM8K problems with thinking {sum((r.get('reasoning_tokens') or 0) > 0 for r in g.values())}"
            f"  GSM8K ${'free' if pin is None else f'{cost(tg, pin, pout):.2f}'}"
            f"  both {'free tier' if c is None else f'${c:.2f}'}"
        )
    p("")

    p("== cost of both tasks at 2026-12-01 list prices (Carbon, Argon $4 / $20); arithmetic from today's token counts")
    for slot in ("hydrogen4", "carbon4", "argon4", "neon4"):
        pin, pout = LATER.get(slot, next((m[3], m[4]) for m in MODELS if m[0] == slot))
        tg, tb = costs[slot]
        p(f"  {slot:10s} ${cost(tg, pin, pout) + cost(tb, pin, pout):.2f}")
    p("")

    p("== Neon minus each model, paired by question (bootstrap 95% range of the difference, 2000 resamples, seed 0)")
    for t in TASKS:
        n = data[("neon4", t)]
        for slot, mid, *_ in MODELS[:-1]:
            o = data[(slot, t)]
            ids = sorted(set(n) & set(o))
            d = [int(bool(n[i].get("right"))) - int(bool(o[i].get("right"))) for i in ids]
            lo, hi = boot(d)
            only_n = sum(x == 1 for x in d)
            only_o = sum(x == -1 for x in d)
            verdict = "above zero" if lo > 0 else ("below zero" if hi < 0 else "includes zero")
            p(
                f"  {t:12s} vs {mid:22s} {sum(d) / len(d):+.4f} [{lo:+.4f}, {hi:+.4f}] {verdict}"
                f"  (Neon only right {only_n}, other only right {only_o}, {len(ids)} paired)"
            )
        p("")

    p("== Neon speed, alone on the connection, 8 requests at a time (not comparable with the 11-14 shared run)")
    for t in TASKS:
        rows = [r for r in data[("neon4", t)].values() if "error" not in r and r.get("seconds")]
        sec = [r["seconds"] for r in rows]
        p(f"  {t:12s} median {statistics.median(sec):.2f} s per request  ({len(rows):,} requests)")

    txt = "\n".join(out) + "\n"
    open(os.path.join(HERE, "compare.txt"), "w").write(txt)
    print(txt, end="")


if __name__ == "__main__":
    main()
