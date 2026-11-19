#!/usr/bin/env python3
"""Gemini 4 Neon at its four thinking levels on AIME 2025 I, computed from the saved replies (aime/).

Costs are the token counts the API returned (prompt, answer, and thinking) times $3 / $15 per 1M
(../../../pricing-2026-11-19.txt). The 2026-11-15 rows of the other Gemini 4 models, same problems,
prompt and 32,768-token limit, are read from docs/gemini-high-vs-low/evidence for a side table; those
ran on another day, so read them as context. Writes derived.txt beside this file.
"""

import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
HL = os.path.join(ROOT, "docs", "gemini-high-vs-low", "evidence")
LEVELS = ["low", "medium", "high", "xhigh"]


def rows(path):
    return {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}


def c(r, pin, pout):
    return (
        r.get("prompt_tokens", 0) * pin + (r.get("completion_tokens", 0) + r.get("reasoning_tokens", 0)) * pout
    ) / 1e6


def line(label, rs, pin, pout):
    right = sum(bool(r.get("right")) for r in rs)
    think = [r.get("reasoning_tokens", 0) for r in rs]
    secs = [r["seconds"] for r in rs if r.get("seconds")]
    cost = sum(c(r, pin, pout) for r in rs)
    per = f"${cost / right:.4f}" if right else "-"
    return (
        (
            f"  {label:22s} right {right:2d} of {len(rs)}  cut {sum(r.get('finish') == 'length' for r in rs)}"
            f"  errors {sum('error' in r for r in rs)}  thinking mean {statistics.mean(think):,.0f} max {max(think):,}"
            f"  median s {statistics.median(secs):.1f}  max s {max(secs):.1f}  cost ${cost:.4f}  per right answer {per}"
        ),
        cost,
        right,
    )


def main():
    out = []
    p = out.append
    neon = {lv: rows(os.path.join(HERE, "aime", f"neon-{lv}.jsonl")) for lv in LEVELS}
    ids = sorted(set.intersection(*[set(v) for v in neon.values()]))
    p(f"problems answered at every level: {len(ids)}")
    p("\n== Gemini 4 Neon, AIME 2025 I, temperature 0, 32,768-token limit")
    tot = 0
    base = {}
    for lv in LEVELS:
        s, cost, right = line(f"neon {lv}", [neon[lv][i] for i in ids], 3.0, 15.0)
        tot += cost
        base[lv] = cost
        p(s)
    p(f"  all four levels ${tot:.4f}")
    p("\n== each level against low, same problems (right only at that level / right only at low), cost ratio")
    for lv in LEVELS[1:]:
        up = sum(bool(neon[lv][i].get("right")) and not neon["low"][i].get("right") for i in ids)
        dn = sum(bool(neon["low"][i].get("right")) and not neon[lv][i].get("right") for i in ids)
        p(f"  {lv:7s} +{up} / -{dn}  cost {base[lv] / base['low']:.2f}x low")
    p("\n== per problem, right (R) or wrong (-) at low medium high xhigh")
    for i in ids:
        p(f"  {i}  " + " ".join("R" if neon[lv][i].get("right") else "-" for lv in LEVELS))

    p("\n== context: the other Gemini 4 models on the same 15 problems, 2026-11-15 (docs/gemini-high-vs-low)")
    for arm in ("low", "high"):
        models = {m["name"]: m for m in json.load(open(os.path.join(HL, f"models-{arm}.json")))}
        for n in ("hydrogen-4", "carbon-4", "argon-4"):
            r = rows(os.path.join(HL, f"aime-{arm}", n + ".jsonl"))
            m = models[n]
            s, *_ = line(f"{n} {m['reasoning_effort']}", [r[i] for i in ids if i in r], m["price_in"], m["price_out"])
            p(s)
    txt = "\n".join(out) + "\n"
    open(os.path.join(HERE, "derived.txt"), "w").write(txt)
    print(txt, end="")


if __name__ == "__main__":
    main()
