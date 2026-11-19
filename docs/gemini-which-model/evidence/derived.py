#!/usr/bin/env python3
"""Every figure the which-model article quotes, computed from saved replies.

AIME 2025 II (aime2/, 2026-11-19): eight models, lowest and high setting, run at the same time.
AIME 2025 I (2026-11-15, docs/gemini-high-vs-low/evidence/aime-{low,high}/; Neon from
gemini-api-google-neon4/.../2026-11-19-thinking-levels/aime/, where its lowest arm was sent as `low`,
and here as `none`). Pooled = I + II, 30 problems.
Costs: tokens returned times list price (models.json); December: Carbon and Argon at $4 / $20.
Statistics: Wilson 95% interval per score; exact two-sided McNemar (binomial on the discordant
problems) for paired comparisons. Writes derived.txt beside this file.
"""

import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HL = os.path.join(ROOT, "docs", "gemini-high-vs-low", "evidence")
NEON1 = os.path.join(ROOT, "gemini-api-google-neon4", "benchmarks", "runs", "2026-11-19-thinking-levels", "aime")
BASE = ["gemma-4-31b", "gemma-4-26b", "flash-lite-3.5", "flash-3.8", "hydrogen-4", "carbon-4", "argon-4", "neon-4"]
LABEL = {
    "gemma-4-31b": "Gemma 4 31B",
    "gemma-4-26b": "Gemma 4 26B",
    "flash-lite-3.5": "3.5 Flash-Lite",
    "flash-3.8": "3.8 Flash",
    "hydrogen-4": "Gemini 4 Hydrogen",
    "carbon-4": "Gemini 4 Carbon",
    "argon-4": "Gemini 4 Argon",
    "neon-4": "Gemini 4 Neon",
}
DEC = {"carbon-4": (4.0, 20.0), "argon-4": (4.0, 20.0)}


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def cost(r, pin, pout):
    return (
        (r.get("prompt_tokens") or 0) * pin
        + ((r.get("completion_tokens") or 0) + (r.get("reasoning_tokens") or 0)) * pout
    ) / 1e6


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def main():
    models = {m["name"]: m for m in json.load(open(os.path.join(HERE, "models.json")))}
    two = {n: load(os.path.join(HERE, "aime2", n + ".jsonl")) for n in models}
    one = {}
    for b in BASE:
        for arm in ("low", "high"):
            if b == "neon-4":
                one[f"{b}-{arm}"] = load(os.path.join(NEON1, f"neon-{'low' if arm == 'low' else 'high'}.jsonl"))
            else:
                one[f"{b}-{arm}"] = load(os.path.join(HL, f"aime-{arm}", b + ".jsonl"))
    out = []
    p = out.append

    def block(title, data, n_expect):
        p(f"== {title}")
        p(
            "   setting                     right  95% range      cut err  thinking mean  median s   cost   per problem  per right   Dec per problem"
        )
        for b in BASE:
            for arm in ("low", "high"):
                name = f"{b}-{arm}"
                m = models[name]
                rows = list(data[name].values())
                if not rows:
                    p(f"   {name:26s} NO ROWS")
                    continue
                k, n = sum(bool(r.get("right")) for r in rows), len(rows)
                lo, hi = wilson(k, n)
                c = sum(cost(r, m["price_in"], m["price_out"]) for r in rows)
                dc = sum(cost(r, *DEC[b]) for r in rows) if b in DEC else c
                think = statistics.mean(r.get("reasoning_tokens") or 0 for r in rows)
                med = statistics.median(r["seconds"] for r in rows if r.get("seconds"))
                per_right = f"${c / k:.4f}" if k else "-"
                flag = "" if n == n_expect else f"  (only {n} of {n_expect})"
                p(
                    f"   {name:18s} {m['reasoning_effort']:7s} {k:2d}/{n:2d}  [{lo:.2f}, {hi:.2f}]  {sum(r.get('finish') == 'length' for r in rows):3d} "
                    f"{sum('error' in r for r in rows):3d}  {think:12,.0f}  {med:8.1f}  ${c:6.3f}  ${c / n:.4f}     {per_right:9s}  ${dc / n:.4f}{flag}"
                )
        p("")

    block("AIME 2025 II, 2026-11-19, all models at the same time", two, 15)
    pooled = {n: {**one[n], **two[n]} for n in models}
    block("AIME 2025 I + II pooled (I from 2026-11-15, Neon's I from 2026-11-19)", pooled, 30)

    p("== each model, high against lowest setting, pooled 30 (right only at high / only at lowest, exact McNemar p)")
    for b in BASE:
        L, H = pooled[f"{b}-low"], pooled[f"{b}-high"]
        ids = sorted(set(L) & set(H))
        up = sum(bool(H[i].get("right")) and not L[i].get("right") for i in ids)
        dn = sum(bool(L[i].get("right")) and not H[i].get("right") for i in ids)
        ml, mh = models[f"{b}-low"], models[f"{b}-high"]
        cl = sum(cost(L[i], ml["price_in"], ml["price_out"]) for i in ids)
        ch = sum(cost(H[i], mh["price_in"], mh["price_out"]) for i in ids)
        ratio = f"{ch / cl:.2f}x" if cl else "free"
        p(f"   {LABEL[b]:18s} +{up} / -{dn}  p={mcnemar(up, dn):.3f}  cost {ratio}  ({len(ids)} paired)")
    p("")

    p("== pairs, pooled 30, same setting arm (right only for first / only for second, exact McNemar p)")
    pairs = [
        ("neon-4-high", "carbon-4-high"),
        ("neon-4-high", "argon-4-high"),
        ("neon-4-high", "hydrogen-4-high"),
        ("argon-4-high", "carbon-4-high"),
        ("argon-4-low", "neon-4-low"),
        ("argon-4-low", "carbon-4-low"),
        ("argon-4-low", "hydrogen-4-low"),
        ("argon-4-low", "flash-3.8-high"),
        ("flash-3.8-high", "flash-lite-3.5-high"),
        ("argon-4-low", "flash-lite-3.5-low"), ("argon-4-low", "flash-3.8-low"), ("argon-4-low", "flash-lite-3.5-high"),
        ("argon-4-low", "gemma-4-31b-high"),
        ("flash-3.8-high", "gemma-4-31b-high"),
        ("gemma-4-31b-high", "gemma-4-26b-high"),
    ]
    for a, b in pairs:
        A, B = pooled[a], pooled[b]
        ids = sorted(set(A) & set(B))
        x = sum(bool(A[i].get("right")) and not B[i].get("right") for i in ids)
        y = sum(bool(B[i].get("right")) and not A[i].get("right") for i in ids)
        ka, kb = sum(bool(A[i].get("right")) for i in ids), sum(bool(B[i].get("right")) for i in ids)
        p(f"   {a:20s} {ka:2d} vs {b:20s} {kb:2d}   +{x} / -{y}  p={mcnemar(x, y):.3f}  ({len(ids)} paired)")
    p("")

    p("== AIME II problems missed, per setting")
    for n in models:
        miss = sorted(i[-2:] for i, r in two[n].items() if not r.get("right"))
        p(f"   {n:20s} {', '.join(miss) if miss else 'none'}")

    txt = "\n".join(out) + "\n"
    open(os.path.join(HERE, "derived.txt"), "w").write(txt)
    print(txt, end="")


if __name__ == "__main__":
    main()
