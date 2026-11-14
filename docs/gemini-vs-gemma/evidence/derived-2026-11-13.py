"""Every figure the two gemini-vs-gemma articles quote, computed from rows on disk.

Reads the five gemini-api-google-* run directories (2026-11-13-accuracy-gemini) and the Gemma 4
llama.cpp accuracy runs on the RTX 3050 OEM, and prices the Gemini runs from Google's pricing page
as saved in pricing-2026-11-14.txt (Standard tier, paid, per 1M tokens; output includes thinking).
Writes derived.txt beside this file. No figure in the articles is computed anywhere else.
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUN = "benchmarks/runs/2026-11-13-accuracy-gemini"

# (slot, model id, lowest thinking level, $ in / 1M, $ out / 1M) — pricing-2026-11-14.txt,
# Standard tab, paid tier, prices in force on 2026-11-14.
GEMINI = [
    ("flashlite35", "gemini-3.5-flash-lite", "minimal", 0.30, 2.50),
    ("flash38", "gemini-3.8-flash", "none", 0.75, 3.75),
    ("hydrogen4", "gemini-4-hydrogen", "none", 1.50, 7.50),
    ("carbon4", "gemini-4-carbon", "none", 2.00, 10.00),
    ("argon4", "gemini-4-argon", "none", 2.00, 10.00),
]
# Dated changes on the same page: Carbon and Argon $4 / $20 from 2026-12-01; 3.8 Flash $1.50 / $7.50
# from 2027-01-01.
LATER = {"carbon4": (4.00, 20.00, "2026-12-01"), "argon4": (4.00, 20.00, "2026-12-01"),
         "flash38": (1.50, 7.50, "2027-01-01")}
L3 = "local-llamacpp-3050oem-{}/benchmarks/runs/{}"
GEMMA = {
    "Gemma 4 E2B": ("2b-q4_0", "2026-10-14-accuracy-fullctx-3050oem", "gsm8k-e2b-none", "bfcl_simple-e2b-none"),
    "Gemma 4 E4B": ("2b-q4_0", "2026-10-14-accuracy-fullctx-3050oem", "gsm8k-e4b-none", "bfcl_simple-e4b-none"),
    "Gemma 4 12B": ("12b-q4_0", "2026-10-16-accuracy-12b-3050oem", "gsm8k-none", "bfcl_simple-none"),
    "Gemma 4 26B": ("26b-q4_0", "2026-10-18-accuracy-26b-3050oem", "gsm8k-none", "bfcl_simple-none"),
    "Gemma 4 31B": ("31b-q4_0", "2026-10-19-accuracy-31b-3050oem", "gsm8k-q4x", "bfcl_simple-none"),
}
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


def wall(rel):
    try:
        return json.loads(open(os.path.join(ROOT, rel)).read().strip().splitlines()[-1])["wall_s"]
    except (OSError, ValueError, KeyError, IndexError):
        return None


def tokens(rows):
    return {k: sum(r.get(k) or 0 for r in rows.values()) for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")}


def main():
    out = []
    p = out.append
    gemma = {g: {t: load(L3.format(s, f"{run}/{(gf, bf)[i]}.jsonl")) for i, t in enumerate(TASKS)}
             for g, (s, run, gf, bf) in GEMMA.items()}
    gem = {}
    for slot, mid, lowest, pin, pout in GEMINI:
        for arm in (lowest, "low"):
            for t in TASKS:
                rel = f"gemini-api-google-{slot}/{RUN}/{t}-{arm}"
                rows = load(rel + ".jsonl")
                if rows:
                    gem[(slot, arm, t)] = (rows, wall(rel + ".log"))

    for t in TASKS:
        p(f"== {t}: right / n, accuracy [95% range]")
        for g in GEMMA:
            rows = gemma[g][t]
            v = [int(bool(r.get("right"))) for r in rows.values()]
            lo, hi = boot(v)
            p(f"  {g:24s} thinking off  {sum(v)}/{len(v)}  {sum(v) / len(v):.4f} [{lo:.4f}, {hi:.4f}]  "
              f"cut {sum(r.get('finish') == 'length' for r in rows.values())}  out tokens {tokens(rows)['completion_tokens']:,}")
        for slot, mid, lowest, pin, pout in GEMINI:
            for arm in (lowest, "low"):
                if (slot, arm, t) not in gem:
                    p(f"  {mid:24s} {arm:12s} NOT RUN")
                    continue
                rows, w = gem[(slot, arm, t)]
                v = [int(bool(r.get("right"))) for r in rows.values()]
                lo, hi = boot(v)
                tk = tokens(rows)
                cost = (tk["prompt_tokens"] * pin + (tk["completion_tokens"] + tk["reasoning_tokens"]) * pout) / 1e6
                p(f"  {mid:24s} {arm:12s} {sum(v)}/{len(v)}  {sum(v) / len(v):.4f} [{lo:.4f}, {hi:.4f}]  "
                  f"errors {sum('error' in r for r in rows.values())}  cut {sum(r.get('finish') == 'length' for r in rows.values())}  "
                  f"in {tk['prompt_tokens']:,}  visible out {tk['completion_tokens']:,}  thinking {tk['reasoning_tokens']:,}  "
                  f"cost ${cost:.2f}  per 1,000 right ${cost / sum(v) * 1000:.3f}  wall {w} s")
                if slot in LATER:
                    li, lo2, d = LATER[slot]
                    c2 = (tk["prompt_tokens"] * li + (tk["completion_tokens"] + tk["reasoning_tokens"]) * lo2) / 1e6
                    p(f"  {'':24s} {'':12s} at the {d} price: cost ${c2:.2f}  per 1,000 right ${c2 / sum(v) * 1000:.3f}")
        ref = gemma["Gemma 4 31B"][t]
        p(f"  paired against Gemma 4 31B on the same {t} items: difference [95% range]  (+Gemini right where 31B wrong / -)")
        for (slot, arm, tt), (rows, _) in gem.items():
            if tt != t:
                continue
            ids = sorted(set(rows) & set(ref))
            d = [int(bool(rows[i].get("right"))) - int(bool(ref[i].get("right"))) for i in ids]
            lo, hi = boot(d)
            verdict = "above zero" if lo > 0 else "below zero" if hi < 0 else "includes zero"
            p(f"    {slot:12s} {arm:8s} {sum(d) / len(d):+.4f} [{lo:+.4f}, {hi:+.4f}] {verdict:13s} +{d.count(1)} / -{d.count(-1)} of {len(ids)}")
        p("")

    p("== thinking tokens per GSM8K item, lowest level vs low")
    for slot, mid, lowest, *_ in GEMINI:
        for arm in (lowest, "low"):
            if (slot, arm, "gsm8k") in gem:
                rows = gem[(slot, arm, "gsm8k")][0]
                tk = tokens(rows)
                p(f"  {mid:24s} {arm:8s} thinking per item {tk['reasoning_tokens'] / len(rows):7.1f}  "
                  f"visible per item {tk['completion_tokens'] / len(rows):6.1f}  items with any thinking "
                  f"{sum((r.get('reasoning_tokens') or 0) > 0 for r in rows.values())} of {len(rows):,}")
    p("")
    p("== GSM8K run cost split by token kind, at today's list price (arithmetic)")
    p(f"  {'model':24s} {'arm':8s} {'input':>8s} {'answer':>8s} {'thinking':>9s} {'total':>8s} {'without thinking':>17s} {'thinking share':>15s}")
    for slot, mid, lowest, pin, pout in GEMINI:
        for arm in (lowest, "low"):
            if (slot, arm, "gsm8k") not in gem:
                continue
            tk = tokens(gem[(slot, arm, "gsm8k")][0])
            ci, ca, ct = tk["prompt_tokens"] * pin / 1e6, tk["completion_tokens"] * pout / 1e6, tk["reasoning_tokens"] * pout / 1e6
            p(f"  {mid:24s} {arm:8s} ${ci:7.2f} ${ca:7.2f} ${ct:8.2f} ${ci + ca + ct:7.2f} ${ci + ca:16.2f} {ct / (ci + ca + ct):14.0%}")
    p("")
    p("== prompt tokens for the same GSM8K prompts (tokenizer difference)")
    for slot, mid, lowest, *_ in GEMINI:
        if (slot, lowest, "gsm8k") in gem:
            p(f"  {mid:24s} {tokens(gem[(slot, lowest, 'gsm8k')][0])['prompt_tokens']:,}")
    text = "\n".join(out) + "\n"
    open(os.path.join(HERE, "derived-2026-11-13.txt"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
