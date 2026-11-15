#!/usr/bin/env python3
"""Every figure the high-vs-low paper quotes, computed from the saved replies (aime-high/, aime-low/).

Costs are token counts the API returned (prompt, answer, and thinking = total - prompt - answer)
times the list prices in models-*.json (pricing page saved 2026-11-14), plus the same tokens at
the 2026-12-01 prices for Carbon and Argon ($4 / $20); the latter is arithmetic.
"""

import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["gemma-4-31b", "gemma-4-26b", "flash-lite-3.5", "flash-3.8", "hydrogen-4", "carbon-4", "argon-4"]
DEC = {"carbon-4": (4.0, 20.0), "argon-4": (4.0, 20.0)}


def load(arm):
    models = {m["name"]: m for m in json.load(open(os.path.join(HERE, f"models-{arm}.json")))}
    rows = {}
    for name in ORDER:
        p = os.path.join(HERE, f"aime-{arm}", name + ".jsonl")
        rows[name] = {r["id"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}
    return models, rows


def toks(r):
    return r.get("prompt_tokens", 0), r.get("completion_tokens", 0), r.get("reasoning_tokens", 0)


def cost(rs, pin, pout):
    return sum((toks(r)[0] * pin + (toks(r)[1] + toks(r)[2]) * pout) / 1e6 for r in rs)


def main():
    arms = {a: load(a) for a in ("low", "high")}
    ids = sorted(set.intersection(*[set(rows[n]) for a in arms for n, rows in [(n, arms[a][1]) for n in ORDER]]))
    print(f"questions answered by every model in both arms: {len(ids)}")
    total = {}
    for a in ("low", "high"):
        models, rows = arms[a]
        print(f"\n== arm {a}")
        total[a] = 0
        for n in ORDER:
            rs = [rows[n][i] for i in ids]
            m = models[n]
            c = cost(rs, m["price_in"], m["price_out"])
            total[a] += c
            right = sum(r["right"] for r in rs)
            think = [toks(r)[2] for r in rs]
            ans = [toks(r)[1] for r in rs]
            secs = [r["seconds"] for r in rs]
            line = (
                f"  {n:15s} {m['reasoning_effort']:7s} right {right:2d} of {len(rs)}  cut {sum(r.get('finish') == 'length' for r in rs)}"
                f"  errors {sum('error' in r for r in rs)}  thinking total {sum(think):,} mean {statistics.mean(think):,.0f} max {max(think):,}"
                f"  answer mean {statistics.mean(ans):,.0f}  median s {statistics.median(secs):.1f}  max s {max(secs):.1f}"
                f"  cost ${c:.4f}"
            )
            if right:
                line += f"  per right answer ${c / right:.4f}"
            if n in DEC:
                line += f"  at 2026-12-01 prices ${cost(rs, *DEC[n]):.4f}"
            print(line)
        print(f"  arm total ${total[a]:.4f}")

    print("\n== high against low, same questions (right only at high / right only at low)")
    for n in ORDER:
        lo, hi = arms["low"][1][n], arms["high"][1][n]
        up = sum(hi[i]["right"] and not lo[i]["right"] for i in ids)
        down = sum(lo[i]["right"] and not hi[i]["right"] for i in ids)
        cl = cost([lo[i] for i in ids], arms["low"][0][n]["price_in"], arms["low"][0][n]["price_out"])
        ch = cost([hi[i] for i in ids], arms["high"][0][n]["price_in"], arms["high"][0][n]["price_out"])
        tl = sum(toks(lo[i])[2] for i in ids)
        th = sum(toks(hi[i])[2] for i in ids)
        ratio = f"{ch / cl:.2f}x" if cl else "free"
        tratio = f"{th / tl:.1f}x" if tl else f"{th:,} vs 0"
        print(f"  {n:15s} +{up} / -{down}   cost {ratio}   thinking {tratio}")

    print(f"\n== both arms ${total['low'] + total['high']:.4f} (low ${total['low']:.4f}, high ${total['high']:.4f})")

    print("\n== per question, right at high / low (H/L), every model")
    for i in ids:
        cells = " ".join(
            ("H" if arms["high"][1][n][i]["right"] else "-") + ("L" if arms["low"][1][n][i]["right"] else "-")
            for n in ORDER
        )
        print(f"  {i}  {cells}")
    print("  columns: " + " ".join(ORDER))


if __name__ == "__main__":
    main()
