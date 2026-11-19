#!/usr/bin/env python3
"""Per-model roadmap figures: cost per 1,000 requests, cost per right answer, thinking, speed, hard-math gain.

Sources, all rows on disk:
  short tasks  gemini-api-google-<slot>/benchmarks/runs/2026-11-14-accuracy-api/{gsm8k,bfcl_simple}.jsonl
               (Neon: 2026-11-19-accuracy-api, same settings, run alone five days later)
  hard math    docs/gemini-high-vs-low/evidence/aime-{low,high}/<name>.jsonl (AIME 2025 I, 2026-11-15)
               (Neon: gemini-api-google-neon4/benchmarks/runs/2026-11-19-thinking-levels/aime/neon-<level>.jsonl)
Prices per 1M tokens, output including thinking, from pricing-2026-11-14.txt and
gemini-api-google-neon4/pricing-2026-11-19.txt; December column applies Carbon's and Argon's 2026-12-01 $4 / $20
to the same token counts. Writes roadmap.txt beside this file.
"""

import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
HL = os.path.join(ROOT, "docs", "gemini-high-vs-low", "evidence")
NEON_AIME = os.path.join(ROOT, "gemini-api-google-neon4", "benchmarks", "runs", "2026-11-19-thinking-levels", "aime")

# slot, label, short-task run, AIME name, low setting, high setting, $in, $out, December ($in, $out) or None
M = [
    ("31b", "Gemma 4 31B", "2026-11-14-accuracy-api", "gemma-4-31b", "minimal", "high", 0, 0, None),
    ("26b", "Gemma 4 26B", "2026-11-14-accuracy-api", "gemma-4-26b", "minimal", "high", 0, 0, None),
    ("flashlite35", "3.5 Flash-Lite", "2026-11-14-accuracy-api", "flash-lite-3.5", "minimal", "high", 0.30, 2.50, None),
    ("flash38", "3.8 Flash", "2026-11-14-accuracy-api", "flash-3.8", "none", "high", 0.75, 3.75, None),
    ("hydrogen4", "Gemini 4 Hydrogen", "2026-11-14-accuracy-api", "hydrogen-4", "none", "xhigh", 1.50, 7.50, None),
    ("carbon4", "Gemini 4 Carbon", "2026-11-14-accuracy-api", "carbon-4", "none", "high", 2.00, 10.00, (4.00, 20.00)),
    ("argon4", "Gemini 4 Argon", "2026-11-14-accuracy-api", "argon-4", "none", "high", 2.00, 10.00, (4.00, 20.00)),
    ("neon4", "Gemini 4 Neon", "2026-11-19-accuracy-api", "neon", "low", "high", 3.00, 15.00, None),
]


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p)) if "id" in r} if os.path.exists(p) else {}


def c(r, pin, pout):
    return (
        (r.get("prompt_tokens") or 0) * pin
        + ((r.get("completion_tokens") or 0) + (r.get("reasoning_tokens") or 0)) * pout
    ) / 1e6


def aime(name, arm, level):
    if name == "neon":
        return load(os.path.join(NEON_AIME, f"neon-{level}.jsonl"))
    return load(os.path.join(HL, f"aime-{arm}", name + ".jsonl"))


def main():
    out = []
    p = out.append
    short = {}
    p(
        "== short tasks at each model's lowest thinking setting: score, $ per 1,000 requests (now / from 2026-12-01), $ per 1,000 right answers"
    )
    for slot, label, run, _, _, _, pin, pout, dec in M:
        for t in ("gsm8k", "bfcl_simple"):
            rows = load(os.path.join(ROOT, f"gemini-api-google-{slot}", "benchmarks", "runs", run, f"{t}.jsonl"))
            short[(slot, t)] = rows
            n = len(rows)
            right = sum(bool(r.get("right")) for r in rows.values())
            cost = sum(c(r, pin, pout) for r in rows.values())
            dcost = sum(c(r, *dec) for r in rows.values()) if dec else cost
            think = [r.get("reasoning_tokens") or 0 for r in rows.values()]
            sec = [
                r["seconds"] for r in rows.values() if r.get("seconds") and "error" not in r and "resent_after" not in r
            ]
            p(
                f"  {t:12s} {label:18s} {right / n:.4f}  per 1k ${cost / n * 1000:7.3f} / ${dcost / n * 1000:7.3f}"
                f"  per 1k right ${cost / right * 1000:7.3f}"
                f"  thinking on {sum(x > 0 for x in think) / n:4.0%} of requests, mean {statistics.mean(think):6.1f} tokens"
                f"  median {statistics.median(sec):5.2f} s{'  (ran alone, not comparable)' if slot == 'neon4' else ''}"
            )
        p("")

    p(
        "== hard math, AIME 2025 I (15 problems): low = lowest setting, high = top setting (Neon: low and its default high)"
    )
    for slot, label, _, name, lo, hi, pin, pout, dec in M:
        L, H = aime(name, "low", lo), aime(name, "high", hi)
        ids = sorted(set(L) & set(H))
        rl = sum(bool(L[i].get("right")) for i in ids)
        rh = sum(bool(H[i].get("right")) for i in ids)
        cl = sum(c(L[i], pin, pout) for i in ids)
        ch = sum(c(H[i], pin, pout) for i in ids)
        up = sum(bool(H[i].get("right")) and not L[i].get("right") for i in ids)
        dn = sum(bool(L[i].get("right")) and not H[i].get("right") for i in ids)
        dch = sum(c(H[i], *dec) for i in ids) if dec else ch
        dcl = sum(c(L[i], *dec) for i in ids) if dec else cl
        ms_l = statistics.median(L[i]["seconds"] for i in ids)
        ms_h = statistics.median(H[i]["seconds"] for i in ids)
        ratio = f"{ch / cl:.2f}x" if cl else "free"
        p(
            f"  {label:18s} low {lo:7s} {rl:2d}/15 ${cl / 15:.4f}/problem (Dec ${dcl / 15:.4f}) {ms_l:6.1f} s"
            f" | high {hi:5s} {rh:2d}/15 ${ch / 15:.4f}/problem (Dec ${dch / 15:.4f}) {ms_h:6.1f} s"
            f" | high vs low +{up}/-{dn}, cost {ratio}"
        )
    txt = "\n".join(out) + "\n"
    open(os.path.join(HERE, "roadmap.txt"), "w").write(txt)
    print(txt, end="")


if __name__ == "__main__":
    main()
