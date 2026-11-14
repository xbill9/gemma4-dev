"""GSM8K and BFCL simple for the five gemini-api-google-* rigs, beside the Gemma 4 family on
llama.cpp with the same harness (jev-tpu-v5e1/gen_eval.py), prompts, token limits and scoring.

Per rig and thinking arm: right/n, accuracy with a 95% bootstrap range, errors, cut at the token
limit, wall time, and tokens (prompt, visible output, thinking). Then each Gemini arm against
Gemma 4 31B, paired by question. Identical copies of this file sit in each gemini rig's run
directory; every number below is computed here, from the rows on disk.

Gemma references are the drafter-free arms with thinking off (--reasoning off), except the 31B's
GSM8K, which exists only with its drafter; the drafter left GSM8K and BFCL unchanged on the 12B and
26B (their REPORT.md files)."""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RUN = "benchmarks/runs/2026-11-13-accuracy-gemini"
GEMINI = [  # rig slot 4, lowest thinking level accepted
    ("flashlite35", "minimal"),
    ("flash38", "none"),
    ("hydrogen4", "none"),
    ("carbon4", "none"),
    ("argon4", "none"),
]
L3 = "local-llamacpp-3050oem-{}/benchmarks/runs/{}"
GEMMA = {  # label: (gsm8k file, bfcl file)
    "gemma4 E2B": (L3.format("2b-q4_0", "2026-10-14-accuracy-fullctx-3050oem/gsm8k-e2b-none.jsonl"),
                   L3.format("2b-q4_0", "2026-10-14-accuracy-fullctx-3050oem/bfcl_simple-e2b-none.jsonl")),
    "gemma4 E4B": (L3.format("2b-q4_0", "2026-10-14-accuracy-fullctx-3050oem/gsm8k-e4b-none.jsonl"),
                   L3.format("2b-q4_0", "2026-10-14-accuracy-fullctx-3050oem/bfcl_simple-e4b-none.jsonl")),
    "gemma4 12B": (L3.format("12b-q4_0", "2026-10-16-accuracy-12b-3050oem/gsm8k-none.jsonl"),
                   L3.format("12b-q4_0", "2026-10-16-accuracy-12b-3050oem/bfcl_simple-none.jsonl")),
    "gemma4 26B": (L3.format("26b-q4_0", "2026-10-18-accuracy-26b-3050oem/gsm8k-none.jsonl"),
                   L3.format("26b-q4_0", "2026-10-18-accuracy-26b-3050oem/bfcl_simple-none.jsonl")),
    "gemma4 31B": (L3.format("31b-q4_0", "2026-10-19-accuracy-31b-3050oem/gsm8k-q4x.jsonl"),
                   L3.format("31b-q4_0", "2026-10-19-accuracy-31b-3050oem/bfcl_simple-none.jsonl")),
}
TASKS = ("gsm8k", "bfcl_simple")


def load(f):
    f = os.path.join(ROOT, f)
    return {r["id"]: r for r in map(json.loads, open(f)) if "id" in r} if os.path.exists(f) else {}


def boot(d, n=2000):
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return b[int(n * 0.025)], b[int(n * 0.975) - 1]


def wall(f):
    try:
        return json.loads(open(os.path.join(ROOT, f)).read().strip().splitlines()[-1])["wall_s"]
    except (OSError, ValueError, KeyError, IndexError):
        return None


def line(label, rows, logf=None):
    v = [int(bool(r.get("right"))) for r in rows.values()]
    lo, hi = boot(v)
    tok = {k: sum(r.get(k) or 0 for r in rows.values()) for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")}
    err = sum("error" in r for r in rows.values())
    cut = sum(r.get("finish") == "length" for r in rows.values())
    w = wall(logf) if logf else None
    return (f"  {label:26s} {sum(v):5d}/{len(v)}  {sum(v) / len(v):.4f} [{lo:.4f}, {hi:.4f}]  errors {err}  cut {cut}"
            f"  wall {w if w is not None else '-'} s  tokens in {tok['prompt_tokens']:,} out {tok['completion_tokens']:,}"
            f" thinking {tok['reasoning_tokens']:,}")


def main():
    arms = {}
    for slot, lowest in GEMINI:
        for arm in (lowest, "low"):
            for t in TASKS:
                f = f"gemini-api-google-{slot}/{RUN}/{t}-{arm}.jsonl"
                rows = load(f)
                if rows:
                    arms[(slot, arm, t)] = (rows, f[:-6] + ".log")
    gemma = {(g, t): load(fs[i]) for g, fs in GEMMA.items() for i, t in enumerate(TASKS)}
    for t in TASKS:
        print(f"{t}")
        for g in GEMMA:
            if gemma[(g, t)]:
                print(line(g + " (3050, llama.cpp)", gemma[(g, t)]))
        for (slot, arm, tt), (rows, logf) in arms.items():
            if tt == t:
                print(line(f"{slot} {arm}", rows, logf))
        ref = gemma[("gemma4 31B", t)]
        print(f"  paired against gemma4 31B ({t}):")
        for (slot, arm, tt), (rows, _) in arms.items():
            if tt != t or not ref:
                continue
            ids = sorted(set(rows) & set(ref))
            d = [int(bool(rows[i].get("right"))) - int(bool(ref[i].get("right"))) for i in ids]
            lo, hi = boot(d)
            print(f"    {slot} {arm:8s} {sum(d) / len(d):+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                  f"+{d.count(1)} / -{d.count(-1)} of {len(ids)}")
        print()


if __name__ == "__main__":
    main()
