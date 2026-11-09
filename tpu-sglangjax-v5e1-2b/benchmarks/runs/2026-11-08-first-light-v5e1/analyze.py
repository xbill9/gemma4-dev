"""Tables of SGLang-JAX first light on v5e-1 (2026-11-08), against the v5e-1 ladder's arms (2026-11-06/07).

    python3 analyze.py          # writes SUMMARY.txt next to this file and prints it

Same method as the ladder's analyze.py (read, not imported): one request is the geometric mean over six
prompts of each prompt's median; sweeps are sweep.py's output tok/s per level; GSM8K is paired by question
with a 95% paired bootstrap range (2,000 resamples, seed 0).
"""

import json
import math
import os
import random
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
LAD = os.path.join(ROOT, "tpu-vllm-v5e1-2b-q4w4a16emb4", "benchmarks", "runs", "2026-11-06-ladder-v5e1")
LAD_VMS = ("tpu-vllm-v5e1-2b-q4w4a16emb4", "ladder-v5e1-torchtpu", "ladder-v5e1-torchtpurs")
SHAPES = ("in512-out128", "in128-out512")
REF = ("vllm", "vllm-text", "jax-bf16", "jaxrust", "jax")


def gm(v):
    return math.exp(sum(map(math.log, v)) / len(v)) if v and all(x > 0 for x in v) else None


def lad(rel):
    for v in LAD_VMS:
        p = os.path.join(LAD, v, rel)
        if os.path.exists(p) and os.path.exists(os.path.join(LAD, v, rel.split("/")[0], "DONE")):
            return p
    return None


def rows():
    rs = [json.loads(x) for x in open(os.path.join(HERE, "results.jsonl"))]
    for v in LAD_VMS:
        p = os.path.join(LAD, v, "results.jsonl")
        if os.path.exists(p):
            rs += [json.loads(x) for x in open(p)]
    by = defaultdict(list)
    for r in rs:
        by[r["arm"]].append(r)
    return by


def per_prompt(rs, kind, key):
    d = defaultdict(list)
    for r in rs:
        if r["kind"] == kind and r.get(key) is not None:
            d[r["prompt"]].append(r[key])
    return {p: statistics.median(x) for p, x in d.items()}


def f(x, w, p=1):
    return f"{x:{w}.{p}f}" if x else " " * (w - 1) + "-"


def one():
    by = rows()
    out = ["One request at a time (six prompts, 256 tokens, greedy, thinking off, 3 repeats)",
           f"{'arm':10s} {'e2e':>7s} {'decode':>7s} {'ttft ms':>8s} {'4 at once':>9s} {'x vllm':>7s} {'repeat':>6s}"]
    base = gm(list(per_prompt(by["vllm"], "e2e", "tps").values()))
    for a in ("sglang",) + REF:
        rs = by.get(a, [])
        e2e = gm(list(per_prompt(rs, "e2e", "tps").values()))
        dec = gm(list(per_prompt(rs, "stream", "tps").values()))
        tt = gm(list(per_prompt(rs, "stream", "ttft_s").values()))
        c4 = [r["tps"] for r in rs if r["kind"] == "batch4"]
        c4 = statistics.median(c4) if c4 else None
        sh = defaultdict(set)
        for r in rs:
            if r["kind"] == "e2e":
                sh[r["prompt"]].add(r.get("sha"))
        rep = sum(len(s) == 1 for s in sh.values())
        out.append(f"{a:10s} {f(e2e, 7)} {f(dec, 7)} {f(tt and tt * 1000, 8)} {f(c4, 9)} "
                   f"{f(e2e and e2e / base, 7, 3)} {rep}/{len(sh)}")
    for a in ("jax-bf16", "jaxrust", "vllm"):
        o = {r["prompt"]: r["sha"] for r in by[a] if r["kind"] == "e2e"}
        s = {r["prompt"]: r["sha"] for r in by["sglang"] if r["kind"] == "e2e"}
        out.append(f"greedy replies identical to {a}: {sum(s[p] == o.get(p) for p in s)}/{len(s)}")
    return out


def levels(p):
    return {r["concurrency"]: r for r in json.load(open(p))["concurrency"] if r.get("status") == "ok"}


def sweeps():
    out = ["\nMany requests at once: output tok/s (sweep.py, 5 repeats, unique prompts), vllm = ladder `vllm` arm"]
    for s in SHAPES:
        a = levels(os.path.join(HERE, "sweep-sglang", s, "concurrency.json"))
        b = levels(lad(f"sweep-vllm/{s}/concurrency.json"))
        out.append(f"{s}: {'c':>3s} {'sglang':>8s} {'vllm':>8s} {'x vllm':>7s} {'sgl ttft ms':>11s} {'sgl spread %':>12s}")
        for c in sorted(a):
            x, y = a[c]["output_tokens_per_second"], b.get(c, {}).get("output_tokens_per_second")
            out.append(f"{'':13s}{c:3d} {x:8.1f} {y or 0:8.1f} {x / y if y else 0:7.3f} "
                       f"{a[c]['ttft_ms_median']:11.1f} {a[c]['output_tps_spread_pct']:12.1f}")
    return out


def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


def paired(x, y):
    ids = sorted(set(x) & set(y))
    d = [int(bool(x[i].get("right"))) - int(bool(y[i].get("right"))) for i in ids]
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return sum(d) / len(d), b[50], b[1949], sum(v != 0 for v in d), len(ids)


def accuracy():
    out = ["\nGSM8K (1,319, 2,048 tokens), one request at a time, thinking off"]
    sg = load(os.path.join(HERE, "acc-sglang", "gsm8k.jsonl"))
    right = sum(bool(r.get("right")) for r in sg.values())
    out.append(f"sglang: {right}/{len(sg)} = {right / len(sg):.4f}")
    for a in REF:
        p = lad(f"acc-{a}/gsm8k.jsonl")
        if not p:
            continue
        o = load(p)
        m, lo, hi, nd, n = paired(sg, o)
        r2 = sum(bool(r.get("right")) for r in o.values())
        out.append(f"sglang - {a:9s} ({r2 / len(o):.4f}): {m:+.4f} [{lo:+.4f}, {hi:+.4f}] differ {nd} of {n}")
    return out


if __name__ == "__main__":
    text = "\n".join(one() + sweeps() + accuracy()) + "\n"
    open(os.path.join(HERE, "SUMMARY.txt"), "w").write(text)
    print(text)
