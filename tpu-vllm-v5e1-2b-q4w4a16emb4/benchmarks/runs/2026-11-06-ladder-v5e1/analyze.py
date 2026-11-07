"""Tables of the v5e-1 ladder (2026-11-06/07), from the three VMs' run directories next to this file.

    python3 analyze.py          # writes SUMMARY.txt next to this file and prints it

One subdirectory per VM (named for its queued resource), each the VM's /opt/ladder-out/2026-11-06-ladder-v5e1.
- One request: per arm of results.jsonl (ladder.py `one`), geometric mean over the six prompts of each prompt's
  median, for end-to-end tok/s, stream decode tok/s and TTFT; median aggregate tok/s of four requests at once;
  prompts whose three greedy e2e replies are identical; server start time.
- Sweeps: output tok/s per level from sweep-<arm>/<shape>/concurrency.json (sweep.py, 5 repeats, unique prompts),
  and the same arm on the G6 L4 (gpu-vllm-g6-2b-q4w4a16emb4 2026-11-06-suite-g6) where the checkpoint matches.
- Accuracy: right answers per task, paired by question against `vllm` with a 95% paired bootstrap range (2,000
  resamples, seed 0), as the G6 analyze.py does.
"""

import json
import math
import os
import random
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
G6 = os.path.join(ROOT, "gpu-vllm-g6-2b-q4w4a16emb4", "benchmarks", "runs", "2026-11-06-suite-g6")
SHAPES = ("in512-out128", "in128-out512")
ARMS = ("vllm", "vllm-text", "vllm-text-d4", "jax", "jax-bf16", "jaxrust")
ONE_ARMS = ARMS + ("torchtpu-bf16", "torchtpu-rust")  # TorchTPU: `one` only, its GSM8K runs were stopped
SWEEP_ARMS = ("vllm", "vllm-text", "vllm-text-d4")
G6_SWEEP = {"vllm": "vllm"}  # same checkpoint (-text-emb4) and engine; the G6 drafter arm used the QAT drafter


def vms():
    return sorted(d for d in os.listdir(HERE) if os.path.isdir(os.path.join(HERE, d)))


def find(rel):
    """The one VM directory holding rel with a DONE marker beside it (A's jax/ dirs hold only MOVED)."""
    for v in vms():
        p = os.path.join(HERE, v, rel)
        if os.path.exists(p) and os.path.exists(os.path.join(HERE, v, rel.split("/")[0], "DONE")):
            return p
    return None


def gm(v):
    return math.exp(sum(map(math.log, v)) / len(v)) if v and all(x > 0 for x in v) else None


def fmt(x, w=7, p=1):
    return f"{x:{w}.{p}f}" if x is not None else " " * (w - 1) + "-"


def one():
    rows = []
    for v in vms():
        p = os.path.join(HERE, v, "results.jsonl")
        if os.path.exists(p):
            rows += [json.loads(x) for x in open(p)]
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(r)

    def per_prompt(rs, kind, key):
        d = defaultdict(list)
        for r in rs:
            if r["kind"] == kind and r.get(key) is not None:
                d[r["prompt"]].append(r[key])
        return {p: statistics.median(x) for p, x in d.items()}

    out = ["One request at a time (six prompts, 256 tokens, greedy, thinking off, 3 repeats), then 4 at once"]
    out.append(
        f"{'arm':14s} {'e2e tok/s':>9s} {'decode':>7s} {'TTFT ms':>7s} {'4 at once':>9s} {'x1':>5s}"
        f" {'repeat':>6s} {'start s':>7s} {'x vllm':>6s}"
    )
    base = None
    for a in ONE_ARMS:
        rs = by.get(a)
        if not rs:
            out.append(f"{a:14s} (not run)")
            continue
        e2e = gm(list(per_prompt(rs, "e2e", "tps").values()))
        dec = gm(list(per_prompt(rs, "stream", "tps").values()))
        tt = gm(list(per_prompt(rs, "stream", "ttft_s").values()))
        b4 = [r["tps"] for r in rs if r["kind"] == "batch4"]
        b4 = statistics.median(b4) if b4 else None
        shas = defaultdict(set)
        for r in rs:
            if r["kind"] == "e2e":
                shas[r["prompt"]].add(r.get("sha"))
        rep = f"{sum(len(s) == 1 for s in shas.values())}/{len(shas)}"
        if a == "vllm":
            base = e2e
        out.append(
            f"{a:14s} {fmt(e2e, 9)} {fmt(dec)} {fmt(tt * 1000 if tt else None)} {fmt(b4, 9)}"
            f" {fmt(b4 / e2e if b4 and e2e else None, 5, 2)} {rep:>6s} {fmt(rs[0].get('ready_s'), 7, 0)}"
            f" {fmt(e2e / base if base and e2e else None, 6, 3)}"
        )
    return out


def levels(path):
    try:
        d = json.load(open(path))
    except (OSError, TypeError):
        return None
    return {c["concurrency"]: c["output_tokens_per_second"] for c in d["concurrency"] if c["status"] == "ok"}


def sweeps():
    out = []
    v5 = {a: {s: levels(find(f"sweep-{a}/{s}/concurrency.json")) for s in SHAPES} for a in SWEEP_ARMS}
    for s in SHAPES:
        lv = sorted({k for a in SWEEP_ARMS for k in (v5[a][s] or {})})
        out.append(f"\n{s}: output tok/s on one v5e chip (5 repeats, unique prompts)")
        out.append(f"{'arm':14s}" + "".join(f"{c:>8d}" for c in lv) + "   peak@c   x c=1")
        for a in SWEEP_ARMS:
            d = v5[a][s]
            if not d:
                out.append(f"{a:14s} (not run)")
                continue
            pk = max(d, key=d.get)
            out.append(
                f"{a:14s}" + "".join(" " + fmt(d.get(c)) for c in lv) + f"  {d[pk]:7.1f}@{pk:<2d} {d[pk] / d[1]:5.2f}"
            )
        x, y = v5["vllm-text-d4"][s], v5["vllm-text"][s]
        if x and y:
            out.append(
                "  vllm-text-d4 / vllm-text" + "".join(f"  c{c}={x[c] / y[c]:.3f}" for c in lv if c in x and c in y)
            )
        x, y = v5["vllm"][s], v5["vllm-text"][s]
        if x and y:
            out.append(
                "  vllm / vllm-text        " + "".join(f"  c{c}={x[c] / y[c]:.3f}" for c in lv if c in x and c in y)
            )
        for a, g in G6_SWEEP.items():
            x, y = v5[a][s], levels(os.path.join(G6, f"sweep-{g}", s, "concurrency.json"))
            if x and y:
                out.append(
                    f"  {a} v5e-1 / L4 {g}".ljust(26)
                    + "".join(f"  c{c}={x[c] / y[c]:.3f}" for c in lv if c in x and c in y)
                )
    return out


def load(path):
    try:
        return {r["id"]: r for r in map(json.loads, open(path))}
    except (OSError, TypeError):
        return None


def wall(path):
    try:
        return json.loads(open(path).read().strip().splitlines()[-1])["wall_s"]
    except (OSError, TypeError, ValueError, KeyError, IndexError):
        return None


def paired(x, y):
    ids = sorted(set(x) & set(y))
    d = [int(bool(x[i].get("right"))) - int(bool(y[i].get("right"))) for i in ids]
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return sum(d) / len(d), b[50], b[1949], sum(v != 0 for v in d)


def accuracy():
    out = []
    for t in ("gsm8k", "bfcl_simple"):
        out.append(f"\n{t}: one request at a time, thinking off")
        out.append(
            f"{'arm':14s} {'n':>5s} {'right':>5s} {'acc':>7s} {'trunc':>5s} {'wall s':>8s} {'x vllm':>6s}"
            f"   arm - vllm paired [95%]      differ"
        )
        rows = {a: load(find(f"acc-{a}/{t}.jsonl")) for a in ARMS}
        walls = {a: wall(find(f"acc-{a}/{t}.log")) for a in ARMS}
        for a in ARMS:
            r = rows[a]
            if not r:
                out.append(f"{a:14s} (not run)")
                continue
            right = sum(bool(x.get("right")) for x in r.values())
            trunc = sum(x.get("finish") == "length" for x in r.values())
            w, wv = walls[a], walls["vllm"]
            s = ""
            if a != "vllm" and rows["vllm"]:
                m, lo, hi, nd = paired(r, rows["vllm"])
                s = f"   {m:+.4f} [{lo:+.4f}, {hi:+.4f}] {nd:6d}"
            out.append(
                f"{a:14s} {len(r):5d} {right:5d} {right / len(r):7.4f} {trunc:5d} {fmt(w, 8, 1)}"
                f" {fmt(w / wv if w and wv else None, 6, 3)}{s}"
            )
    return out


if __name__ == "__main__":
    text = "\n".join(one() + sweeps() + accuracy()) + "\n"
    open(os.path.join(HERE, "SUMMARY.txt"), "w").write(text)
    print(text)
