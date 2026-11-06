"""Concurrency sweeps and GSM8K/BFCL accuracy of the G6 flight (one NVIDIA L4), against the 3050 runs.

    python3 analyze.py            # writes SWEEP.txt, COMPARE-3050.txt and ACCURACY.txt next to this file

Sweeps: output tok/s per level from each arm's <shape>/concurrency.json (sweep.py, 5 repeats, unique
prompts), the same file from the matching 3050 run, and their ratio. Accuracy: right answers per task, and
paired by question against the same engine on the 3050 and against llama.cpp on the L4, with a 95% paired
bootstrap range (2,000 resamples, seed 0), as the 3050 compare.py scripts do.
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
SHAPES = ("in512-out128", "in128-out512")


def r3050(rig, run, sub=""):
    return os.path.join(ROOT, rig, "benchmarks/runs", run, sub)


V = "local-vllm-3050oem-2b-q4w4a16emb4"
S = "local-sglang-3050oem-2b-q4w4a16emb4"
L = "local-llamacpp-3050oem-2b-q4_0"
M = "local-mistralrs-3050oem-2b-q4_0"
SWEEP_3050 = {
    "vllm": r3050(V, "2026-11-03-concurrency-3050oem"),
    "vllm-d1": r3050(V, "2026-11-03-concurrency-drafters-3050oem", "w4a16-n1"),
    "vllm-d4": r3050(V, "2026-11-03-concurrency-drafters-3050oem", "w4a16-n4"),
    "sglang": r3050(S, "2026-11-04-concurrency-3050oem"),
    "sglang-d4": r3050(S, "2026-11-04-concurrency-drafters-3050oem", "w4a16-n4"),
    "llama": r3050(L, "2026-11-03-concurrency-3050oem"),
    "llama-d2": r3050(L, "2026-11-03-concurrency-drafters-3050oem", "q4x-n2"),
    "mistralrs": r3050(M, "2026-11-04-concurrency-3050oem"),
    "mistralrs-card": None,
}
LA = "local-litertlm-3050oem-2b-q4w4a16emb4"
ACC_3050 = {  # arm -> {task: (dir, stem)}
    "llama": (r3050(L, "2026-10-14-accuracy-fullctx-3050oem"), "{t}-e2b-none"),
    "llama-d2": (r3050(L, "2026-10-14-accuracy-fullctx-3050oem"), "{t}-e2b-q4x"),
    "vllm": (r3050(V, "2026-11-03-accuracy-3050oem"), "{t}-vllm"),
    "vllm-d4": (r3050(V, "2026-11-03-accuracy-drafter-3050oem"), "{t}-vllm-w4a16n4"),
    "sglang": (r3050(S, "2026-11-04-accuracy-3050oem"), "{t}-sglang"),
    "sglang-d4": (r3050(S, "2026-11-04-accuracy-drafter-3050oem"), "{t}-sglang-w4a16n4"),
    "mistralrs": (r3050(M, "2026-11-04-accuracy-3050oem"), "{t}-mistralrs"),
    "pytorch": (r3050("local-pytorch-3050oem-2b-q4w4a16", "2026-10-31-accuracy-3050oem"), "{t}-pytorch"),
    "litert-int248": (r3050(LA, "2026-11-02-accuracy-3050oem"), "{t}-int248"),
    "litert-q4": (r3050(LA, "2026-11-02-accuracy-3050oem"), "{t}-q4exact"),
}
ACC_ARMS = list(ACC_3050)
SWEEP_ARMS = list(SWEEP_3050)


def levels(path):
    try:
        d = json.load(open(path))
    except OSError:
        return None
    return {c["concurrency"]: c["output_tokens_per_second"] for c in d["concurrency"] if c["status"] == "ok"}


def fmt(x, w=7, p=1):
    return f"{x:{w}.{p}f}" if x is not None else " " * (w - 1) + "-"


def sweeps():
    out, cmp_ = [], []
    l4 = {a: {s: levels(os.path.join(HERE, f"sweep-{a}", s, "concurrency.json")) for s in SHAPES} for a in SWEEP_ARMS}
    for s in SHAPES:
        lv = sorted({k for a in SWEEP_ARMS for k in (l4[a][s] or {})})
        out.append(f"\n{s}: output tok/s on the L4 (5 repeats, unique prompts)")
        out.append(f"{'arm':15s}" + "".join(f"{c:>8d}" for c in lv) + "   peak@c  x c=1  vs vllm@64")
        base64 = (l4["vllm"][s] or {}).get(64)
        for a in SWEEP_ARMS:
            d = l4[a][s]
            if not d:
                out.append(f"{a:15s} (not run)")
                continue
            pk = max(d, key=d.get)
            out.append(
                f"{a:15s}"
                + "".join(" " + fmt(d.get(c)) for c in lv)
                + f"  {d[pk]:7.1f}@{pk:<2d} {d[pk] / d[1]:5.2f}  {fmt(d.get(64) / base64 if base64 and d.get(64) else None, 6, 3)}"
            )
        out.append(f"drafter against no drafter, same engine and shape:")
        for a, b in (("vllm-d1", "vllm"), ("vllm-d4", "vllm"), ("sglang-d4", "sglang"), ("llama-d2", "llama")):
            x, y = l4[a][s], l4[b][s]
            if x and y:
                out.append(f"  {a:13s}/{b:7s}" + "".join(f"  c{c}={x[c] / y[c]:.3f}" for c in lv if c in x and c in y))
        cmp_.append(f"\n{s}: L4 output tok/s / 3050 output tok/s at each level")
        cmp_.append(f"{'arm':15s}" + "".join(f"{c:>8d}" for c in lv) + "   L4 peak  3050 peak  ratio")
        for a in SWEEP_ARMS:
            if not SWEEP_3050[a]:
                continue
            x, y = l4[a][s], levels(os.path.join(SWEEP_3050[a], s, "concurrency.json"))
            if not x or not y:
                cmp_.append(f"{a:15s} (missing: {'L4' if not x else '3050'})")
                continue
            px, py = max(x.values()), max(y.values())
            cmp_.append(
                f"{a:15s}"
                + "".join(" " + fmt(x[c] / y[c] if c in x and c in y else None, 7, 3) for c in lv)
                + f"  {px:8.1f}  {py:9.1f}  {px / py:5.3f}"
            )
    return out, cmp_


def load(path):
    try:
        return {r["id"]: r for r in map(json.loads, open(path))}
    except OSError:
        return None


def wall(path):
    try:
        return json.loads(open(path).read().strip().splitlines()[-1])["wall_s"]
    except (OSError, ValueError, KeyError, IndexError):
        return None


def boot(d):
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return b[50], b[1949]


def paired(x, y):
    ids = sorted(set(x) & set(y))
    d = [int(bool(x[i].get("right"))) - int(bool(y[i].get("right"))) for i in ids]
    lo, hi = boot(d)
    return len(ids), sum(d) / len(d), lo, hi, sum(v != 0 for v in d)


def accuracy():
    out = []
    for t in ("gsm8k", "bfcl_simple"):
        out.append(f"\n{t}: one request at a time, thinking off")
        out.append(
            f"{'arm':14s} {'n':>5s} {'right':>5s} {'acc':>7s} {'trunc':>5s} {'wall s':>8s} {'x llama':>7s}"
            f"  {'3050 acc':>8s} {'3050 s':>7s} {'L4/3050 time':>12s}  L4-3050 paired [95%]       differ"
            f"   L4 arm - L4 llama [95%]"
        )
        rows = {a: load(os.path.join(HERE, f"acc-{a}", f"{t}.jsonl")) for a in ACC_ARMS}
        walls = {a: wall(os.path.join(HERE, f"acc-{a}", f"{t}.log")) for a in ACC_ARMS}
        for a in ACC_ARMS:
            r = rows[a]
            if not r:
                out.append(f"{a:14s} (not run)")
                continue
            right = sum(bool(x.get("right")) for x in r.values())
            trunc = sum(x.get("finish") == "length" for x in r.values())
            w, wl = walls[a], walls["llama"]
            d3, stem = ACC_3050[a]
            p3 = os.path.join(d3, stem.format(t=t))
            r3, w3 = load(p3 + ".jsonl"), wall(p3 + ".log")
            s3 = ""
            if r3:
                n, m, lo, hi, nd = paired(r, r3)
                acc3 = sum(bool(x.get("right")) for x in r3.values()) / len(r3)
                s3 = (
                    f"  {acc3:8.4f} {fmt(w3, 7, 0)} {fmt(w / w3 if w and w3 else None, 12, 3)}"
                    f"  {m:+.4f} [{lo:+.4f}, {hi:+.4f}] {nd:5d}"
                )
            else:
                s3 = "  (no 3050 file)"
            sl = ""
            if a != "llama" and rows["llama"]:
                n, m, lo, hi, nd = paired(r, rows["llama"])
                sl = f"   {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            out.append(
                f"{a:14s} {len(r):5d} {right:5d} {right / len(r):7.4f} {trunc:5d} {fmt(w, 8, 1)}"
                f" {fmt(w / wl if w and wl else None, 7, 3)}{s3}{sl}"
            )
    return out


if __name__ == "__main__":
    sw, cmp_ = sweeps()
    acc = accuracy()
    for name, lines in (("SWEEP.txt", sw), ("COMPARE-3050.txt", cmp_), ("ACCURACY.txt", acc)):
        text = "\n".join(lines).lstrip("\n") + "\n"
        open(os.path.join(HERE, name), "w").write(text)
        print(f"== {name}\n{text}")
