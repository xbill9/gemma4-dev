"""26B A4B scores with and without the drafter, paired: accuracy per arm, drafter minus none with a
95% paired bootstrap range, disagreements, word-for-word identical replies and wall time. Then the
26B against the 12B at the 12B's own serving setting (q4_0 cache, -c 81920, exact Q4_0 12B drafter /
none; local-llamacpp-3050oem-12b-q4_0/benchmarks/runs/2026-10-17-kvcache-12b-3050oem), paired by
question. Same binary (7182ef71e), card and method; different context size. Adapted from the 12B's
2026-10-16-accuracy-12b-3050oem/compare.py."""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
B12 = os.path.join(HERE, "..", "..", "..", "..", "local-llamacpp-3050oem-12b-q4_0", "benchmarks", "runs",
                   "2026-10-17-kvcache-12b-3050oem")
ARMS12 = {"q4x": "q4x", "none": "q4none"}


def load(f):
    return {r["id"]: r for r in map(json.loads, open(f)) if "id" in r} if os.path.exists(f) else {}


def boot(d):
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return b[50], b[1949]


def wall(log):
    try:
        return json.loads(open(log).read().strip().splitlines()[-1])["wall_s"]
    except (OSError, ValueError, KeyError, IndexError):
        return None


def paired(x, y):
    ids = sorted(set(x) & set(y))
    d = [int(bool(x[i].get("right"))) - int(bool(y[i].get("right"))) for i in ids]
    lo, hi = boot(d)
    return ids, d, lo, hi


for t in ("gsm8k", "bfcl_simple"):
    arms = {a: load(os.path.join(HERE, f"{t}-{a}.jsonl")) for a in ("q4x", "none")}
    for a, rows in arms.items():
        if rows:
            acc = sum(bool(r.get("right")) for r in rows.values()) / len(rows)
            right = sum(bool(r.get("right")) for r in rows.values())
            w = wall(os.path.join(HERE, f"{t}-{a}.log"))
            trunc = sum(r.get("finish") == "length" for r in rows.values())
            print(f"{t:12s} 26B {a:5s} n={len(rows)} right {right} acc {acc:.4f} truncated {trunc} wall {w} s")
    x, y = arms["q4x"], arms["none"]
    if x and y:
        ids, d, lo, hi = paired(x, y)
        same = sum((x[i].get("text") or "", json.dumps(x[i].get("calls"), sort_keys=True))
                   == (y[i].get("text") or "", json.dumps(y[i].get("calls"), sort_keys=True)) for i in ids)
        wx, wy = wall(os.path.join(HERE, f"{t}-q4x.log")), wall(os.path.join(HERE, f"{t}-none.log"))
        speed = f"{wy / wx:.2f}x" if wx and wy else "n/a"
        print(f"{t:12s} 26B drafter - none: {sum(d) / len(d):+.4f} 95% [{lo:+.4f}, {hi:+.4f}] "
              f"disagree {sum(v != 0 for v in d)} (+{d.count(1)} / -{d.count(-1)}) "
              f"identical-text {same} of {len(ids)}; time none/drafter {speed}")
    for a, a12 in ARMS12.items():
        z = load(os.path.join(B12, f"{t}-{a12}.jsonl"))
        if not z:
            continue
        print(f"{t:12s} 12B {a:5s} n={len(z)} acc {sum(bool(r.get('right')) for r in z.values()) / len(z):.4f} (reference)")
        if arms[a]:
            ids, d, lo, hi = paired(arms[a], z)
            print(f"{t:12s} 26B {a} - 12B {a}: {sum(d) / len(d):+.4f} 95% [{lo:+.4f}, {hi:+.4f}] "
                  f"(+{d.count(1)} / -{d.count(-1)} of {len(ids)})")
