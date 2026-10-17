"""12B scores with and without the drafter, paired: accuracy per arm, drafter minus none with a
95% paired bootstrap range, disagreements, word-for-word identical replies and wall time. E4B's
full-context one-at-a-time scores (2026-10-14-accuracy-fullctx-3050oem, llama.cpp 77c4540ce,
-c 131072) are printed beside them for scale; they are a different model on a different build."""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
E4B = os.path.join(
    HERE,
    "..",
    "..",
    "..",
    "..",
    "local-llamacpp-3050oem-2b-q4_0",
    "benchmarks",
    "runs",
    "2026-10-14-accuracy-fullctx-3050oem",
)


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


for t in ("gsm8k", "bfcl_simple"):
    arms = {a: load(os.path.join(HERE, f"{t}-{a}.jsonl")) for a in ("q4x", "none")}
    for a, rows in arms.items():
        if rows:
            acc = sum(bool(r.get("right")) for r in rows.values()) / len(rows)
            w = wall(os.path.join(HERE, f"{t}-{a}.log"))
            trunc = sum(r.get("finish") == "length" for r in rows.values())
            print(f"{t:12s} 12B {a:5s} n={len(rows)} acc {acc:.4f} truncated {trunc} wall {w} s")
    x, y = arms["q4x"], arms["none"]
    ids = sorted(set(x) & set(y))
    if ids:
        d = [int(bool(x[i].get("right"))) - int(bool(y[i].get("right"))) for i in ids]
        lo, hi = boot(d)
        same = sum((x[i].get("text") or "", json.dumps(x[i].get("calls"), sort_keys=True))
                   == (y[i].get("text") or "", json.dumps(y[i].get("calls"), sort_keys=True)) for i in ids)
        wx, wy = wall(os.path.join(HERE, f"{t}-q4x.log")), wall(os.path.join(HERE, f"{t}-none.log"))
        speed = f"{wy / wx:.2f}x" if wx and wy else "n/a"
        print(
            f"{t:12s} 12B drafter - none: {sum(d) / len(d):+.4f} 95% [{lo:+.4f}, {hi:+.4f}] "
            f"disagree {sum(v != 0 for v in d)} identical-text {same} of {len(ids)}; time none/drafter {speed}"
        )
    for a in ("q4x", "none"):
        e = load(os.path.join(E4B, f"{t}-e4b-{a}.jsonl"))
        if e:
            print(
                f"{t:12s} E4B {a:5s} n={len(e)} acc {sum(bool(r.get('right')) for r in e.values()) / len(e):.4f} (reference)"
            )
    for a, mine in (("q4x", x), ("none", y)):
        z = load(os.path.join(E4B, f"{t}-e4b-{a}.jsonl"))
        ids = sorted(set(mine) & set(z))
        if ids:
            d = [int(bool(mine[i].get("right"))) - int(bool(z[i].get("right"))) for i in ids]
            lo, hi = boot(d)
            print(
                f"{t:12s} 12B {a} - E4B {a}: {sum(d) / len(d):+.4f} 95% [{lo:+.4f}, {hi:+.4f}] (different build and -c)"
            )
