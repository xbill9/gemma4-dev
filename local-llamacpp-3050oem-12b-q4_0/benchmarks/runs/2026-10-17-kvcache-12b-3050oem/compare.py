"""Answer quality by context cache type, paired by question against the f16 cache of
2026-10-16-accuracy-12b-3050oem (same binary, model, drafter and method; f16 served at -c 24576
with the drafter and 24576 without). Prints accuracy per arm, arm minus f16 with the same drafter
setting with a 95% paired bootstrap range, disagreements, identical replies and wall time."""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
F16 = os.path.join(HERE, "..", "2026-10-16-accuracy-12b-3050oem")
# arm -> (directory, file suffix, f16 arm with the same drafter setting)
ARMS = {
    "f16 drafter": (F16, "q4x", None),
    "f16 none": (F16, "none", None),
    "q8_0 drafter": (HERE, "q8x", "f16 drafter"),
    "q8_0 none": (HERE, "q8none", "f16 none"),
    "q4_0 drafter": (HERE, "q4x", "f16 drafter"),
    "q4_0 none": (HERE, "q4none", "f16 none"),
}


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


def key(r):
    return (r.get("text") or "", json.dumps(r.get("calls"), sort_keys=True))


for t in ("gsm8k", "bfcl_simple"):
    rows = {a: load(os.path.join(d, f"{t}-{s}.jsonl")) for a, (d, s, _) in ARMS.items()}
    for a, (d, s, ref) in ARMS.items():
        x = rows[a]
        if not x:
            continue
        acc = sum(bool(r.get("right")) for r in x.values()) / len(x)
        trunc = sum(r.get("finish") == "length" for r in x.values())
        line = (
            f"{t:12s} {a:13s} n={len(x)} acc {acc:.4f} truncated {trunc} wall {wall(os.path.join(d, f'{t}-{s}.log'))} s"
        )
        y = rows.get(ref) if ref else None
        ids = sorted(set(x) & set(y)) if y else []
        if ids:
            dd = [int(bool(x[i].get("right"))) - int(bool(y[i].get("right"))) for i in ids]
            lo, hi = boot(dd)
            same = sum(key(x[i]) == key(y[i]) for i in ids)
            line += (
                f" | minus {ref}: {sum(dd) / len(dd):+.4f} 95% [{lo:+.4f}, {hi:+.4f}]"
                f" disagree {sum(v != 0 for v in dd)} identical {same} of {len(ids)}"
            )
        print(line)
