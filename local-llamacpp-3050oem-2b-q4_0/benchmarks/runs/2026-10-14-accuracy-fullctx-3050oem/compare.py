"""Full-context one-at-a-time scores (this run) per arm, with the 2026-10-14 scores at -c 32768
--parallel 8 (cache on) beside them, paired disagreements and 95% paired bootstrap ranges."""
import json
import os
import random
HERE = os.path.dirname(os.path.abspath(__file__))
P8 = os.path.join(HERE, "..", "2026-10-14-drafter-q4exact-3050oem", "accuracy")
def load(f):
    return {r["id"]: r for r in map(json.loads, open(f)) if "id" in r} if os.path.exists(f) else {}
def boot(d):
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return b[50], b[1949]
for t in ("gsm8k", "bfcl_simple"):
    for arm in ("e4b-q4x", "e4b-none", "e2b-q4x", "e2b-none"):
        a = load(os.path.join(HERE, f"{t}-{arm}.jsonl"))
        if not a:
            continue
        acc = sum(bool(r.get("right")) for r in a.values()) / len(a)
        line = f"{t:12s} {arm:9s} n={len(a)} full-ctx c=1 {acc:.4f}"
        b = load(os.path.join(P8, f"{t}-{arm}.jsonl"))
        ids = sorted(set(a) & set(b))
        if ids:
            d = [int(bool(a[i].get("right"))) - int(bool(b[i].get("right"))) for i in ids]
            lo, hi = boot(d)
            line += f" | -c 32768 c=8 {sum(bool(b[i].get('right')) for i in ids) / len(ids):.4f} diff {sum(d) / len(d):+.4f} 95% [{lo:+.4f}, {hi:+.4f}] disagree {sum(x != 0 for x in d)}"
        print(line)
    for m in ("e4b", "e2b"):
        x, y = load(os.path.join(HERE, f"{t}-{m}-q4x.jsonl")), load(os.path.join(HERE, f"{t}-{m}-none.jsonl"))
        ids = sorted(set(x) & set(y))
        if ids:
            d = [int(bool(x[i].get("right"))) - int(bool(y[i].get("right"))) for i in ids]
            lo, hi = boot(d)
            print(f"{t:12s} {m} drafter - none at full ctx: {sum(d) / len(d):+.4f} 95% [{lo:+.4f}, {hi:+.4f}] disagree {sum(v != 0 for v in d)} identical-text {sum((x[i].get('text') or '') == (y[i].get('text') or '') for i in ids)}")
