"""Cache off (this run) against cache on (2026-10-14-drafter-q4exact-3050oem/accuracy), per arm and task:
accuracy, paired disagreements, identical texts, and a 95% paired bootstrap range of the difference."""
import json
import os
import random
HERE = os.path.dirname(os.path.abspath(__file__))
ON = os.path.join(HERE, "..", "2026-10-14-drafter-q4exact-3050oem", "accuracy")
for arm in ("e4b-none", "e4b-q4x", "e2b-none", "e2b-q4x"):
    for t in ("gsm8k", "bfcl_simple"):
        off_f = os.path.join(HERE, f"{t}-{arm}-nocache.jsonl")
        on_f = os.path.join(ON, f"{t}-{arm}.jsonl")
        if not (os.path.exists(off_f) and os.path.exists(on_f)):
            continue
        off = {r["id"]: r for r in map(json.loads, open(off_f)) if "id" in r}
        on = {r["id"]: r for r in map(json.loads, open(on_f)) if "id" in r}
        ids = sorted(set(off) & set(on))
        if not ids:
            continue
        d = [int(bool(off[i].get("right"))) - int(bool(on[i].get("right"))) for i in ids]
        rng = random.Random(0)
        boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
        same = sum((off[i].get("text") or "") == (on[i].get("text") or "") for i in ids)
        acc_off = sum(bool(off[i].get("right")) for i in ids) / len(ids)
        acc_on = sum(bool(on[i].get("right")) for i in ids) / len(ids)
        print(f"{arm:9s} {t:12s} n={len(ids)} cache on {acc_on:.4f} off {acc_off:.4f} diff {acc_off - acc_on:+.4f} "
              f"95% [{boots[50]:+.4f}, {boots[1949]:+.4f}] disagree {sum(x != 0 for x in d)} identical-text {same}")
