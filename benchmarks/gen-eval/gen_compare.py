#!/usr/bin/env python3
"""gen_compare.py -- pair two gen_eval.py outputs record for record.

  gen_compare.py <reference.jsonl> <test.jsonl> [<label-ref> <label-test>]

Prints one markdown row: n, both accuracies, the difference (test - reference), its 95% range from
10,000 bootstrap resamples of records (seed 0), and the two discordant counts. Records present in
only one file are an error, not dropped.
"""

import json
import random
import sys


def load(path):
    return {r["id"]: bool(r["right"]) for r in map(json.loads, open(path))}


def compare(ref, test, resamples=10_000, seed=0):
    if set(ref) != set(test):
        raise SystemExit(f"record sets differ: {len(set(ref) ^ set(test))} ids in only one file")
    ids = sorted(ref)
    diffs = [int(test[i]) - int(ref[i]) for i in ids]
    n = len(ids)
    rng = random.Random(seed)
    boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return {
        "n": n,
        "ref": sum(ref.values()) / n,
        "test": sum(test.values()) / n,
        "diff": sum(diffs) / n,
        "lo": boot[int(0.025 * resamples)],
        "hi": boot[int(0.975 * resamples) - 1],
        "ref_only": diffs.count(-1),
        "test_only": diffs.count(1),
    }


def main():
    ref, test = sys.argv[1], sys.argv[2]
    names = sys.argv[3:5] if len(sys.argv) >= 5 else [ref, test]
    c = compare(load(ref), load(test))
    print(
        f"| {names[1]} vs {names[0]} | {c['n']} | {c['ref']:.3f} | {c['test']:.3f} | {c['diff']:+.3f} "
        f"| {c['lo']:+.3f} to {c['hi']:+.3f} | {c['ref_only']} | {c['test_only']} |"
    )


if __name__ == "__main__":
    main()
