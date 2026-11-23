#!/usr/bin/env python3
"""12B AIME 2025 thinking-off replies that hit the 32,768-token limit: does each repeat a block verbatim?

A reply counts as repeating when some block of at least 200 characters, sampled at a quarter-block
stride in its last 20,000 characters, occurs 3 or more times. A crude test: a repeat that drifts by
a character between copies is missed, so it undercounts. Writes aime_loops.txt beside this file.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "..", "..", "local-llamacpp-3050oem-12b-q4_0", "benchmarks", "runs",
                   "2026-11-17-thinking-12b-3050oem", "aime2025-repack-none-thinkoff.jsonl")


def repeated_block(t):
    s = t[-20000:]
    for L in (2000, 1000, 500, 200):
        seen = {}
        for i in range(0, len(s) - L, max(1, L // 4)):
            k = s[i:i + L]
            seen[k] = seen.get(k, 0) + 1
            if seen[k] >= 3:
                return L
    return 0


rows = [json.loads(line) for line in open(RUN)]
cut = [r for r in rows if r.get("finish") == "length"]
out = [f"12B AIME 2025 thinking off: {len(cut)} of {len(rows)} replies hit the limit"]
n = 0
for r in cut:
    b = repeated_block(r.get("text") or "")
    n += b > 0
    out.append(f"  {r['id']}  {'right' if r.get('right') else 'wrong'}  {len(r.get('text') or ''):,} chars  repeated block {b or 'none'}")
out.append(f"{n} of {len(cut)} repeat a block of 200+ characters 3 or more times")
open(os.path.join(HERE, "aime_loops.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
