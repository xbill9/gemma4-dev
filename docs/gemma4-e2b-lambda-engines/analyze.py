#!/usr/bin/env python3
"""Medians, Lambda/native ratios and seconds Lambda adds per 256-token request, from evidence/."""
import glob
import json
import statistics as st

# Native G6 non-streaming tok/s for one request and 4 at once (evidence/g6-native-SUMMARY.txt, e2e and 4 at once).
NATIVE = {"vllm-drafter": (275.3, 711.3), "vllm-nodrafter": (140.4, 537.2), "llamacpp": (207.7, 373.2),
          "sglang": (136.5, 475.4), "pytorch": (134.7, 134.5), "jax": (48.3, None)}
# The first request on a new environment (compiles), left out: vLLM drafter run1's first 1-request row, JAX run1.
SKIP = {("vllm-drafter", "evidence/lambda-vllm-drafter-run1.json", 0), ("jax", "evidence/lambda-jax-run1.json", 0)}

for eng, (n1, n4) in NATIVE.items():
    one, secs, four = [], [], []
    for f in sorted(glob.glob(f"evidence/lambda-{eng}-run*.json")):
        for i, row in enumerate(json.load(open(f))):
            if (eng, f, i) in SKIP:
                continue
            if row["concurrency"] == 1:
                one.append(row["agg_tok_s"])
                secs.append(row["median_request_s"])
            elif row["concurrency"] == 4:
                four.append(row["agg_tok_s"])
    m1, m4, s = st.median(one), st.median(four), st.median(secs)
    line = (f"{eng:15} one {m1:6.1f} tok/s ({len(one)} runs) native {n1:6.1f} ratio {m1 / n1:.2f} | "
            f"s/request lambda {s:.3f} native {256 / n1:.3f} added {s - 256 / n1:.3f} | "
            f"4 at once {m4:6.1f} ({len(four)} runs, {min(four)}-{max(four)})")
    if n4:
        line += f" native {n4} ratio {m4 / n4:.2f}"
    print(line + f" | 4 vs one {m4 / m1:.2f}x")
