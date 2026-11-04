"""SGLang with the W4A16 drafter at 1-4 drafted tokens under load, against SGLang without a drafter
(../2026-11-04-concurrency-3050oem) and vLLM with the same drafter at its best length per level
(local-vllm-3050oem-2b-q4w4a16emb4/.../2026-11-03-concurrency-drafters-3050oem, no drafter
2026-11-03-concurrency-3050oem). Aggregate output tok/s, median of 5 repeats. The 4-token SGLang arm holds at
most 16 running requests; beyond 16 the rest queue."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
VR = os.path.join(ROOT, "local-vllm-3050oem-2b-q4w4a16emb4", "benchmarks", "runs")
ARMS = [f"w4a16-n{n}" for n in (1, 2, 3, 4)]


def cells(path):
    return {
        c["concurrency"]: c["output_tokens_per_second"]
        for c in json.load(open(path))["concurrency"]
        if c.get("status", "ok") == "ok"
    }


for shape in ("in512-out128", "in128-out512"):
    s0 = cells(os.path.join(HERE, "..", "2026-11-04-concurrency-3050oem", shape, "concurrency.json"))
    v0 = cells(os.path.join(VR, "2026-11-03-concurrency-3050oem", shape, "concurrency.json"))
    s = {a: cells(os.path.join(HERE, a, shape, "concurrency.json")) for a in ARMS}
    v = {
        a: cells(os.path.join(VR, "2026-11-03-concurrency-drafters-3050oem", a, shape, "concurrency.json"))
        for a in ARMS
    }
    print(f"\n{shape}")
    print(
        f"{'c':>3} {'SGL none':>8} "
        + " ".join(f"{'SGL n' + a[-1]:>12}" for a in ARMS)
        + f" {'SGL best':>14} {'vLLM none':>9} {'vLLM best':>15} {'SGL/vLLM best':>13}"
    )
    for c in sorted(s0):
        sb = max(ARMS, key=lambda a: s[a][c])
        vb = max(ARMS, key=lambda a: v[a][c])
        best_s = max(s0[c], s[sb][c])
        best_v = max(v0[c], v[vb][c])
        print(
            f"{c:3d} {s0[c]:8.1f} "
            + " ".join(f"{s[a][c]:7.1f} {s[a][c] / s0[c]:4.2f}x" for a in ARMS)
            + f" {s[sb][c]:7.1f} (n{sb[-1]}) {v0[c]:9.1f} {v[vb][c]:7.1f} (n{vb[-1]})"
            + f" {best_s / best_v:13.2f}"
        )
