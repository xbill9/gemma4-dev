#!/usr/bin/env python3
"""Every derived figure in the article, from the files beside this script. Writes derived.txt."""

import json
import re
from pathlib import Path

H = Path(__file__).resolve().parent
out = []


def p(*a):
    out.append(" ".join(str(x) for x in a))


def rows(*files):
    d = {}
    for f in files:
        for r in json.loads((H / f).read_text()):
            d.setdefault(r["concurrency"], []).append(r)
    return d


nd = rows(
    "first-light-bench-run1.json",
    "first-light-bench-run2.json",
    "nodrafter-repeat-bench-run1.json",
    "nodrafter-repeat-bench-run2.json",
)
dr = rows("drafter-bench-run1.json", "drafter-bench-run2.json")


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]  # upper middle for an even count


p("## Output tok/s, 256 tokens per request")
for c in sorted(dr):
    a = [r["agg_tok_s"] for r in nd[c]]
    b = [r["agg_tok_s"] for r in dr[c]]
    p(
        f"c={c} no-drafter min {min(a)} max {max(a)} median {med(a)} | drafter {b} median {med(b)} "
        f"| ratio of medians {med(b) / med(a):.2f}"
    )

p("## Single-request seconds (256 tokens)")
p("no drafter", sorted(r["requests"][0]["seconds"] for r in nd[1]))
p("drafter", sorted(r["requests"][0]["seconds"] for r in dr[1]))

p("## Errors")
p(
    "no drafter",
    sum(r["errors"] for v in nd.values() for r in v),
    "of",
    sum(r["concurrency"] for v in nd.values() for r in v),
    "requests",
)
p(
    "drafter",
    sum(r["errors"] for v in dr.values() for r in v),
    "of",
    sum(r["concurrency"] for v in dr.values() for r in v),
    "requests",
)

p("## Base image")
m = json.loads((H / "base-image-manifest.json").read_text())
p("compressed layers GB", round(sum(x["size"] for x in m["layers"]) / 1e9, 2), "layers", len(m["layers"]))
cfg = json.loads((H / "base-image-config.json").read_text())
env = dict(e.split("=", 1) for e in cfg["config"]["Env"])
p("CUDA_VERSION", env["CUDA_VERSION"], "created", cfg["created"][:10])
hist = " ".join(h.get("created_by", "") for h in cfg["history"])
p(
    "FRAMEWORK_VERSION",
    re.search(r"FRAMEWORK_VERSION=(\S+)", hist).group(1),
    "PYTHON_VERSION",
    re.search(r"PYTHON_VERSION=(\S+)", hist).group(1),
)

p("## Drafter acceptance (vLLM SpecDecoding metrics)")
txt = (H / "drafter-spec-metrics.log").read_text()
L = [float(x) for x in re.findall(r"Mean acceptance length: ([0-9.]+)", txt)]
R = [float(x) for x in re.findall(r"Avg Draft acceptance rate: ([0-9.]+)%", txt)]
p(
    "mean acceptance length min",
    min(L),
    "max",
    max(L),
    "| draft acceptance % min",
    min(R),
    "max",
    max(R),
    "| lines",
    len(L),
)

p("## Startup (first light)")
s = (H / "first-light-startup.log").read_text()
p("initReport ms", re.search(r'"durationMs":([0-9.]+)', s).group(1))
p(re.search(r"Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds", s).group(0))
p(re.search(r"torch.compile took [0-9.]+ s in total", s).group(0))

p("## Quota arithmetic")
p("default 3 environments x 8 vCPUs per g6.2xlarge =", 3 * 8, "vCPUs; the account's G quota is 16 vCPUs")
p("## Request size")
p("max_tokens with ignore_eos: 256 tokens per request")

(H / "derived.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
