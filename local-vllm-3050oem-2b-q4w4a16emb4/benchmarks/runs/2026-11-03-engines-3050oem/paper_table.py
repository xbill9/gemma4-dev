"""All six serving engines on the RTX 3050 OEM, Gemma 4 E2B on the QAT Q4_0 grid: one table per question, every
number read from a run directory in this repo. Prints Markdown; `python3 paper_table.py > PAPER-TABLE.md`.

One request (engines.py, six prompts, geometric mean of per-prompt medians, end-to-end tok/s): llama.cpp,
PyTorch, LiteRT and vLLM from this directory (2026-11-03); mistral.rs with a same-day llama.cpp reference and
SGLang with a same-day vLLM reference from their own 2026-11-04 directories. "best drafter" is the fastest
drafted arm measured for that engine on that day. Many requests (sweep.py, median of 5 repeats, no drafter):
each batching engine's concurrency directory. Accuracy (gen_eval.py, one request at a time, no drafter,
thinking off): each engine's accuracy directory. A missing file prints as "pending"."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import summarize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))


def run(rig, d):
    return os.path.join(ROOT, rig, "benchmarks", "runs", d)


LL, VL, MR, SG = (
    "local-llamacpp-3050oem-2b-q4_0",
    "local-vllm-3050oem-2b-q4w4a16emb4",
    "local-mistralrs-3050oem-2b-q4_0",
    "local-sglang-3050oem-2b-q4w4a16emb4",
)
# engine: (label, engines dir, no-drafter arm, drafter arm prefix, same-day reference arm)
ENGINES = {
    "llama.cpp": ("llama.cpp 7182ef71e, exact Q4_0 GGUF", HERE, "llama", "llama-d", None),
    "vLLM": ("vLLM 0.33.0, W4A16 int4 tables", HERE, "vllm", "vllm-d", None),
    "SGLang": (
        "SGLang 0.5.23 (2 patches), W4A16 int4 tables",
        run(SG, "2026-11-04-engines-3050oem"),
        "sglang",
        "sglang-d",
        "vllm",
    ),
    "mistral.rs": (
        "mistral.rs v0.9.5, exact Q4_0 GGUF",
        run(MR, "2026-11-04-engines-3050oem"),
        "mistralrs",
        "mistralrs-d",
        "llama",
    ),
    "PyTorch": ("PyTorch int4 engine (this repo), W4A16", HERE, "pytorch", None, None),
    "LiteRT-LM": ("LiteRT-LM 0.19.0, exact Q4_0", HERE, "litert-q4", "litert-int248-d", None),
}
CONC = {
    "llama.cpp": run(LL, "2026-11-03-concurrency-3050oem"),
    "vLLM": run(VL, "2026-11-03-concurrency-3050oem"),
    "SGLang": run(SG, "2026-11-04-concurrency-3050oem"),
    "mistral.rs": run(MR, "2026-11-04-concurrency-3050oem"),
}
ACC = {
    "llama.cpp": (run(LL, "2026-10-14-accuracy-fullctx-3050oem"), "gsm8k-e2b-none", "bfcl_simple-e2b-none"),
    "vLLM": (run(VL, "2026-11-03-accuracy-3050oem"), "gsm8k-vllm", "bfcl_simple-vllm"),
    "SGLang": (run(SG, "2026-11-04-accuracy-3050oem"), "gsm8k-sglang", "bfcl_simple-sglang"),
    "mistral.rs": (run(MR, "2026-11-04-accuracy-3050oem"), "gsm8k-mistralrs", "bfcl_simple-mistralrs"),
    "PyTorch": (
        run("local-pytorch-3050oem-2b-q4w4a16", "2026-10-31-accuracy-3050oem"),
        "gsm8k-pytorch",
        "bfcl_simple-pytorch",
    ),
    "LiteRT-LM": (
        run("local-litertlm-3050oem-2b-q4w4a16emb4", "2026-11-02-accuracy-3050oem"),
        "gsm8k-q4exact",
        os.path.join(
            run("local-litertlm-3050oem-2b-int248", "2026-11-03-tools-constraint-3050oem"), "bfcl_simple-q4exact-off"
        ),
    ),
}


def f(v, p=1):
    return "pending" if v is None else f"{v:,.{p}f}"


def score(d, stem):
    path = os.path.join(d, stem + ".jsonl")
    if not os.path.exists(path):
        return None, None, None
    rows = [json.loads(x) for x in open(path)]
    right = sum(bool(r.get("right")) for r in rows)
    try:
        wall = json.loads(open(os.path.join(d, stem + ".log")).read().strip().splitlines()[-1])["wall_s"]
    except (OSError, ValueError, KeyError, IndexError):
        wall = None
    return right / len(rows), len(rows), wall


def conc(d, shape):
    path = os.path.join(d, shape, "concurrency.json")
    if not os.path.exists(path):
        return {}
    return {
        c["concurrency"]: c["output_tokens_per_second"]
        for c in json.load(open(path))["concurrency"]
        if c.get("status", "ok") == "ok"
    }


def peak_mib(d):
    try:
        return max(int(line.split(",")[4].split()[0]) for line in open(os.path.join(d, "gpu.csv")) if line.strip())
    except (OSError, ValueError, IndexError):
        return None


print("#### One request (tok/s, end to end)\n")
print("| engine | no drafter | vs llama.cpp | best drafter | x no drafter | 4 at once | memory after warm-up |")
print("|---|---|---|---|---|---|---|")
base = summarize(HERE)
for name, (label, d, arm, dprefix, ref) in ENGINES.items():
    t = summarize(d) if os.path.exists(os.path.join(d, "results.jsonl")) else {}
    a = t.get(arm, {})
    e2e = a.get("e2e")
    # ratio to llama.cpp: direct on 2026-11-03; through the same-day reference otherwise
    if ref is None:
        vs = e2e / base["llama"]["e2e"] if e2e else None
    else:
        r = t.get(ref, {}).get("e2e")
        vs = e2e / r * base[ref]["e2e"] / base["llama"]["e2e"] if e2e and r else None
    drafted = {k: v["e2e"] for k, v in t.items() if dprefix and k.startswith(dprefix) and v.get("e2e")}
    best = max(drafted, key=drafted.get) if drafted else None
    # LiteRT's drafter runs on Google's int248 file: its ratio is to that file without a drafter
    dbase = t.get(best[: -len("-d")], {}).get("e2e") if best and best.endswith("-d") else e2e
    print(
        f"| {label} | {f(e2e)} | {f(vs, 2)} | "
        + (f"{f(drafted[best])} ({best}) | {f(drafted[best] / dbase, 2)}" if best and dbase else "none measured | -")
        + f" | {f(a.get('batch4'))} | {a.get('mem', 'pending')} |"
    )

print("\n#### Many requests (aggregate output tok/s, no drafter)\n")
print("| engine | 512 in / 128 out: 1 | 8 | 64 | 128 in / 512 out: 1 | 8 | 64 | peak memory |")
print("|---|---|---|---|---|---|---|---|")
for name, d in CONC.items():
    a, b = conc(d, "in512-out128"), conc(d, "in128-out512")
    pm = peak_mib(d)
    print(
        f"| {name} | {f(a.get(1))} | {f(a.get(8))} | {f(a.get(64))} | {f(b.get(1))} | {f(b.get(8))} | "
        f"{f(b.get(64))} | {'pending' if pm is None else f'{pm:,} MiB'} |"
    )
print("| PyTorch, LiteRT-LM | one request at a time | | | | | | |")

print("\n#### Accuracy (one request at a time, greedy, thinking off)\n")
print("| engine | GSM8K (1,319) | BFCL simple (400) | GSM8K wall s |")
print("|---|---|---|---|")
for name, (d, g, b) in ACC.items():
    ga, _, gw = score(d, g)
    ba, _, _ = score(*os.path.split(b)) if os.path.isabs(b) else score(d, b)
    print(f"| {name} | {f(ga, 4)} | {f(ba, 4)} | {f(gw, 0)} |")
