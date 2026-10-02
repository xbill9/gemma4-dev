#!/usr/bin/env python3
"""tables.py <run dir>: markdown tables from the run's JSON (every number is read, none computed by hand)."""
import glob, json, os, re, sys
R = sys.argv[1]
V1 = "/home/xbill/gemma4-dev/tpu-vllm-v5e1-12b-w8a8emb4/benchmarks/runs/2026-09-30-long12b-12b-w8a8-emb4-v5e1/logs/2026-09-30-long12b-v5e1-logs"
lens = sorted((int(d[3:]), d) for d in map(os.path.basename, glob.glob(f"{R}/len*")) if d[3:].isdigit())

def kv(d):
    p = f"{R}/{d}/vllm-container.log"
    t = open(p).read() if os.path.exists(p) else ""
    m = re.search(r"KV cache size: ([\d,]+) tokens", t)
    ok = "Application startup complete" in t
    return (m.group(1) if m else "—"), ok

print("| MAX_MODEL_LEN | Boot | KV cache (tokens) | Full-length requests the pool holds |")
print("|---:|---|---:|---:|")
for L, d in lens:
    k, ok = kv(d)
    full = f"{int(k.replace(',', '')) / L:.2f}" if k != "—" else "—"
    print(f"| {L:,} | {'served' if ok else 'failed'} | {k} | {full} |")
print()
print("| MAX_MODEL_LEN | Prompt tokens | Requests | Output tok/s | Total tok/s | Median TTFT (s) |")
print("|---:|---:|---:|---:|---:|---:|")
for L, d in lens:
    for f in sorted(glob.glob(f"{R}/{d}/loadlong.in*.json"), key=lambda p: [int(x) for x in re.findall(r"\d+", os.path.basename(p))]):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        print(f"| {L:,} | {j['prompt_tokens_per_request']:,} | {j['concurrency']} | {j['output_tok_per_s']:,} | {j['total_tok_per_s']:,} | {j['ttft_median_s']} |")
print()
print("| MAX_MODEL_LEN | Requests | Output tok/s (min-max) |")
print("|---:|---:|---:|")
for L, d in lens:
    for c in (16, 64):
        f = f"{R}/{d}/load.c{c}.json"
        if os.path.exists(f):
            try:
                j = json.load(open(f))
            except Exception:
                continue
            print(f"| {L:,} | {c} | {j['output_tok_per_s']:,} ({j['output_tok_per_s_min']:,}-{j['output_tok_per_s_max']:,}) |")
print()
# v5e-1 comparison at the one prompt length both measured
print("| Prompt tokens | Requests | v5e-1 output tok/s | v5e-4 output tok/s | v5e-4 / v5e-1 | v5e-1 TTFT (s) | v5e-4 TTFT (s) |")
print("|---:|---:|---:|---:|---:|---:|---:|")
for c in (1, 16):
    a = json.load(open(f"{V1}/12b-w8a8-emb4.long3584.c{c}.json"))
    for L, d in lens:
        f = f"{R}/{d}/loadlong.in3584.c{c}.json"
        if os.path.exists(f):
            b = json.load(open(f))
            print(f"| {b['prompt_tokens_per_request']:,} (v5e-1 {a['prompt_tokens_per_request']:,}) | {c} | {a['output_tok_per_s']} | {b['output_tok_per_s']} | {b['output_tok_per_s'] / a['output_tok_per_s']:.2f}x | {a['ttft_median_s']} | {b['ttft_median_s']} |")
print()
# boot timing from each container log
from datetime import datetime
def ts(line):
    m = re.search(r"(\d\d-\d\d \d\d:\d\d:\d\d)", line)
    return datetime.strptime("2026-" + m.group(1), "%Y-%m-%d %H:%M:%S") if m else None
print("| Start | MAX_MODEL_LEN | XLA cache | KV page size | Engine init to KV sized (s) | KV sized to ready (s) | Sum of logged compile steps (s) | First log line to ready (s) |")
print("|---|---:|---|---:|---:|---:|---:|---:|")
starts = []
for f in glob.glob(f"{R}/len*/vllm-container*.log"):
    lines = open(f).read().splitlines()
    first = next((ts(l) for l in lines if ts(l)), None)
    init = next((ts(l) for l in lines if "Initializing a V1 LLM engine" in l), None)
    kvl = next((ts(l) for l in lines if "KV cache size" in l), None)
    ready_line = next((l for l in lines if "Application startup complete" in l), None)
    # the API server's startup line has no timestamp; use the last timestamped line before it
    ready = None
    if ready_line:
        idx = lines.index(ready_line)
        ready = next((ts(l) for l in reversed(lines[:idx]) if ts(l)), None)
    comp = sum(float(x) for x in re.findall(r"finished in ([\d.]+) \[secs\]", "\n".join(lines)))
    page = re.search(r"block_size: (\d+)", "\n".join(lines))
    L = int(re.search(r"len(\d+)", f).group(1))
    cache = "warm (same shapes compiled once before)" if f.split("/")[-2].endswith("-warm") else "mounted, empty for these shapes" if os.path.exists(os.path.join(os.path.dirname(f), "docker-run.txt")) and "xla-cache" in open(os.path.join(os.path.dirname(f), "docker-run.txt")).read() else "none"
    tag = os.path.relpath(f, R)
    d = lambda a, b: f"{(b - a).total_seconds():.0f}" if a and b else "—"
    starts.append((first, f"| `{tag}` | {L:,} | {cache} | {page.group(1) if page else '—'} | {d(init, kvl)} | {d(kvl, ready)} | {comp:.0f} | {d(first, ready)} |"))
for _, row in sorted(starts, key=lambda s: s[0]):
    print(row)
