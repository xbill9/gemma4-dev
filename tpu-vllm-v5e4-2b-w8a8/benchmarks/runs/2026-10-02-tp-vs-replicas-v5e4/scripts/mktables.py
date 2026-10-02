"""Print the README tables for 2026-10-02-tp-vs-replicas-v5e4 from its JSON and logs. No hand arithmetic."""

import datetime
import json
import os
import re
import sys

O = sys.argv[1]


def J(f):
    p = os.path.join(O, f)
    return json.load(open(p)) if os.path.exists(p) else None


def load(label, c):
    d = J(f"{label}.load.c{c}.json")
    return d["output_tok_per_s"] if d else None


def longd(label):
    return J(f"{label}.loadlong.c16.json") or {}


def kv(log):
    m = re.findall(r"KV cache size: ([\d,]+) tokens", open(os.path.join(O, log)).read())
    return m[0] if m else "?"


def init(log):
    return re.findall(r"init engine .* took ([\d.]+) s \(compilation: ([\d.]+) s\)", open(os.path.join(O, log)).read())


def f(x):
    return "—" if x is None else f"{x:,.1f}"


base = {c: load("tp1", c) for c in (1, 4, 16, 64)}
ROWS = [
    ("TP=4, one server (rig boot)", "tp4", "4", "1", "vllm-tp4.log"),
    ("TP=2, one server, chips 0-1", "tp2", "2", "1", "vllm-tp2.log"),
    ("TP=1, one server, chip 0", "tp1", "1", "1", "vllm-tp1.log"),
    ("4 x TP=1, ports 8000-8003", "rep4", "1", "4", "vllm-rep4-r0.log"),
    ("`--data-parallel-size 4`, one port", "dp4", "1", "4", "vllm-dp4.log"),
    ("`--data-parallel-size 4`, second start", "dp4-rerun", "1", "4", "vllm-dp4-cache2.log"),
    ("`--data-parallel-size 4 --max-model-len 8192`", "dp4-len8k", "1", "4", "vllm-dp4-len8k.log"),
]
print("#### Layouts, output tok/s by concurrent requests")
print()
print(
    "| Layout | Chips per engine | Engines | KV tokens per engine | 1 | 4 | 16 | 64 | 256 "
    "| Long 16 x 2,048: output tok/s | total tok/s | TTFT median s |"
)
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for name, lab, cp, n, log in ROWS:
    ld = longd(lab)
    print(
        f"| {name} | {cp} | {n} | {kv(log)} | "
        + " | ".join(f(load(lab, c)) for c in (1, 4, 16, 64, 256))
        + f" | {f(ld.get('output_tok_per_s'))} | {f(ld.get('total_tok_per_s'))} | {ld.get('ttft_median_s', '—')} |"
    )
print()
for c in (16, 64, 256):
    print(
        f"At {c} requests: DP=4 / TP=4 = {load('dp4', c) / load('tp4', c):.2f}x; "
        f"DP=4 at 8,192 / TP=4 = {load('dp4-len8k', c) / load('tp4', c):.2f}x.  "
    )
print()
print("#### Ratio to TP=1 on one chip, same VM, same arguments")
print()
print("| Layout | 1 | 4 | 16 | 64 | Long 16 |")
print("|---|---:|---:|---:|---:|---:|")
for name, lab, *_ in ROWS[:5]:
    r = [f"{load(lab, c) / base[c]:.2f}x" for c in (1, 4, 16, 64)]
    r.append(f"{longd(lab)['output_tok_per_s'] / longd('tp1')['output_tok_per_s']:.2f}x")
    print(f"| {name} | " + " | ".join(r) + " |")
print()
print(
    f"TP=4 relaunched (same arguments plus profiler and compile cache), 16 requests: "
    f"{f(load('tp4-relaunch', 16))}; at boot {f(load('tp4', 16))}."
)
print()
print("#### One chip, TP=1: `--max-model-len` and `--max-num-seqs`")
print()
VARS = [
    ("16,384", "256 (default)", "tp1", "vllm-tp1.log"),
    ("8,192", "256 (default)", "len8k", "vllm-tp1-len8k.log"),
    ("4,096", "256 (default)", "len4k", "vllm-tp1-len4k.log"),
    ("2,048", "256 (default)", "len2k", "vllm-tp1-len2k.log"),
    ("16,384", "16", "seqs16", "vllm-tp1-seqs16.log"),
    ("2,048", "16", "jev", "vllm-tp1-jev.log"),
]
print(
    "| `--max-model-len` | `--max-num-seqs` | KV tokens | 1 | 4 | 16 | 64 | Long 16 x 2,048: output tok/s | 16 vs 16,384/256 |"
)
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for ml, ns, lab, log in VARS:
    ld = longd(lab)
    lo = f(ld["output_tok_per_s"]) if "output_tok_per_s" in ld else ld.get("status", "—")
    print(
        f"| {ml} | {ns} | {kv(log)} | "
        + " | ".join(f(load(lab, c)) for c in (1, 4, 16, 64))
        + f" | {lo} | {load(lab, 16) / base[16]:.2f}x |"
    )
print()
print("#### Profiles: one burst of 16 requests x 128 output tokens, per chip")
print()
bursts = {json.loads(x)["label"]: json.loads(x) for x in open(os.path.join(O, "profile-bursts.jsonl"))}
COLL = (
    "all-reduce",
    "all-gather",
    "collective-permute-start",
    "collective-permute-done",
    "all-to-all",
    "reduce-scatter",
)
print(
    "| Server | Chips | Burst wall s | Device busy ms | Attention decode kernel ms | Matmul fusions ms "
    "| Collectives ms | Collectives share |"
)
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for lab, pf in (
    ("tp1", "hlo-stats-tp1.json"),
    ("tp2", "hlo-stats-tp2.json"),
    ("tp4", "hlo-stats-tp4.json"),
    ("tp1-maxlen2048", "hlo-stats-tp1-maxlen2048.json"),
):
    d = J(pf)
    cat = d["by_category_ms_per_device"]
    tot = d["self_time_ms_per_device"]
    coll = sum(cat.get(k, 0) for k in COLL)
    print(
        f"| {lab} | {d['devices']} | {bursts[lab]['wall_s']} | {tot:,.1f} | {d['top_ops_ms_per_device']['RPAd']['ms']:,.1f} | "
        f"{cat.get('convolution fusion', 0):,.1f} | {coll:,.1f} | {coll / tot:.1%} |"
    )
print()
ak = {}
for blob in re.findall(r"\{.*?\n\}", open(os.path.join(O, "attention-kernels.json")).read(), re.S):
    ak.update(json.loads(blob))
print("| Server | Attention decode kernel variant | ms per chip | calls per chip |")
print("|---|---|---:|---:|")
for lab, ndev in (("tp1", 1), ("tp4", 4), ("len2k", 1)):
    for k, v in ak[lab].items():
        if k.startswith("RPAd"):
            print(
                f"| {lab} | `{k}` | {v['self_ms_all_devices'] / ndev:,.1f} | {v['occurrences_all_devices'] / ndev:,.0f} |"
            )
print()
print("#### Engine start: `init engine` seconds from vLLM's log")
print()
print("| Container | Compile cache | init engine s | of which compilation s |")
print("|---|---|---:|---:|")
for name, log, cache in (
    ("TP=4 boot", "vllm-tp4.log", "off"),
    ("TP=2", "vllm-tp2.log", "off"),
    ("TP=1", "vllm-tp1.log", "off"),
    ("4 x TP=1, replica 0", "vllm-rep4-r0.log", "off"),
    ("DP=4", "vllm-dp4.log", "off"),
    ("TP=4 relaunch", "vllm-tp4-relaunch.log", "on, empty"),
    ("TP=1, max-model-len 2048", "vllm-tp1-len2k.log", "on, other shapes only"),
    ("TP=1, max-num-seqs 16", "vllm-tp1-seqs16.log", "on, other shapes only"),
    ("TP=1, 2048 and 16", "vllm-tp1-jev.log", "on, other shapes only"),
    ("TP=1, max-model-len 4096", "vllm-tp1-len4k.log", "on, other shapes only"),
    ("TP=1, max-model-len 8192", "vllm-tp1-len8k.log", "on, other shapes only"),
    ("DP=4, first start with cache", "vllm-dp4-cache1.log", "on, other shapes only"),
    ("DP=4, second start", "vllm-dp4-cache2.log", "on, holds this layout"),
    ("DP=4, max-model-len 8192", "vllm-dp4-len8k.log", "on, holds TP=1 at 8,192"),
):
    m = init(log)
    a = " / ".join(x for x, _ in m)
    b = " / ".join(y for _, y in m)
    print(f"| {name}{' (each engine)' if len(m) > 1 else ''} | {cache} | {a} | {b} |")

qr = J("queued-resource.json")
t0 = datetime.datetime.fromisoformat(qr["createTime"][:26].rstrip("Z") + "+00:00")
t1 = datetime.datetime(2026, 10, 2, 23, 53, 11, tzinfo=datetime.timezone.utc)  # delete issued
h = (t1 - t0).total_seconds() / 3600
print(
    f"\nQueued Resource created {qr['createTime']}, delete issued {t1.isoformat()}: {h:.2f} h, {h * 2.40:.2f} USD at 2.40 USD/h."
)
