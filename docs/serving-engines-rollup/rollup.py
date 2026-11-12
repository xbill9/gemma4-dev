#!/usr/bin/env python3
"""Cross-platform roll-up of the four Gemma 4 E2B serving-engine ladders.

Reads only the copies in evidence/ (each ladder's own summary files) and writes
evidence/rollup.txt. Every figure the article states across platforms is either
read here from a source line or computed here; nothing is typed in by hand.

    python3 rollup.py
"""

from __future__ import annotations

import re
from pathlib import Path

EV = Path(__file__).resolve().parent / "evidence"
PLATFORMS = ["3050", "T4", "L4", "v5e-1"]


def num(s: str) -> float:
    return float(s.replace(",", ""))


def arms(path: str) -> dict[str, float]:
    """First 'arm  e2e ...' value per arm in an engines summary (one-request table)."""
    out: dict[str, float] = {}
    for line in (EV / path).read_text().splitlines():
        t = line.split()
        if len(t) >= 2 and re.fullmatch(r"[a-z][a-z0-9-]*", t[0]) and re.fullmatch(r"\d+\.\d", t[1]):
            out.setdefault(t[0], float(t[1]))
    return out


def block(path: str, start: str) -> list[str]:
    """Lines after the first line containing `start`, up to the next blank line."""
    lines = (EV / path).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if start in l)
    out = []
    for l in lines[i + 1 :]:
        if not l.strip():
            if out:
                break
            continue
        out.append(l)
    return out


def sweep_rows(path: str, start: str) -> dict[str, list[float]]:
    """arm -> 7 levels (1..64) from a whitespace sweep table."""
    out = {}
    for l in block(path, start):
        t = l.split()
        if len(t) >= 8 and all(re.fullmatch(r"[\d.]+", x) for x in t[1:8]):
            out[t[0]] = [float(x) for x in t[1:8]]
    return out


def md_rows(path: str, start: str) -> list[list[str]]:
    rows = []
    for l in block(path, start):
        if l.startswith("|") and not set(l) <= set("|-: "):
            rows.append([c.strip() for c in l.strip().strip("|").split("|")])
    return rows


def acc(path: str, section: str, arm: str) -> float:
    for l in block(path, section):
        t = l.split()
        if t and t[0] == arm:
            return float(t[3])
    raise KeyError((path, section, arm))


LEVELS = [1, 2, 4, 8, 16, 32, 64]
out: list[str] = []
p = out.append

# ---------------------------------------------------------------- one request
a3050 = (
    arms("3050-engines-summary.txt")
    | arms("3050-sglang-engines-summary.txt")
    | arms("3050-mistralrs-engines-summary.txt")
)
# the 3050 vLLM and llama.cpp arms come from the main summary (the others ran beside a same-day reference)
a3050["vllm"], a3050["vllm-d4"] = arms("3050-engines-summary.txt")["vllm"], arms("3050-engines-summary.txt")["vllm-d4"]
a3050["llama"], a3050["llama-d2"] = (
    arms("3050-engines-summary.txt")["llama"],
    arms("3050-engines-summary.txt")["llama-d2"],
)
aL4 = arms("l4-engines-summary.txt")
aT4 = arms("t4-engines-summary.txt")
aT4sg = arms("t4-sglang-int4-summary.txt")
aV = arms("v5e-ladder-summary.txt")
aVsg = arms("v5e-sglangjax-summary.txt")

ONE = {  # engine -> platform -> (no drafter, best drafter, best arm)
    "vLLM": {
        "3050": (a3050["vllm"], a3050["vllm-d4"], "4"),
        "T4": (aT4["vllm"], aT4["vllm-d3"], "3"),
        "L4": (aL4["vllm"], aL4["vllm-d4"], "4"),
        "v5e-1": (aV["vllm"], None, "out of memory"),
    },
    "SGLang": {
        "3050": (a3050["sglang"], a3050["sglang-d4"], "4"),
        "T4": (aT4sg["sglang4"], aT4sg["sglang4-d4"], "4"),
        "L4": (aL4["sglang"], aL4["sglang-d4"], "4"),
        "v5e-1": (aVsg["sglang"], None, "SGLang-JAX, none"),
    },
    "llama.cpp": {
        "3050": (a3050["llama"], a3050["llama-d2"], "2"),
        "T4": (aT4["llama"], aT4["llama-d1"], "1"),
        "L4": (aL4["llama"], aL4["llama-d2"], "2"),
        "v5e-1": (None, None, "no TPU backend"),
    },
    "mistral.rs": {
        "3050": (a3050["mistralrs"], a3050["mistralrs-d1"], "1"),
        "T4": (aT4["mistralrs"], None, "drafter fails"),
        "L4": (aL4["mistralrs"], aL4["mistralrs-d1"], "1"),
        "v5e-1": (None, None, "no TPU backend"),
    },
    "PyTorch": {
        "3050": (a3050["pytorch"], None, ""),
        "T4": (aT4["pytorch"], None, ""),
        "L4": (aL4["pytorch"], None, ""),
        "v5e-1": (aV["torchtpu-bf16"], None, "TorchTPU"),
    },
    "LiteRT-LM": {
        "3050": (a3050["litert-int248"], a3050["litert-int248-d"], "own"),
        "T4": (aT4["litert-int248"], aT4["litert-int248-d"], "own"),
        "L4": (aL4["litert-int248"], aL4["litert-int248-d"], "own"),
        "v5e-1": (None, None, ""),
    },
    "JAX (bf16)": {"v5e-1": (aV["jax-bf16"], None, "")},
    "Rust over PJRT": {"v5e-1": (aV["jaxrust"], None, "")},
}

p("ONE REQUEST, tok/s end to end (six prompts, 256 tokens, geometric mean of medians)")
p(f"{'engine':<16}" + "".join(f"{pl:>22}" for pl in PLATFORMS))
for eng, d in ONE.items():
    cells = []
    for pl in PLATFORMS:
        nd, bd, n = d.get(pl, (None, None, ""))
        if nd is None:
            cells.append(f"{(n or '-'):>22}")
        elif bd is None:
            cells.append(f"{nd:>22.1f}")
        else:
            cells.append(f"{f'{nd:.1f} / {bd:.1f} ({n})':>22}")
    p(f"{eng:<16}" + "".join(cells))
p("(cell = no drafter / best drafter (drafted tokens); v5e-1 SGLang = SGLang-JAX bf16, PyTorch = TorchTPU bf16)")
p("")

p("DRAFTER GAIN for one request (best drafter / no drafter)")
for eng in ["vLLM", "llama.cpp", "SGLang", "mistral.rs", "LiteRT-LM"]:
    row = []
    for pl in ["3050", "T4", "L4"]:
        nd, bd, n = ONE[eng][pl]
        row.append(f"{pl} {bd / nd:.2f}x" if bd else f"{pl} {n}")
    p(f"  {eng:<12} " + "   ".join(row))
p("")

CUDA = ["vLLM", "SGLang", "llama.cpp", "mistral.rs", "PyTorch"]
p("ONE REQUEST, NO DRAFTER: range over the five CUDA engines")
for pl in ["3050", "T4", "L4"]:
    v = [ONE[e][pl][0] for e in CUDA]
    p(f"  {pl:<6} {min(v):.1f} to {max(v):.1f}  max/min {max(v) / min(v):.3f}")
v = [ONE[e]["v5e-1"][0] for e in ["vLLM", "SGLang", "JAX (bf16)", "Rust over PJRT"]]
p(f"  v5e-1  {min(v):.1f} to {max(v):.1f}  max/min {max(v) / min(v):.3f}  (vLLM, SGLang-JAX, JAX bf16, Rust over PJRT)")
p("LiteRT-LM (Google's int2/4/8 file) against llama.cpp, no drafter:")
for pl in ["3050", "T4", "L4"]:
    p(f"  {pl:<6} {ONE['LiteRT-LM'][pl][0] / ONE['llama.cpp'][pl][0]:.3f}")
p("")

p("FASTEST ONE REQUEST per platform (any engine, any drafter)")
for pl in PLATFORMS:
    best = max(
        ((max(x for x in d[pl][:2] if x), e) for e, d in ONE.items() if pl in d and d[pl][0]), key=lambda z: z[0]
    )
    p(f"  {pl:<6} {best[1]} {best[0]:.1f}")
p("")

p("SAME ENGINE, OTHER CHIP: one request, no drafter, against the L4")
for eng in ["vLLM", "SGLang", "llama.cpp", "mistral.rs", "PyTorch"]:
    r = [f"{pl} {ONE[eng][pl][0] / ONE[eng]['L4'][0]:.3f}" for pl in ["3050", "T4"]]
    p(f"  {eng:<12} " + "   ".join(r))
p(f"  vLLM         v5e-1 {ONE['vLLM']['v5e-1'][0] / ONE['vLLM']['L4'][0]:.3f}")
p("with each engine's best drafter, against the L4:")
for eng in ["vLLM", "llama.cpp", "SGLang"]:
    r = [f"{pl} {ONE[eng][pl][1] / ONE[eng]['L4'][1]:.3f}" for pl in ["3050", "T4"]]
    p(f"  {eng:<12} " + "   ".join(r))
p("")


# ---------------------------------------------------------------- load
def b3050(shape_hdr: str) -> dict[str, tuple[float, str]]:
    rows = md_rows("3050-best-per-level.md", shape_hdr)
    hdr = rows[0]
    r64 = next(r for r in rows if r[0] == "64")
    out = {}
    for name, cell in zip(hdr[1:], r64[1:]):
        m = re.match(r"([\d,]+) \((.+)\)", cell)
        out[name] = (num(m.group(1)), m.group(2))
    return out


def pt3050_nodraft() -> dict[str, tuple[float, float]]:
    rows = md_rows("3050-paper-table.md", "Many requests")
    out = {}
    for r in rows[1:]:
        if r[1].replace(",", "").replace(".", "").isdigit():
            out[r[0]] = (num(r[3]), num(r[6]))  # 512/128 at 64, 128/512 at 64
    return out


L4_5 = sweep_rows("l4-sweep.txt", "in512-out128: output tok/s")
L4_1 = sweep_rows("l4-sweep.txt", "in128-out512: output tok/s")
T4_5 = sweep_rows("t4-ladder.txt", "in512-out128: output tok/s")
T4_1 = sweep_rows("t4-ladder.txt", "in128-out512: output tok/s")
V_5 = sweep_rows("v5e-ladder-summary.txt", "in512-out128: output tok/s")
V_1 = sweep_rows("v5e-ladder-summary.txt", "in128-out512: output tok/s")


def sgj(start: str) -> dict[int, float]:
    out = {}
    for l in block("v5e-sglangjax-summary.txt", start):
        t = l.split()
        if not (t and t[0].isdigit()):
            break  # the next shape's header line
        out[int(t[0])] = float(t[1])
    return out


SGJ_5, SGJ_1 = sgj("in512-out128:"), sgj("in128-out512:")
n3050 = pt3050_nodraft()
b3050_1, b3050_5 = b3050("128 in / 512 out"), b3050("512 in / 128 out")


def best64(rows: dict[str, list[float]], prefix: str) -> tuple[float, str]:
    c = [(v[-1], a) for a, v in rows.items() if a == prefix or a.startswith(prefix + "-d")]
    return max(c)


LOAD = {  # engine -> platform -> {shape: (no drafter, best, best arm)}
    "vLLM": {
        "3050": {"128/512": (n3050["vLLM"][1], *b3050_1["vLLM"]), "512/128": (n3050["vLLM"][0], *b3050_5["vLLM"])},
        "T4": {
            "128/512": (T4_1["vllm"][-1], *best64(T4_1, "vllm")),
            "512/128": (T4_5["vllm"][-1], *best64(T4_5, "vllm")),
        },
        "L4": {
            "128/512": (L4_1["vllm"][-1], *best64(L4_1, "vllm")),
            "512/128": (L4_5["vllm"][-1], *best64(L4_5, "vllm")),
        },
        "v5e-1": {
            "128/512": (V_1["vllm"][-1], V_1["vllm"][-1], "vllm"),
            "512/128": (V_5["vllm"][-1], V_5["vllm"][-1], "vllm"),
        },
    },
    "SGLang": {
        "3050": {
            "128/512": (n3050["SGLang"][1], *b3050_1["SGLang"]),
            "512/128": (n3050["SGLang"][0], *b3050_5["SGLang"]),
        },
        "T4": {
            "128/512": (T4_1["sglang4"][-1], T4_1["sglang4"][-1], "sglang4"),
            "512/128": (T4_5["sglang4"][-1], T4_5["sglang4"][-1], "sglang4"),
        },
        "L4": {
            "128/512": (L4_1["sglang"][-1], *best64(L4_1, "sglang")),
            "512/128": (L4_5["sglang"][-1], *best64(L4_5, "sglang")),
        },
        "v5e-1": {"128/512": (SGJ_1[64], SGJ_1[64], "SGLang-JAX"), "512/128": (SGJ_5[64], SGJ_5[64], "SGLang-JAX")},
    },
    "llama.cpp": {
        "3050": {
            "128/512": (n3050["llama.cpp"][1], *b3050_1["llama.cpp"]),
            "512/128": (n3050["llama.cpp"][0], *b3050_5["llama.cpp"]),
        },
        "T4": {
            "128/512": (T4_1["llama"][-1], *best64(T4_1, "llama")),
            "512/128": (T4_5["llama"][-1], *best64(T4_5, "llama")),
        },
        "L4": {
            "128/512": (L4_1["llama"][-1], *best64(L4_1, "llama")),
            "512/128": (L4_5["llama"][-1], *best64(L4_5, "llama")),
        },
    },
    "mistral.rs": {
        "3050": {
            "128/512": (n3050["mistral.rs"][1], *b3050_1["mistral.rs"]),
            "512/128": (n3050["mistral.rs"][0], *b3050_5["mistral.rs"]),
        },
        "T4": {
            "128/512": (T4_1["mistralrs"][-1], T4_1["mistralrs"][-1], "mistralrs"),
            "512/128": (T4_5["mistralrs"][-1], T4_5["mistralrs"][-1], "mistralrs"),
        },
        "L4": {
            "128/512": (L4_1["mistralrs-card"][-1], L4_1["mistralrs-card"][-1], "mistralrs-card"),
            "512/128": (L4_5["mistralrs-card"][-1], L4_5["mistralrs-card"][-1], "mistralrs-card"),
        },
    },
}

for shape in ["128/512", "512/128"]:
    p(f"64 REQUESTS, {shape} (in/out), aggregate output tok/s: no drafter / best setting")
    p(f"{'engine':<12}" + "".join(f"{pl:>30}" for pl in PLATFORMS))
    for eng, d in LOAD.items():
        cells = []
        for pl in PLATFORMS:
            if pl not in d:
                cells.append(f"{'-':>30}")
                continue
            nd, bv, ba = d[pl][shape]
            cells.append(f"{f'{nd:,.1f} / {bv:,.1f} ({ba})':>30}")
        p(f"{eng:<12}" + "".join(cells))
    p("leader at 64 requests (best setting), and leader / runner-up:")
    for pl in PLATFORMS:
        c = sorted(((d[pl][shape][1], e) for e, d in LOAD.items() if pl in d), reverse=True)
        p(f"  {pl:<6} {c[0][1]} {c[0][0]:,.1f}   x{c[0][0] / c[1][0]:.3f} {c[1][1]} {c[1][0]:,.1f}")
    p("")

p("vLLM DRAFTER AT 64 REQUESTS (best drafted setting / no drafter)")
for pl in ["3050", "T4", "L4"]:
    r = [
        f"{s} {LOAD['vLLM'][pl][s][1] / LOAD['vLLM'][pl][s][0]:.3f} = "
        f"{LOAD['vLLM'][pl][s][1] / LOAD['vLLM'][pl][s][0]:.2f}x ({LOAD['vLLM'][pl][s][2]})"
        for s in ["128/512", "512/128"]
    ]
    p(f"  {pl:<6} " + "   ".join(r))
p("")

p("vLLM UNDER LOAD, OTHER CHIP: 64 requests, no drafter, against the L4")
for pl in ["3050", "T4", "v5e-1"]:
    r = [f"{s} {LOAD['vLLM'][pl][s][0] / LOAD['vLLM']['L4'][s][0]:.3f}" for s in ["128/512", "512/128"]]
    p(f"  {pl:<6} " + "   ".join(r))
p("vLLM scaling, 64 requests / 1 request, 128/512, no drafter:")
s1 = {"3050": None, "T4": T4_1["vllm"], "L4": L4_1["vllm"], "v5e-1": V_1["vllm"]}
p(f"  3050   {n3050['vLLM'][1] / num(md_rows('3050-paper-table.md', 'Many requests')[2][4]):.2f}x")
for pl in ["T4", "L4", "v5e-1"]:
    p(f"  {pl:<6} {s1[pl][-1] / s1[pl][0]:.2f}x")
p("")

# ---------------------------------------------------------------- accuracy
pt = {r[0]: r for r in md_rows("3050-paper-table.md", "Accuracy")[1:]}
GSM = {
    "3050": {
        "llama.cpp": num(pt["llama.cpp"][1]),
        "vLLM": num(pt["vLLM"][1]),
        "SGLang": num(pt["SGLang"][1]),
        "mistral.rs": num(pt["mistral.rs"][1]),
        "PyTorch": num(pt["PyTorch"][1]),
    },
    "T4": {
        e: acc("t4-accuracy.txt", "gsm8k:", a)
        for e, a in [
            ("llama.cpp", "llama"),
            ("vLLM", "vllm"),
            ("SGLang", "sglang4"),
            ("mistral.rs", "mistralrs"),
            ("PyTorch", "pytorch"),
        ]
    },
    "L4": {
        e: acc("l4-accuracy.txt", "gsm8k:", a)
        for e, a in [
            ("llama.cpp", "llama"),
            ("vLLM", "vllm"),
            ("SGLang", "sglang"),
            ("mistral.rs", "mistralrs"),
            ("PyTorch", "pytorch"),
        ]
    },
}
BFCL = {
    "3050": {e: num(pt[e][2]) for e in ["llama.cpp", "vLLM", "SGLang", "mistral.rs", "PyTorch"]},
    "T4": {
        e: acc("t4-accuracy.txt", "bfcl_simple:", a)
        for e, a in [
            ("llama.cpp", "llama"),
            ("vLLM", "vllm"),
            ("SGLang", "sglang4"),
            ("mistral.rs", "mistralrs"),
            ("PyTorch", "pytorch"),
        ]
    },
    "L4": {
        e: acc("l4-accuracy.txt", "bfcl_simple:", a)
        for e, a in [
            ("llama.cpp", "llama"),
            ("vLLM", "vllm"),
            ("SGLang", "sglang"),
            ("mistral.rs", "mistralrs"),
            ("PyTorch", "pytorch"),
        ]
    },
}
VGSM = {
    "vLLM (int4)": acc("v5e-ladder-summary.txt", "gsm8k:", "vllm"),
    "vLLM (bf16 tables)": acc("v5e-ladder-summary.txt", "gsm8k:", "vllm-text"),
    "JAX (W4A16)": acc("v5e-ladder-summary.txt", "gsm8k:", "jax"),
    "JAX (bf16)": acc("v5e-ladder-summary.txt", "gsm8k:", "jax-bf16"),
    "Rust over PJRT (bf16)": acc("v5e-ladder-summary.txt", "gsm8k:", "jaxrust"),
}
m = re.search(r"sglang: \d+/\d+ = ([\d.]+)", (EV / "v5e-sglangjax-summary.txt").read_text())
VGSM["SGLang-JAX (bf16)"] = float(m.group(1))

p("GSM8K (1,319) and BFCL simple (400), one request at a time, greedy, thinking off, no drafter")
p(f"{'engine':<12}" + "".join(f"{pl:>18}" for pl in ["3050", "T4", "L4"]))
for e in CUDA:
    p(f"{e:<12}" + "".join(f"{f'{GSM[pl][e]:.4f} / {BFCL[pl][e]:.4f}':>18}" for pl in ["3050", "T4", "L4"]))
allg = [v for pl in GSM.values() for v in pl.values()]
allb = [v for pl in BFCL.values() for v in pl.values()]
p(
    f"five CUDA engines x three GPUs: {len(allg)} GSM8K runs {min(allg):.4f} to {max(allg):.4f} (spread {max(allg) - min(allg):.4f}); "
    f"BFCL {min(allb):.4f} to {max(allb):.4f} (spread {max(allb) - min(allb):.4f}, {round((max(allb) - min(allb)) * 400)} questions)"
)
p("v5e-1 GSM8K:")
for k, v in VGSM.items():
    p(f"  {k:<24} {v:.4f}")
p(f"v5e-1 BFCL, vLLM (only engine returning tool calls): {acc('v5e-ladder-summary.txt', 'bfcl_simple:', 'vllm'):.4f}")
lt = {
    "3050": acc("l4-accuracy.txt", "gsm8k:", "litert-int248"),
    "T4": acc("t4-accuracy.txt", "gsm8k:", "litert"),
    "L4": acc("l4-accuracy.txt", "gsm8k:", "litert-int248"),
}
# the L4 table's '3050 acc' column (field 7) holds the 3050 run of the same arm
for l in block("l4-accuracy.txt", "gsm8k:"):
    t = l.split()
    if t and t[0] == "litert-int248":
        lt["3050"] = float(t[7])
p("LiteRT-LM, Google's int2/4/8 file, GSM8K: " + ", ".join(f"{pl} {lt[pl]:.4f}" for pl in ["3050", "T4", "L4"]))
p(
    f"  below the lowest CUDA engine on the same card: "
    + ", ".join(f"{pl} {lt[pl] - min(GSM[pl].values()):+.4f}" for pl in ["3050", "T4", "L4"])
)
p("")

(EV / "rollup.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
