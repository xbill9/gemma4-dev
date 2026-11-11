"""T4 GSM8K / BFCL per arm, each paired question by question with a reference run of the same engine.

Plain arms: the L4 suite run (gpu-vllm-g6-2b-q4w4a16emb4/.../2026-11-06-suite-g6). LiteRT BFCL ran with its
constrained decoding off, as on the 3050 (2026-11-03-tools-constraint-3050oem), so its BFCL reference is that
3050 run; the L4 ran BFCL with it on. Drafter arms: the same engine without a drafter on the T4 (the L4 ran
other draft lengths). Every arm is also paired with T4 llama.cpp. Paired ranges are a 2,000-draw resample, 95%.
"""
import json, os, random
R = os.path.expanduser("~/gemma4-dev")
L4 = "gpu-vllm-g6-2b-q4w4a16emb4/benchmarks/runs/2026-11-06-suite-g6/acc-{a}/{t}"
LI = "gpu-litertlm-t4-2b-int248/benchmarks/runs/2026-11-06-accuracy-t4/"
T4 = {  # arm: (T4 path, reference label, reference path)
    "vllm": ("gpu-vllm-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-06-accuracy-t4/vllm/{t}-vllm", "L4", L4.replace("{a}", "vllm")),
    "llama": ("gpu-llamacpp-t4-2b-q4_0/benchmarks/runs/2026-11-06-accuracy-t4/llama/{t}-llama", "L4", L4.replace("{a}", "llama")),
    "pytorch": ("gpu-pytorch-t4-2b-q4w4a16/benchmarks/runs/2026-11-06-accuracy-t4/pytorch/{t}-pytorch", "L4", L4.replace("{a}", "pytorch")),
    "sglang4": ("gpu-sglang-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-10-int4-t4/accuracy/sglang4/{t}-sglang4", "L4", L4.replace("{a}", "sglang")),
    "litert": ({"gsm8k": LI + "gsm8k-int248", "bfcl_simple": LI + "bfcl_simple-int248-off"},
               {"gsm8k": "L4", "bfcl_simple": "3050 off"},
               {"gsm8k": L4.replace("{a}", "litert-int248"),
                "bfcl_simple": "local-litertlm-3050oem-2b-int248/benchmarks/runs/2026-11-03-tools-constraint-3050oem/bfcl_simple-int248-off"}),
    "vllm-d3": ("gpu-vllm-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-06-accuracy-t4/vllm-d3/{t}-vllm-d3", "T4 none", "vllm"),
    "llama-d1": ("gpu-llamacpp-t4-2b-q4_0/benchmarks/runs/2026-11-06-accuracy-t4/llama-d1/{t}-llama-d1", "T4 none", "llama"),
}
def pick(v, t): return (v[t] if isinstance(v, dict) else v).replace("{t}", t)
def load(p):
    try: return {r["id"]: r for r in map(json.loads, open(os.path.join(R, p + ".jsonl")))}
    except OSError: return None
def wall(p):
    try: return json.loads(open(os.path.join(R, p + ".log")).read().strip().splitlines()[-1])["wall_s"]
    except Exception: return None
def boot(d):
    rng = random.Random(0); b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000)); return b[50], b[1949]
def paired(x, y):
    ids = sorted(set(x) & set(y)); d = [int(bool(x[i].get("right"))) - int(bool(y[i].get("right"))) for i in ids]
    lo, hi = boot(d); return sum(d) / len(d), lo, hi, sum(v != 0 for v in d)
def acc(r): return sum(bool(x.get("right")) for x in r.values()) / len(r)
def f(x, w, p): return f"{x:{w}.{p}f}" if x is not None else " " * (w - 1) + "-"
for t in ("gsm8k", "bfcl_simple"):
    print(f"\n{t}: one request at a time, thinking off, T4")
    print(f"{'arm':9s} {'n':>5s} {'right':>5s} {'acc':>7s} {'trunc':>5s} {'wall s':>8s}  {'ref':9s} {'ref acc':>7s} {'ref s':>7s} {'time/ref':>8s}  arm-ref paired [95%]      differ   arm - T4 llama [95%]")
    base = load(pick(T4["llama"][0], t))
    for a, (p, lab, rp) in T4.items():
        p, lab = pick(p, t), pick(lab, t)
        rp = pick(T4[rp][0], t) if isinstance(rp, str) and rp in T4 else pick(rp, t)
        r = load(p)
        if not r: print(f"{a:9s} (not run)"); continue
        right = sum(bool(x.get("right")) for x in r.values()); trunc = sum(x.get("finish") == "length" for x in r.values())
        rr, w, wr = load(rp), wall(p), wall(rp)
        m, lo, hi, nd = paired(r, rr)
        sl = "" if a == "llama" else "   {:+.4f} [{:+.4f}, {:+.4f}]".format(*paired(r, base)[:3])
        print(f"{a:9s} {len(r):5d} {right:5d} {right/len(r):7.4f} {trunc:5d} {f(w,8,1)}  {lab:9s} {acc(rr):7.4f} {f(wr,7,1)} {f(w/wr if w and wr else None,8,3)}  {m:+.4f} [{lo:+.4f}, {hi:+.4f}] {nd:5d}{sl}")
