import json, os, random, sys
R = os.path.expanduser("~/gemma4-dev")
T4 = {"vllm": "gpu-vllm-t4-2b-q4w4a16emb4/benchmarks/runs/2026-11-06-accuracy-t4/vllm/{t}-vllm",
      "llama": "gpu-llamacpp-t4-2b-q4_0/benchmarks/runs/2026-11-06-accuracy-t4/llama/{t}-llama",
      "pytorch": "gpu-pytorch-t4-2b-q4w4a16/benchmarks/runs/2026-11-06-accuracy-t4/pytorch/{t}-pytorch"}
L4 = "gpu-vllm-g6-2b-q4w4a16emb4/benchmarks/runs/2026-11-06-suite-g6/acc-{a}/{t}"
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
for t in ("gsm8k", "bfcl_simple"):
    print(f"\n{t}: one request at a time, thinking off")
    print(f"{'arm':8s} {'n':>5s} {'right':>5s} {'acc':>7s} {'trunc':>5s} {'wall s':>8s}  {'L4 acc':>7s} {'L4 s':>7s} {'T4/L4 time':>10s}  T4-L4 paired [95%]       differ")
    for a, p in T4.items():
        p = p.format(t=t); r = load(p)
        if not r: print(f"{a:8s} (not run)"); continue
        right = sum(bool(x.get("right")) for x in r.values()); trunc = sum(x.get("finish") == "length" for x in r.values())
        w = wall(p); lp = L4.format(a=a, t=t); rl = load(lp); wl = wall(lp)
        accl = sum(bool(x.get("right")) for x in rl.values()) / len(rl)
        m, lo, hi, nd = paired(r, rl)
        print(f"{a:8s} {len(r):5d} {right:5d} {right/len(r):7.4f} {trunc:5d} {w:8.1f}  {accl:7.4f} {wl:7.1f} {w/wl:10.3f}  {m:+.4f} [{lo:+.4f}, {hi:+.4f}] {nd:5d}")
