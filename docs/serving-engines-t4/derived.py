"""Ratios quoted in devto-serving-engines-t4.md, computed from the evidence files. Writes evidence/derived.txt."""
import json, os, re
H = os.path.dirname(os.path.abspath(__file__)); EV = os.path.join(H, "evidence")
def summary(f):
    return {l.split()[0]: l.split() for l in open(os.path.join(EV, f)).read().splitlines()[1:] if l.strip()}
S = summary("engines-summary.txt"); S4 = summary("sglang-int4-summary.txt")
tps = lambda t, a: float(t[a][1])
def ladder(shape):
    out, cur = {}, None
    for l in open(os.path.join(EV, "ladder.txt")):
        if l.startswith(shape + ": output tok/s"): cur = True; continue
        if cur and not l.strip(): 
            if out: break
            continue
        if cur and not l.startswith("arm"): f = l.split(); out[f[0]] = [float(x) for x in f[1:]]
    return out  # levels 1,2,4,8,16,32,64
L = ladder("in128-out512"); P = ladder("in512-out128")
wall = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(\S+)\s+1319\s+\d+\s+\S+\s+\d+\s+([\d.]+)", open(os.path.join(EV, "accuracy.txt")).read(), re.M)}
think = json.loads(re.search(r"END bfcl_simple rc=0 (\{.*\})", open(os.path.join(EV, "mistralrs-thinkdefault-driver.log")).read()).group(1))
off = float(re.search(r"^mistralrs\s+400\s+\d+\s+\S+\s+\d+\s+([\d.]+)", open(os.path.join(EV, "accuracy.txt")).read(), re.M).group(1))
l4 = [float(l.split()[3]) for l in open(os.path.join(EV, "compare-l4.txt")).read().splitlines()[1:]]
r = lambda x: f"{x:.2f}x"
rows = [
 ("vLLM 3 drafted / vLLM, one request", r(tps(S, "vllm-d3") / tps(S, "vllm"))),
 ("llama.cpp 1 drafted / llama.cpp, one request", r(tps(S, "llama-d1") / tps(S, "llama"))),
 ("SGLang 4-bit 4 drafted / none, one request", r(tps(S4, "sglang4-d4") / tps(S4, "sglang4"))),
 ("LiteRT drafter / none, one request", r(tps(S, "litert-int248-d") / tps(S, "litert-int248"))),
 ("SGLang 4-bit / 16-bit, one request", r(tps(S4, "sglang4") / tps(S4, "sglang"))),
 ("vLLM-d1 / vLLM at 64, 128/512", r(L["vllm-d1"][6] / L["vllm"][6])),
 ("vLLM-d1 / vLLM at 64, 512/128", r(P["vllm-d1"][6] / P["vllm"][6])),
 ("llama.cpp / vLLM at 64, 512/128", r(P["llama"][6] / P["vllm"][6])),
 ("llama.cpp / vLLM at 64, 128/512", r(L["llama"][6] / L["vllm"][6])),
 ("mistral.rs 128/512, levels 4-64, min-max", f"{min(L['mistralrs'][2:]):.1f}-{max(L['mistralrs'][2:]):.1f}, about {round(sum(L['mistralrs'][2:])/5)} tokens"),
 ("GSM8K time vLLM 3 drafted / llama.cpp", r(wall["vllm-d3"] / wall["llama"])),
 ("GSM8K time vLLM 3 drafted / vLLM", r(wall["vllm-d3"] / wall["vllm"])),
 ("mistral.rs BFCL thinking default / off, time", r(think["wall_s"] / off)),
 ("T4/L4 no drafter, CUDA engines, min-max", f"{min(l4[:5]):.2f}x to {max(l4[:5]):.2f}x"),
 ("T4/L4 with drafter, min-max", f"{min(l4[6:]):.2f}x to {max(l4[6:]):.2f}x"),
 ("SGLang 4-bit with Marlin atomic-add on, prompts that repeat", S4["sglang4-atomic"][-1].replace("/", " of ")),
]
with open(os.path.join(EV, "derived.txt"), "w") as f:
    for k, v in rows: print(f"{k}: {v}", file=f); print(f"{k}: {v}")
