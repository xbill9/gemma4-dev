"""Engine vs stock_cpu.py token for token, then decode rate, memory and recall as the context held grows."""
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [here, os.path.abspath(os.path.join(here, "../../.."))]
import torch
torch.set_grad_enabled(False)
from cases import cases
from e4b_engine import Engine
max_ctx = int(sys.argv[1]); t0 = time.time()
eng = Engine(max_ctx=max_ctx)
print(json.dumps({"load_s": round(time.time() - t0, 1), "max_ctx": max_ctx, "weights_mib": eng.weights_mib, "kv_mib": eng.kv_mib}), flush=True)
ref = {r["case"]: r for r in map(json.loads, open(os.path.join(here, "stock.jsonl")))}
same = 0
for name, msgs, n in cases():
    ids = eng.render(msgs); r = eng.generate(ids, max_new=n)
    want = [t for t in ref[name]["tokens"] if t not in eng.stop_ids]
    ok = r["tokens"] == want; same += ok
    print(json.dumps({"case": name, "n_in": ids.shape[1], "n_ref": ref[name]["n_in"], "identical": ok, "got": r["tokens"][:8], "want": want[:8]}), flush=True)
    if not ok:
        i = next(j for j, (x, y) in enumerate(zip(r["tokens"] + [-1], want + [-2])) if x != y)
        pre = torch.cat([ids, torch.tensor([want[:i]], device=0, dtype=ids.dtype)], 1)
        lp = torch.log_softmax(eng.read_prompt(pre)[0].float(), -1)
        print(json.dumps({"case": name, "split_at": i, "engine_token": r["tokens"][i], "stock_token": want[i],
                          "engine_lp": [round(float(lp[r["tokens"][i]]), 4), round(float(lp[want[i]]), 4)]}), flush=True)
print(json.dumps({"identical": same, "of": len(ref)}), flush=True)
filler = "".join(f"Record {i:06d}: the code for record {i:06d} is {(i * 7919 + 13) % 900000 + 100000}.\n" for i in range(1, 12000))
hdr = eng.tok(filler, return_tensors="pt")["input_ids"].shape[1] / len(filler)
for depth in [d for d in (4096, 16384, 32768, 40000) if d + 300 < max_ctx]:
    lines = filler.split("\n"); n = int(depth * len(lines) / eng.tok(filler, return_tensors="pt")["input_ids"].shape[1])
    tgt = n // 3
    text = "\n".join(lines[:n]) + f"\n\nQuestion: what is the code for record {tgt:06d}? Answer with the code only."
    ids = eng.render([{"role": "user", "content": text}])
    torch.cuda.reset_peak_memory_stats()
    a = eng.generate(ids, max_new=16)
    print(json.dumps({"depth": ids.shape[1], "prompt_s": round(a["prompt_s"], 1), "prompt_tok_s": round(ids.shape[1] / a["prompt_s"]),
                      "reply": eng.tok.decode(a["tokens"]), "want": (tgt * 7919 + 13) % 900000 + 100000,
                      "peak_mib": torch.cuda.max_memory_allocated() // 2**20}), flush=True)
    # 128 decoded tokens at this depth, continuing the same held context
    prompt = [{"role": "user", "content": text.replace("Answer with the code only.", "Then list ten uses of a paperclip.")}]
    ids = eng.render(prompt)
    b = eng.generate(ids, max_new=128, stop_ids=[])
    print(json.dumps({"depth": ids.shape[1], "n": len(b["tokens"]), "dec_tok_s": round((len(b["tokens"]) - 1) / b["decode_s"], 2),
                      "peak_mib": torch.cuda.max_memory_allocated() // 2**20}), flush=True)
