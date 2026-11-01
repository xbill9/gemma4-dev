"""Load the E4B int4 engine, run the eight short prompts twice, report decode rate, memory and token identity."""
import json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import torch
from e4b_engine import Engine

here = os.path.dirname(os.path.abspath(__file__))
max_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
t0 = time.time()
eng = Engine(max_ctx=max_ctx)
print(json.dumps({"load_s": round(time.time() - t0, 1), "max_ctx": max_ctx, "weights_mib": eng.weights_mib,
                  "kv_mib": eng.kv_mib, "alloc_mib": torch.cuda.memory_allocated() // 2**20}), flush=True)
prompts = json.load(open(os.path.join(here, "prompts-eight.json")))
seen = {}
for rep in range(2):
    for i, p in enumerate(prompts):
        msgs = p if isinstance(p, list) else [{"role": "user", "content": p["prompt"] if isinstance(p, dict) else p}]
        ids = eng.render(msgs)
        torch.cuda.reset_peak_memory_stats()
        r = eng.generate(ids, max_new=128)
        same = seen.setdefault(i, r["tokens"]) == r["tokens"]
        print(json.dumps({"rep": rep, "i": i, "n_in": ids.shape[1], "n_out": len(r["tokens"]), "finish": r["finish"],
                          "prompt_s": round(r["prompt_s"], 3),
                          "dec_tok_s": round((len(r["tokens"]) - 1) / r["decode_s"], 2) if r["decode_s"] else None,
                          "peak_mib": torch.cuda.max_memory_allocated() // 2**20, "repeat_same": same,
                          "tokens": r["tokens"], "text": eng.tok.decode(r["tokens"])[:90]}), flush=True)
