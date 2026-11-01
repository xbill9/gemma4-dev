"""Stock bf16 logprobs at each first split between stock_cpu.py and engine.jsonl, for fp32 too on the same weights."""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cases import cases
R = "google/gemma-4-E4B-it-qat-q4_0-unquantized"
tok = AutoTokenizer.from_pretrained(R)
c = {n: m for n, m, _ in cases()}
ref = {r["case"]: r for r in map(json.loads, open("stock.jsonl"))}
splits = [json.loads(l) for l in open("engine.jsonl") if '"split_at"' in l]
m = AutoModelForCausalLM.from_pretrained(R, dtype=torch.bfloat16, device_map="cpu").eval()
torch.set_num_threads(10)
for dt in (torch.bfloat16, torch.float32):
    if dt == torch.float32:
        m = m.to(torch.float32)
    with torch.inference_mode():
        for s in splits:
            ids = tok.apply_chat_template(c[s["case"]], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
            ids = torch.cat([ids, torch.tensor([ref[s["case"]]["tokens"][: s["split_at"]]])], 1)
            lp = torch.log_softmax(m(input_ids=ids, logits_to_keep=1).logits[0, -1].float(), -1)
            print(json.dumps({"dtype": str(dt), "case": s["case"], "split_at": s["split_at"], "engine_token_lp": round(float(lp[s["engine_token"]]), 4), "stock_token_lp": round(float(lp[s["stock_token"]]), 4)}), flush=True)
