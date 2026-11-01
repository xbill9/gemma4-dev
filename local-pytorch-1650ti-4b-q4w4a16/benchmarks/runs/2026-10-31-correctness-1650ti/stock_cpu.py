"""Greedy reference from stock transformers on Google's unquantized QAT weights, bf16 on the CPU, no engine code."""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cases import cases
R = "google/gemma-4-E4B-it-qat-q4_0-unquantized"
tok = AutoTokenizer.from_pretrained(R)
m = AutoModelForCausalLM.from_pretrained(R, dtype=torch.bfloat16, device_map="cpu").eval()
torch.set_num_threads(10)
with torch.inference_mode():
    for name, msgs, n in cases():
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        o = m.generate(**ids, max_new_tokens=n, do_sample=False, top_p=None, top_k=None, temperature=None)
        toks = o[0, ids["input_ids"].shape[1]:].tolist()
        print(json.dumps({"case": name, "n_in": ids["input_ids"].shape[1], "tokens": toks}), flush=True)
