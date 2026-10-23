"""First-token log-probabilities from stock transformers, bf16 on the CPU, for the cases in first_token_vs_llamacpp.py.

Loads the -text checkpoint unpacked to bf16 (every int4 code on Google's QAT grid), no engine code.
"""

import json
import os
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

J = os.path.expanduser("~/gemma4-dev/jev-tpu-v5e1")
sys.path[:0] = [J]
import gen_eval  # noqa: E402

TEXT = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
tok = AutoTokenizer.from_pretrained(TEXT)
model = AutoModelForCausalLM.from_pretrained(
    TEXT,
    dtype=torch.bfloat16,
    key_mapping={r"^model\.language_model\.": "model."},
    device_map="cpu",
    quantization_config=transformers.CompressedTensorsConfig(run_compressed=False),
).eval()
bfcl = {r["id"]: r for r in map(json.loads, open(os.path.join(J, "data", "bfcl_simple.jsonl")))}
cases = [("chat", [{"role": "user", "content": "What is the capital of France?"}], None)]
cases += [(i, bfcl[i]["question"], gen_eval.tools_for(bfcl[i]["functions"])) for i in ("simple_53", "simple_205", "simple_258")]
torch.set_num_threads(12)
with torch.inference_mode():
    for name, msgs, tools in cases:
        ids = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        lp = torch.log_softmax(model(**ids, logits_to_keep=1).logits[0, -1].float(), -1)
        top = torch.topk(lp, 5)
        print(json.dumps({"case": name, "prompt_tokens": ids["input_ids"].shape[1],
                          "stock_bf16_cpu": [[tok.decode([int(i)]), round(float(v), 3)] for v, i in zip(top.values, top.indices)]}), flush=True)
