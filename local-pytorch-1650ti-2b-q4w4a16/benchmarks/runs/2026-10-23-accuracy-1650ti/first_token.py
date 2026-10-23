"""Top-5 first-token logits for BFCL records, with the prompt read in fp32 and in fp16.

    python3 first_token.py fp32|fp16 simple_205 simple_258 ...
"""

import json
import os
import sys

import torch

RIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
J = os.path.expanduser("~/gemma4-dev/jev-tpu-v5e1")
sys.path[:0] = [RIG, J]

import gen_eval  # noqa: E402
from e2b_engine import Engine  # noqa: E402

dtype = {"fp32": torch.float32, "fp16": torch.float16}[sys.argv[1]]
want = set(sys.argv[2:])
recs = [r for r in map(json.loads, open(os.path.join(J, "data", "bfcl_simple.jsonl"))) if r["id"] in want]
eng = Engine(max_ctx=8192, prefill_dtype=dtype)
with torch.inference_mode():
    for rec in recs:
        ids = eng.render(rec["question"], tools=gen_eval.tools_for(rec["functions"]))
        lg = eng.read_prompt(ids)[0].float()
        top = torch.topk(lg, 5)
        r = eng.generate(ids, max_new=64)
        print(json.dumps({
            "id": rec["id"], "prefill": sys.argv[1], "prompt_tokens": ids.shape[1],
            "top5": [[eng.tok.decode([int(i)]), round(float(v), 3)] for v, i in zip(top.values, top.indices)],
            "logits_finite": bool(torch.isfinite(lg).all()),
            "reply": eng.tok.decode(r["tokens"], skip_special_tokens=False)[:200],
        }), flush=True)
