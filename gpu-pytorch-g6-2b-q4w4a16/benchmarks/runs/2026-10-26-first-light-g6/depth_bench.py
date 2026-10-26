"""Prompt read, decode and recall at depth on the deployed engine; the method of the 1650 Ti's e2b_long.py.

    python3 depth_bench.py OUT.jsonl 0,4096,32768 [float16|float32]

Prompts come from one seeded generator in the order the depths run (dump_prompts.py in
local-pytorch-1650ti-2b-q4w4a16/benchmarks/runs/2026-10-23-long-context-1650ti), so `0,4096,32768` rebuilds that
run's prompts-short.json and `32768,131072` its prompts-deep.json. Decode is 63 greedy tokens with stop tokens off.
"""

import json
import random
import sys
import time

import torch
import transformers

sys.path.insert(0, "/opt/torch-g6-q4/app")
from e2b_engine import Engine  # noqa: E402

out_path, depths = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
prefill = getattr(torch, sys.argv[3]) if len(sys.argv) > 3 else torch.float16
fh = open(out_path, "a")


def out(**kw):
    fh.write(json.dumps(kw) + "\n")
    fh.flush()
    print(json.dumps(kw), flush=True)


t0 = time.perf_counter()
eng = Engine(max_ctx=131072, prefill_dtype=prefill)
out(step="load", device=torch.cuda.get_device_name(0), prefill_dtype=str(prefill), weights_mib=eng.weights_mib,
    kv_mib=eng.kv_mib, load_s=round(time.perf_counter() - t0, 1), cuda_alloc_mib=torch.cuda.memory_allocated() // 2**20,
    torch=torch.__version__, transformers=transformers.__version__)

tok = eng.tok
rng = random.Random(1234)
QUESTION = "\n\nQuestion: what is the code for record {r:06d}? Answer with the code only."


def build(depth):
    head = tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False)
    overhead = len(tok(head + QUESTION.format(r=0), add_special_tokens=False)["input_ids"])
    records, used, codes, i = [], overhead, {}, 0
    while True:
        i += 1
        code = rng.randrange(100000, 999999)
        line = f"Record {i:06d}: the code for record {i:06d} is {code}.\n"
        cost = len(tok(line, add_special_tokens=False)["input_ids"])
        if used + cost > depth and records:
            break
        records.append(line)
        codes[i] = code
        used += cost
        if depth == 0:
            break
    target = max(1, len(records) // 4)
    return eng.render([{"role": "user", "content": "".join(records) + QUESTION.format(r=target)}]), codes[target]


for depth in depths:
    ids, code = build(depth)
    torch.cuda.reset_peak_memory_stats()
    r = eng.generate(ids, max_new=64, stop_ids=[])
    toks = r["tokens"]
    cut = next((i for i, t in enumerate(toks) if t in eng.stop_ids), len(toks))
    answer = tok.decode(toks[:cut], skip_special_tokens=True).strip()
    out(step="depth", depth=depth, prompt_tokens=ids.shape[1], prompt_s=round(r["prompt_s"], 2),
        prompt_tok_per_s=round(ids.shape[1] / r["prompt_s"], 1), decode_tokens=len(toks) - 1,
        decode_tok_per_s=round((len(toks) - 1) / r["decode_s"], 2),
        peak_cuda_mib=torch.cuda.max_memory_allocated() // 2**20, recall_code=code, answer=answer[:80],
        recalled=str(code) in answer)
