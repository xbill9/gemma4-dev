"""Prompt-read speed of the rig engine (sliding-layer K/V copied, w4attn.py) on prompts-short.json, 131,072-token cache.

Compare with prefill-profile-fp32.jsonl, read by this run's e2b_long.py with K/V broadcast.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))
from e2b_engine import Engine  # noqa: E402

eng = Engine(max_ctx=131072)
for p in json.load(open("prompts-short.json")):
    ids = eng.render([{"role": "user", "content": p["content"]}])
    for rep in range(2):
        r = eng.generate(ids, max_new=64, stop_ids=[])
        ans = eng.tok.decode(r["tokens"], skip_special_tokens=True).split("\n")[0].strip()
        print(json.dumps({"prompt_tokens": ids.shape[1], "rep": rep, "prompt_tok_per_s": round(ids.shape[1] / r["prompt_s"], 1),
                          "decode_tok_per_s": round((len(r["tokens"]) - 1) / r["decode_s"], 2), "recalled": ans == str(p["code"])}), flush=True)
