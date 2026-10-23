"""Replay BFCL records through the engine and print the raw reply (special tokens kept) beside the parsed calls.

    python3 replay_bfcl.py simple_205 simple_258 ... > replay_bfcl.jsonl
"""

import json
import os
import sys

RIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
J = os.path.expanduser("~/gemma4-dev/jev-tpu-v5e1")
sys.path[:0] = [RIG, J]

import gen_eval  # noqa: E402
from e2b_engine import Engine  # noqa: E402
from gemma4_tools import parse_tool_calls  # noqa: E402

want = set(sys.argv[1:])
recs = [r for r in map(json.loads, open(os.path.join(J, "data", "bfcl_simple.jsonl"))) if r["id"] in want]
eng = Engine(max_ctx=8192)
for rec in recs:
    ids = eng.render(rec["question"], tools=gen_eval.tools_for(rec["functions"]))
    r = eng.generate(ids, max_new=512)
    raw = eng.tok.decode(r["tokens"], skip_special_tokens=False)
    content, calls = parse_tool_calls(raw)
    print(json.dumps({"id": rec["id"], "finish": r["finish"], "raw": raw, "content": content, "calls": calls}), flush=True)
