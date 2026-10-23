"""First-token log-probabilities from the engine beside llama-server's (/completion n_probs) on the same rendered prompt.

    llama-server on 127.0.0.1:8199 (CPU is enough); python3 first_token_vs_llamacpp.py > first_token_vs_llamacpp.jsonl
"""

import json
import os
import sys
import urllib.request

import torch

RIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
J = os.path.expanduser("~/gemma4-dev/jev-tpu-v5e1")
sys.path[:0] = [RIG, J]

import gen_eval  # noqa: E402
from e2b_engine import Engine  # noqa: E402


def post(path, body):
    req = urllib.request.Request("http://127.0.0.1:8199" + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


bfcl = {r["id"]: r for r in map(json.loads, open(os.path.join(J, "data", "bfcl_simple.jsonl")))}
cases = [("chat", [{"role": "user", "content": c}], None) for c in ("Hi", "What is the capital of France?", "Write a haiku about rain.")]
cases += [(i, bfcl[i]["question"], gen_eval.tools_for(bfcl[i]["functions"])) for i in ("simple_53", "simple_205", "simple_258")]
eng = Engine(max_ctx=8192)
with torch.inference_mode():
    for name, msgs, tools in cases:
        ids = eng.render(msgs, tools=tools)
        lp = torch.log_softmax(eng.read_prompt(ids)[0].float(), -1)
        top = torch.topk(lp, 5)
        prompt = post("/apply-template", {"messages": msgs, **({"tools": tools} if tools else {})})["prompt"]
        r = post("/completion", {"prompt": prompt, "n_predict": 1, "n_probs": 5, "temperature": 0, "cache_prompt": False})
        print(json.dumps({
            "case": name, "prompt_tokens": ids.shape[1], "llamacpp_prompt_tokens": r["tokens_evaluated"],
            "engine": [[eng.tok.decode([int(i)]), round(float(v), 3)] for v, i in zip(top.values, top.indices)],
            "llamacpp": [[t["token"], round(t["logprob"], 3)] for t in r["completion_probabilities"][0]["top_logprobs"]],
        }), flush=True)
