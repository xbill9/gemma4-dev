#!/usr/bin/env python3
"""gen_eval.py -- generation and tool-calling reads against a served model, scored in code.

  gen_eval.py run <upstream> <model> <task> <out.jsonl> [--limit N] [--concurrency C]
  gen_eval.py score <out.jsonl>

Tasks (records from build_gen_set.py, under data/):
  gsm8k        free-form chain of thought, greedy, JEV_GEN_MAX_TOKENS tokens (default 768); right when the number the model gives
               after "Answer:" (else its last number) equals the reference.
  bfcl_simple  one tool offered through the OpenAI `tools` field, tool_choice auto, greedy; right
               when the server returns exactly one tool call whose name and arguments pass the checks
               below. Needs the server started with --enable-auto-tool-choice --tool-call-parser gemma4.

The BFCL check follows BFCL's AST checker for the simple category: the name must match; every
required parameter must be present; no parameter outside the schema; every given value must be one
of the accepted values, with strings compared after lower-casing and dropping spaces and ,./-_*^,
ints accepted where a float is expected, tuples read as lists, and dicts checked key by key; an
omitted optional parameter is accepted only when "" is among its accepted values.

Every output record keeps the raw completion, so a score can be recomputed without the server.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
GSM8K_PROMPT = (
    "Solve the following math problem. Think step by step, then give the final answer "
    "on its own last line as 'Answer: <number>'.\n\n{question}"
)
GSM8K_MAX_TOKENS = int(os.environ.get("JEV_GEN_MAX_TOKENS", "768"))
TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": None}


# ---------- GSM8K ----------


def number(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v


def gsm8k_extract(text):
    m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text or "")
    if m:
        return number(m[-1])
    m = re.findall(r"-?[\d,]*\.?\d+", text or "")
    return number(m[-1]) if m else None


def gsm8k_right(text, answer):
    got = gsm8k_extract(text)
    return got is not None and abs(got - float(answer)) < 1e-6


# ---------- BFCL simple ----------


def tool_name(name):
    return name.replace(".", "_")


def convert_schema(p):
    """BFCL's parameter schema to JSON Schema, as BFCL itself does for OpenAI-style models."""
    p = dict(p)
    t = p.get("type")
    if t in TYPES:
        if TYPES[t] is None:
            p.pop("type")
        else:
            p["type"] = TYPES[t]
    if "properties" in p:
        p["properties"] = {k: convert_schema(v) for k, v in p["properties"].items()}
    if "items" in p:
        p["items"] = convert_schema(p["items"])
    return p


def tools_for(functions):
    return [
        {
            "type": "function",
            "function": {
                "name": tool_name(f["name"]),
                "description": f["description"],
                "parameters": convert_schema(f["parameters"]),
            },
        }
        for f in functions
    ]


def standardize(s):
    return re.sub(r"[ ,./\-_*^]", "", s).lower().replace("'", '"')


def value_ok(value, accepted, ptype):
    for a in accepted:
        if a == "":
            continue
        if match(value, a, ptype):
            return True
    return False


def match(v, a, ptype):
    if ptype == "float" and isinstance(v, int) and not isinstance(v, bool):
        v = float(v)
    if ptype == "tuple" and isinstance(v, list):
        a = list(a) if isinstance(a, (list, tuple)) else a
    if isinstance(a, str):
        return isinstance(v, str) and standardize(v) == standardize(a)
    if isinstance(a, bool) or isinstance(v, bool):
        return type(a) is type(v) and a == v
    if isinstance(a, (int, float)):
        return isinstance(v, (int, float)) and float(v) == float(a) and (ptype != "integer" or isinstance(v, int))
    if isinstance(a, list):
        if not isinstance(v, list) or len(v) != len(a):
            return False
        return all(match(x, y, None) for x, y in zip(v, a))
    if isinstance(a, dict):
        if not isinstance(v, dict):
            return False
        for k, acc in a.items():
            if k not in v:
                if "" not in acc:
                    return False
                continue
            if not value_ok(v[k], acc, None):
                return False
        return set(v) <= set(a)
    return v == a


def bfcl_check(calls, record):
    """calls: [{"name", "arguments"(dict)}] from the server. Returns (right, reason)."""
    if len(calls) != 1:
        return False, f"{len(calls)} tool calls"
    func = record["functions"][0]
    possible = record["possible"][0]
    want_name = next(iter(possible))
    call = calls[0]
    if call["name"] not in (want_name, tool_name(want_name)):
        return False, f"name {call['name']}"
    args = call["arguments"]
    if not isinstance(args, dict):
        return False, "arguments not an object"
    props = func["parameters"].get("properties", {})
    for r in func["parameters"].get("required", []):
        if r not in args:
            return False, f"missing required {r}"
    accepted = possible[want_name]
    for k, v in args.items():
        if k not in props:
            return False, f"unexpected parameter {k}"
        if k not in accepted:
            return False, f"parameter {k} not in answer"
        if not value_ok(v, accepted[k], props[k].get("type")):
            return False, f"value {k}={json.dumps(v)[:80]}"
    for k, acc in accepted.items():
        if k not in args and "" not in acc:
            return False, f"missing {k}"
    return True, "ok"


# ---------- running ----------


def post(upstream, body, timeout=900):
    req = urllib.request.Request(
        upstream.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def one(upstream, model, task, rec):
    t0 = time.time()
    if task == "gsm8k":
        body = {
            "model": model,
            "temperature": 0,
            "max_tokens": GSM8K_MAX_TOKENS,
            "messages": [{"role": "user", "content": GSM8K_PROMPT.format(question=rec["question"])}],
        }
    else:
        body = {
            "model": model,
            "temperature": 0,
            "max_tokens": 512,
            "messages": rec["question"],
            "tools": tools_for(rec["functions"]),
            "tool_choice": "auto",
        }
    out = {"id": rec["id"], "task": task, "max_tokens": body["max_tokens"]}
    try:
        r = post(upstream, body)
    except Exception as e:  # recorded, scored wrong, counted separately
        out.update(error=f"{type(e).__name__}: {e}"[:300], right=False, seconds=round(time.time() - t0, 2))
        return out
    msg = r["choices"][0]["message"]
    out.update(
        text=msg.get("content"),
        finish=r["choices"][0].get("finish_reason"),
        completion_tokens=r["usage"]["completion_tokens"],
        seconds=round(time.time() - t0, 2),
    )
    if task == "gsm8k":
        out["answer"] = rec["answer"]
        out["extracted"] = gsm8k_extract(msg.get("content"))
        out["right"] = gsm8k_right(msg.get("content"), rec["answer"])
    else:
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                a = json.loads(c["function"]["arguments"])
            except (ValueError, TypeError):
                a = c["function"]["arguments"]
            calls.append({"name": c["function"]["name"], "arguments": a})
        out["calls"] = calls
        out["right"], out["reason"] = bfcl_check(calls, rec)
    return out


def run(a):
    recs = [json.loads(line) for line in open(os.path.join(HERE, "data", a.task + ".jsonl"))]
    if a.limit:
        recs = recs[: a.limit]
    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        rows = list(ex.map(lambda r: one(a.upstream, a.model, a.task, r), recs))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            r["model"] = a.model
            f.write(json.dumps(r) + "\n")
    s = summary(rows)
    s["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(s))


def summary(rows):
    n = len(rows)
    right = sum(r["right"] for r in rows)
    return {
        "task": rows[0]["task"] if rows else None,
        "max_tokens": rows[0].get("max_tokens") if rows else None,
        "n": n,
        "right": right,
        "accuracy": round(right / n, 4) if n else None,
        "errors": sum("error" in r for r in rows),
        "truncated": sum(r.get("finish") == "length" for r in rows),
    }


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("upstream")
    r.add_argument("model")
    r.add_argument("task", choices=["gsm8k", "bfcl_simple"])
    r.add_argument("out")
    r.add_argument("--limit", type=int)
    r.add_argument("--concurrency", type=int, default=16)
    s = sub.add_parser("score")
    s.add_argument("out")
    a = p.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        print(json.dumps(summary([json.loads(line) for line in open(a.out)])))


if __name__ == "__main__":
    sys.exit(main())
