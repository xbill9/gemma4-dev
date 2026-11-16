#!/usr/bin/env python3
"""gen_eval.py -- generation and tool-calling reads against a served model, scored in code.

  gen_eval.py run <upstream> <model> <task> <out.jsonl> [--limit N] [--concurrency C]
  gen_eval.py score <out.jsonl>

Tasks (records from build_gen_set.py, under data/):
  gsm8k        free-form chain of thought, greedy, GEN_EVAL_MAX_TOKENS tokens (default 768); right when the number the model gives
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
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def setting(name, default=None):
    """GEN_EVAL_<name>, else the older JEV_<name> spelling, else default."""
    return os.environ.get("GEN_EVAL_" + name, os.environ.get("JEV_" + name, default))


GSM8K_PROMPT = (
    "Solve the following math problem. Think step by step, then give the final answer "
    "on its own last line as 'Answer: <number>'.\n\n{question}"
)
GSM8K_MAX_TOKENS = int(os.environ.get("GEN_EVAL_MAX_TOKENS") or setting("GEN_MAX_TOKENS", "768"))
# GEN_EVAL_CACHE_PROMPT=0 sends llama.cpp's "cache_prompt": false (unset: the server default, on).
CACHE_PROMPT = setting("CACHE_PROMPT")
# GEN_EVAL_ENABLE_THINKING=0/1 sends chat_template_kwargs {"enable_thinking": ...} (unset: the server default).
# mistral.rs v0.9.5 thinks by default on Gemma 4; llama.cpp, vLLM, PyTorch and LiteRT do not.
ENABLE_THINKING = setting("ENABLE_THINKING")
# Hosted OpenAI-compatible APIs (all unset: a local server at <upstream>/v1/chat/completions, no auth).
# GEN_EVAL_CHAT_URL replaces <upstream>/v1/chat/completions with a full URL; GEN_EVAL_API_KEY_FILE sends its
# contents as a Bearer token; GEN_EVAL_REASONING_EFFORT sends "reasoning_effort"; GEN_EVAL_RETRIES retries
# HTTP 429/500/502/503/504 and network errors that many times, waiting 2, 4, 8 ... s (capped at 60).
CHAT_URL = setting("CHAT_URL")
API_KEY_FILE = setting("API_KEY_FILE")
REASONING_EFFORT = setting("REASONING_EFFORT")
RETRIES = int(setting("RETRIES", "0"))
# GEN_EVAL_TIMEOUT_S: seconds one request may wait before it counts as a network error (default 900).
TIMEOUT_S = float(setting("TIMEOUT_S", "900"))
# aime2025 and gpqa_diamond (build_hard_set.py): GEN_EVAL_HARD_MAX_TOKENS tokens (default 32768).
HARD_MAX_TOKENS = int(setting("HARD_MAX_TOKENS", "32768"))
AIME_PROMPT = (
    "Solve the following competition math problem. The answer is an integer from 0 to 999. Think step by "
    "step, then give the final answer on its own last line as 'Answer: <number>'.\n\n{question}"
)
GPQA_PROMPT = (
    "Answer the following multiple-choice question. Think step by step, then give the letter of the "
    "correct option on its own last line as 'Answer: <letter>'.\n\n{question}\n\n{options}"
)
RETRY_CODES = {429, 500, 502, 503, 504}
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


# ---------- GPQA Diamond ----------


def gpqa_extract(text):
    m = re.findall(r"Answer:\s*\**\s*\(?([ABCD])\b", text or "")
    return m[-1] if m else None


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


def post(upstream, body, timeout=None, chat_url=None, key_file=None):
    timeout = timeout or TIMEOUT_S
    url = chat_url or CHAT_URL or upstream.rstrip("/") + "/v1/chat/completions"
    key_file = key_file or API_KEY_FILE
    headers = {"Content-Type": "application/json"}
    if key_file:
        headers["Authorization"] = "Bearer " + open(os.path.expanduser(key_file)).read().strip()
    data = json.dumps(body).encode()
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data, headers), timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_CODES or attempt == RETRIES:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == RETRIES:
                raise
        time.sleep(min(60, 2 ** (attempt + 1)))


def one(upstream, model, task, rec, opts=None):
    """opts overrides the environment settings for this request: chat_url, key_file,
    reasoning_effort, max_tokens, timeout_s (budget_run.py drives several models in one process)."""
    opts = opts or {}
    t0 = time.time()
    if task in ("aime2025", "gpqa_diamond"):
        if task == "aime2025":
            content = AIME_PROMPT.format(question=rec["question"])
        else:
            options = "\n".join(f"{L}) {o}" for L, o in zip("ABCD", rec["options"]))
            content = GPQA_PROMPT.format(question=rec["question"], options=options)
        body = {
            "model": model,
            "temperature": 0,
            "max_tokens": opts.get("max_tokens", HARD_MAX_TOKENS),
            "messages": [{"role": "user", "content": content}],
        }
    elif task == "gsm8k":
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
    if CACHE_PROMPT is not None:
        body["cache_prompt"] = CACHE_PROMPT != "0"
    effort = opts.get("reasoning_effort", REASONING_EFFORT)
    if effort:
        body["reasoning_effort"] = effort
    if ENABLE_THINKING is not None:
        body["chat_template_kwargs"] = {"enable_thinking": ENABLE_THINKING != "0"}
    out = {"id": rec["id"], "task": task, "max_tokens": body["max_tokens"]}
    try:
        r = post(upstream, body, opts.get("timeout_s"), opts.get("chat_url"), opts.get("key_file"))
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
    # llama.cpp returns the thinking as reasoning_content and its speed and drafter counts as timings;
    # keep both when present, so thinking can be tokenized later and drafter acceptance read per reply.
    if msg.get("reasoning_content"):
        out["reasoning"] = msg["reasoning_content"]
    if isinstance(r.get("timings"), dict):
        out["timings"] = r["timings"]
    u = r["usage"]
    if "prompt_tokens" in u:
        out["prompt_tokens"] = u["prompt_tokens"]
    # Gemini's OpenAI-compatible endpoint leaves thinking out of completion_tokens and counts it only
    # in total_tokens; record the difference so token and cost columns include it.
    hidden = u.get("total_tokens", 0) - u.get("prompt_tokens", 0) - u["completion_tokens"]
    if "total_tokens" in u and hidden > 0:
        out["reasoning_tokens"] = hidden
    if task in ("gsm8k", "aime2025"):
        out["answer"] = rec["answer"]
        out["extracted"] = gsm8k_extract(msg.get("content"))
        out["right"] = gsm8k_right(msg.get("content"), rec["answer"])
    elif task == "gpqa_diamond":
        out["answer"] = rec["answer"]
        out["extracted"] = gpqa_extract(msg.get("content"))
        out["right"] = out["extracted"] == rec["answer"]
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
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    # Every reply is appended to <out>.partial as it finishes, so a crash loses at most the replies in
    # flight. --resume (or GEN_EVAL_RESUME=1) keeps the partial file's rows that have no error and asks
    # only the rest; without it the partial file starts empty. <out> is written, in dataset order, at the end.
    partial = a.out + ".partial"
    done = {}
    if (a.resume or setting("RESUME") == "1") and os.path.exists(partial):
        for line in open(partial):
            try:
                r = json.loads(line)
            except ValueError:
                continue  # a line cut short by the crash
            if "error" not in r:
                done[r["id"]] = r
    elif os.path.exists(partial):
        os.remove(partial)
    todo = [r for r in recs if r["id"] not in done]
    lock = threading.Lock()
    with open(partial, "a") as pf:

        def ask(rec):
            r = one(a.upstream, a.model, a.task, rec)
            r["model"] = a.model
            with lock:
                pf.write(json.dumps(r) + "\n")
                pf.flush()
            return r

        with ThreadPoolExecutor(a.concurrency) as ex:
            new = {r["id"]: r for r in ex.map(ask, todo)}
    rows = [done.get(rec["id"]) or new[rec["id"]] for rec in recs]
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.remove(partial)
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
    r.add_argument("task", choices=["gsm8k", "bfcl_simple", "aime2025", "gpqa_diamond"])
    r.add_argument("out")
    r.add_argument("--limit", type=int)
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--resume", action="store_true", help="keep finished rows from <out>.partial and ask only the rest")
    s = sub.add_parser("score")
    s.add_argument("out")
    a = p.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        print(json.dumps(summary([json.loads(line) for line in open(a.out)])))


if __name__ == "__main__":
    sys.exit(main())
