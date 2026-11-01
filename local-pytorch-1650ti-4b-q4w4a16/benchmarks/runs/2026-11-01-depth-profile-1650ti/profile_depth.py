"""Where a decode step's GPU time goes at short context and at depth, E2B and E4B through the same method.

    python3 profile_depth.py {2b|4b} OUT.jsonl [0,32768]

Prompts are the seeded records of depth_bench.py. At each depth: read the prompt once, decode 63 tokens the way
generate() does for the step rate, then torch.profiler over PASSES blocks of STEPS replays of one fixed token. Each
kernel call is kept by its place in the step, so one attention layer can be told from another. Both models run at
max_ctx 40960: decode reads slots up to the current position, whatever the capacity.
"""

import collections
import json
import random
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

model, out_path = sys.argv[1], sys.argv[2]
depths = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,32768").split(",")]
ROOT = "/home/xbill/gemma4-dev/local-pytorch-1650ti-"
sys.path.insert(0, ROOT + model + "-q4w4a16")
from importlib import import_module  # noqa: E402

Engine = import_module("e2b_engine" if model == "2b" else "e4b_engine").Engine
PASSES, STEPS = 2, 50
fh = open(out_path, "a")


def out(**kw):
    fh.write(json.dumps(kw) + "\n")
    fh.flush()
    print(json.dumps({k: v for k, v in kw.items() if k != "calls"}), flush=True)


t0 = time.perf_counter()
eng = Engine(max_ctx=40960)
out(step="load", model=model, torch=torch.__version__, weights_mib=eng.weights_mib, kv_mib=eng.kv_mib,
    load_s=round(time.perf_counter() - t0, 1))
tok, ple = eng.tok, getattr(eng, "ple", None)
rng = random.Random(1234)
QUESTION = "\n\nQuestion: what is the code for record {r:06d}? Answer with the code only."


def build(depth):
    head = tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False)
    used, records, i = len(tok(head + QUESTION.format(r=0), add_special_tokens=False)["input_ids"]), [], 0
    while True:
        i += 1
        line = f"Record {i:06d}: the code for record {i:06d} is {rng.randrange(100000, 999999)}.\n"
        cost = len(tok(line, add_special_tokens=False)["input_ids"])
        if used + cost > depth and records:
            break
        records.append(line)
        used += cost
        if depth == 0:
            break
    return eng.render([{"role": "user", "content": "".join(records) + QUESTION.format(r=max(1, len(records) // 4))}])


def set_token(t, pos):
    eng.tok_buf.fill_(t)
    if ple is not None:
        ple.load(t)
    eng.st.pos.fill_(pos)


def profile_pass():
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(STEPS):
            eng.graph.replay()
        torch.cuda.synchronize()
    evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    evs.sort(key=lambda e: e.time_range.start)
    per = len(evs) // STEPS
    calls = collections.defaultdict(float)
    names = {}
    for i, e in enumerate(evs):
        calls[i % per] += e.device_time_total / 1e3 / STEPS
        names[i % per] = e.name[:80]
    return [[names[i], round(calls[i], 4)] for i in range(per)]


with torch.inference_mode():
    for depth in depths:
        ids = build(depth)
        t = time.perf_counter()
        first = int(eng.read_prompt(ids).argmax(-1))
        prompt_s = time.perf_counter() - t
        eng.st.decode = True
        n, toks = ids.shape[1], [first]
        torch.cuda.synchronize()
        t = time.perf_counter()
        while len(toks) < 64:
            set_token(toks[-1], n + len(toks) - 1)
            eng.graph.replay()
            toks.append(int(eng.next_buf))
        rate = (len(toks) - 1) / (time.perf_counter() - t)
        pos = n + len(toks) - 1
        set_token(toks[-1], pos)
        for p in range(PASSES):
            for _ in range(10):
                eng.graph.replay()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(STEPS):
                eng.graph.replay()
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t) / STEPS * 1e3
            out(step="profile", model=model, depth=depth, prompt_tokens=n, pos=pos, prompt_s=round(prompt_s, 1),
                decode_tok_per_s=round(rate, 2), graph_ms_wall=round(wall, 3), passn=p, calls=profile_pass())
