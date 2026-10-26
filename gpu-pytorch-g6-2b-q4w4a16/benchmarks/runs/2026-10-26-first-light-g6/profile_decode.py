"""Where a decode step's GPU time goes: torch.profiler over 50 CUDA-graph replays of the engine's captured step.

    python3 profile_decode.py OUT.json
"""

import collections
import json
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/opt/torch-g6-q4/app")
from e2b_engine import Engine  # noqa: E402

eng = Engine(max_ctx=131072, prefill_dtype=torch.float16)
ids = eng.render([{"role": "user", "content": "Write a long story about a lighthouse keeper."}])
with torch.inference_mode():
    first = int(eng.read_prompt(ids).argmax(-1))
    eng.st.decode = True
    eng.tok_buf.fill_(first)
    eng.st.pos.fill_(ids.shape[1])
    for _ in range(10):
        eng.graph.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(50):
        eng.graph.replay()
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - t) / 50 * 1e3
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(50):
            eng.graph.replay()
        torch.cuda.synchronize()

by = collections.defaultdict(lambda: [0.0, 0])
for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        by[e.name][0] += e.device_time_total / 1e3 / 50
        by[e.name][1] += 1
rows = sorted(((n, round(v[0], 4), v[1] // 50) for n, v in by.items()), key=lambda r: -r[1])
kernel_ms = sum(r[1] for r in rows)
res = {"device": torch.cuda.get_device_name(0), "step_ms_wall": round(step_ms, 3), "kernel_ms_sum": round(kernel_ms, 3),
       "kernels_per_step": sum(r[2] for r in rows), "top": [{"name": n[:120], "ms": ms, "calls": c} for n, ms, c in rows[:30]]}
json.dump(res, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "top"}))
for r in res["top"][:25]:
    print(f'{r["ms"]:8.4f} ms {r["calls"]:4d}x  {r["name"]}')
