"""Is the ~17-20 us per-call floor host dispatch (which HIP graphs remove) or GPU time?"""
import json, torch
import gemm_decode_shapes as g

def per_call_graph(fn, copies, iters=200, reps=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3): fn(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(iters): fn(i)
    graph.replay(); torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); graph.replay(); b.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / 1e3 / iters)
    return sorted(out)[len(out)//2]

rows = []
for name, n, k, count in g.E2B_SHAPES:
    for m in (1, 8, 64):
        for dt in ("bf16", "fp8"):
            copies = g.copies_needed(g.weight_bytes(n, k, dt), 1 << 30)
            fn = g._build(torch, dt, m, n, k, copies)
            eager = sorted(g._time(torch, fn, 500, copies) for _ in range(3))[1]
            try:
                graphed = per_call_graph(fn, copies)
                err = None
            except Exception as e:
                graphed, err = None, f"{type(e).__name__}: {str(e)[:150]}"
            r = {"shape": name, "n": n, "k": k, "count": count, "m": m, "dtype": dt, "copies": copies,
                 "eager_us": eager * 1e6, "graph_us": None if graphed is None else graphed * 1e6, "error": err}
            if graphed:
                r["graph_pct_measured_bw"] = 100 * g.weight_bytes(n, k, dt) / graphed / g.MEASURED_COPY_BW
            rows.append(r); print(json.dumps(r), flush=True)
            fn = None; torch.cuda.empty_cache()

# per-token sums, computed here
summary = []
for m in (1, 8, 64):
    for dt in ("bf16", "fp8"):
        sel = [r for r in rows if r["m"] == m and r["dtype"] == dt]
        eager = sum(r["eager_us"] * r["count"] for r in sel)
        graph = None if any(r["graph_us"] is None for r in sel) else sum(r["graph_us"] * r["count"] for r in sel)
        summary.append({"m": m, "dtype": dt, "eager_us_per_token": eager, "graph_us_per_token": graph})
for s in summary:
    b = next(x for x in summary if x["m"] == s["m"] and x["dtype"] == "bf16")
    s["eager_x_bf16"] = b["eager_us_per_token"] / s["eager_us_per_token"]
    s["graph_x_bf16"] = None if s["graph_us_per_token"] is None else b["graph_us_per_token"] / s["graph_us_per_token"]
print("SUMMARY " + json.dumps(summary))
json.dump({"torch": torch.__version__, "rows": rows, "summary": summary}, open("/tmp/graph_check.json", "w"), indent=2)
