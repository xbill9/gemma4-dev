"""Time bf16 / fp16 / fp8 / int8 GEMMs at the shapes Gemma 4 E2B decodes with.

The dtype ranking in `../HARDWARE.md` (fp8 1.77x bf16, int8 0.69x) came from one
8192^3 matmul. That is the friendliest shape a GEMM gets — square, compute-bound,
every CU busy — and decode never runs it. At decode M is the number of sequences
in the batch, 1 to a few dozen, and N x K are the layer's weight shape, so the
kernel is a skinny matrix-vector product whose cost is streaming the weight, not
multiplying it. This script reruns the comparison there.

Four things keep the number honest:

* **Kernels are replayed from a HIP graph.** vLLM captures decode into graphs, so
  the host never dispatches a decode GEMM one call at a time. Timed eagerly, every
  small shape here floors at ~17 us bf16 / ~20 us fp8 whatever its size — that
  is Python and hipBLASLt dispatch, not the GPU — and on 2026-09-18 it made fp8
  read 0.97x bf16 per token when the graphed figure is 1.50x. `--mode eager`
  keeps the dispatch-bound number as a diagnostic; it is not the headline.

* **Weights are cold.** MI300X has a 256 MB Infinity Cache and most E2B layer
  weights are a few MB, so timing one weight tensor in a loop measures cache
  bandwidth. Each cell cycles through enough copies of the weight to exceed
  `--rotate-bytes` (1 GiB by default). `--rotate-bytes 0` gives the hot figure,
  as a diagnostic.
* **Failures are rows, not crashes.** `torch._scaled_mm` and `torch._int_mm` have
  shape constraints that small M may violate; a cell that cannot run is recorded
  `unsupported` with the error, because "cannot run at M=1" is a result.
* **All arithmetic is here.** Ratios against bf16, achieved bandwidth, percent of
  the measured copy bandwidth and the per-token decode sum are computed below and
  printed; nothing is left for a reader to divide.

What it does NOT measure: the per-call dynamic activation quantisation vLLM's fp8
path adds, or attention. The fp8 and int8
cells are therefore an upper bound on what those formats buy in a decode step.

Run it inside the serving image, which carries a ROCm torch (the workstation has
no GPU and no torch). With vLLM serving, beside it in the same container:

    scp -i ~/amd gemm_decode_shapes.py root@<ip>:/tmp/
    ssh -i ~/amd root@<ip> 'docker cp /tmp/gemm_decode_shapes.py vllm:/tmp/ && \\
        docker exec vllm python3 /tmp/gemm_decode_shapes.py --out /tmp/gemm.json'

The GEMM-free parts import without torch, so `tests/test_gemm_decode_shapes.py`
runs offline.
"""

import argparse
import json
import math
import statistics
import sys
import time
from typing import Any, Callable, Optional

# Weight shapes as vLLM executes them: QKV and gate/up are fused into one GEMM
# each (QKVParallelLinear, MergedColumnParallelLinear). Geometry is from
# ../MODELS.md — 28 sliding layers at head_dim 256, 7 full at 512, one KV head;
# `use_double_wide_mlp` doubles intermediate_size on the 20 KV-shared layers.
# (name, N out, K in, GEMMs of this shape per decoded token)
E2B_SHAPES: list[tuple[str, int, int, int]] = [
    ("qkv_sliding", 2048 + 256 + 256, 1536, 28),
    ("qkv_full", 4096 + 512 + 512, 1536, 7),
    ("o_sliding", 1536, 2048, 28),
    ("o_full", 1536, 4096, 7),
    ("gate_up", 2 * 6144, 1536, 15),
    ("gate_up_wide", 2 * 12288, 1536, 20),
    ("down", 1536, 6144, 15),
    ("down_wide", 1536, 12288, 20),
    ("lm_head", 262144, 1536, 1),
]

# The shape the original ranking came from. Run at M=8192 it reproduces that
# number, which is the check that this harness agrees with the one it questions.
CONTROL_SHAPE = ("control_8192", 8192, 8192, 0)

DECODE_MS = [1, 8, 64]  # 64 is the heaviest concurrency in the serving sweep
DTYPES = ["bf16", "fp16", "fp8", "int8"]

BYTES_PER_ELEMENT = {"bf16": 2, "fp16": 2, "fp8": 1, "int8": 1}

# ../HARDWARE.md: 3.76 TB/s measured device-to-device copy under load; 5.3 spec.
MEASURED_COPY_BW = 3.76e12
SPEC_BW = 5.3e12


def flops(m: int, n: int, k: int) -> int:
    return 2 * m * n * k


def weight_bytes(n: int, k: int, dtype: str) -> int:
    return n * k * BYTES_PER_ELEMENT[dtype]


def copies_needed(one_copy: int, rotate_bytes: int) -> int:
    """How many weight copies to cycle so the working set exceeds rotate_bytes."""
    if rotate_bytes <= 0:
        return 1
    return max(2, math.ceil(rotate_bytes / one_copy))


def cell_metrics(m: int, n: int, k: int, dtype: str, seconds_per_call: float) -> dict[str, float]:
    wb = weight_bytes(n, k, dtype)
    return {
        "us_per_call": seconds_per_call * 1e6,
        "tflops": flops(m, n, k) / seconds_per_call / 1e12,
        "weight_gbps": wb / seconds_per_call / 1e9,
        "pct_measured_bw": 100.0 * wb / seconds_per_call / MEASURED_COPY_BW,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Ratios against bf16 per (shape, M), and the per-token decode GEMM sum.

    Speedup is bf16 time / dtype time, so >1 means the dtype is faster. The
    per-token sum is only reported for a dtype when every E2B shape ran at that
    M — a partial sum would read as a faster format.
    """
    ok = {(r["shape"], r["m"], r["dtype"]): r for r in rows if r["status"] == "ok"}
    speedups = []
    for r in rows:
        base = ok.get((r["shape"], r["m"], "bf16"))
        if r["status"] != "ok" or base is None:
            continue
        speedups.append(
            {
                "shape": r["shape"],
                "m": r["m"],
                "dtype": r["dtype"],
                "speedup_vs_bf16": base["us_per_call"] / r["us_per_call"],
            }
        )

    per_token = []
    ms = sorted({r["m"] for r in rows if r["shape"] != CONTROL_SHAPE[0]})
    for m in ms:
        for dtype in DTYPES:
            cells = [ok.get((name, m, dtype)) for name, _, _, _ in E2B_SHAPES]
            if any(c is None for c in cells):
                per_token.append({"m": m, "dtype": dtype, "us_per_token": None, "missing": True})
                continue
            total = sum(c["us_per_call"] * count for c, (_, _, _, count) in zip(cells, E2B_SHAPES, strict=True))
            per_token.append({"m": m, "dtype": dtype, "us_per_token": total, "missing": False})
    for entry in per_token:
        base = next((p for p in per_token if p["m"] == entry["m"] and p["dtype"] == "bf16"), None)
        if entry["us_per_token"] is not None and base and base["us_per_token"] is not None:
            entry["speedup_vs_bf16"] = base["us_per_token"] / entry["us_per_token"]
        else:
            entry["speedup_vs_bf16"] = None

    streamed = sum(n * k * count for _, n, k, count in E2B_SHAPES)
    return {"speedups": speedups, "per_token": per_token, "streamed_params_per_token": streamed}


def plan(ms: list[int], include_control: bool) -> list[tuple[str, int, int, int]]:
    """Every (shape, N, K, M) cell to run, control first."""
    cells = []
    if include_control:
        name, n, k, _ = CONTROL_SHAPE
        for m in sorted(set(ms) | {8192}):
            cells.append((name, n, k, m))
    for name, n, k, _ in E2B_SHAPES:
        for m in ms:
            cells.append((name, n, k, m))
    return cells


# --- GPU side: everything below needs torch -------------------------------------


def _fp8_dtype(torch: Any) -> Any:
    """CDNA 3 uses e4m3fnuz; the OCP e4m3fn raises HIPBLAS_STATUS_NOT_SUPPORTED."""
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return torch.float8_e4m3fnuz if arch.startswith("gfx94") else torch.float8_e4m3fn


def _build(torch: Any, dtype: str, m: int, n: int, k: int, copies: int) -> Callable[[int], Any]:
    """Return fn(i) that runs one GEMM against weight copy i % copies."""
    dev = "cuda"
    if dtype in ("bf16", "fp16"):
        td = torch.bfloat16 if dtype == "bf16" else torch.float16
        a = torch.randn(m, k, device=dev, dtype=td)
        ws = [torch.randn(n, k, device=dev, dtype=td) for _ in range(copies)]
        return lambda i: torch.matmul(a, ws[i % copies].t())
    if dtype == "fp8":
        f8 = _fp8_dtype(torch)
        a = torch.randn(m, k, device=dev).to(f8)
        # [N, K] row-major transposed is the column-major [K, N] _scaled_mm wants.
        ws = [torch.randn(n, k, device=dev).to(f8) for _ in range(copies)]
        one = torch.ones((), device=dev, dtype=torch.float32)
        return lambda i: torch._scaled_mm(a, ws[i % copies].t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    if dtype == "int8":
        a = torch.randint(-127, 127, (m, k), device=dev, dtype=torch.int8)
        ws = [torch.randint(-127, 127, (n, k), device=dev, dtype=torch.int8) for _ in range(copies)]
        return lambda i: torch._int_mm(a, ws[i % copies].t())
    raise ValueError(dtype)


def graph_calls(iters: int, out_bytes: int, budget: int = 2 << 30) -> int:
    """Calls to capture in one graph: each captured call keeps its own output alive."""
    return max(10, min(iters, budget // max(out_bytes, 1)))


def _time_graph(torch: Any, fn: Callable[[int], Any], calls: int, warmup: int) -> float:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(warmup):
            fn(i)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            fn(i)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    del graph
    return start.elapsed_time(end) / 1e3 / calls


def _time(torch: Any, fn: Callable[[int], Any], iters: int, warmup: int) -> float:
    for i in range(warmup):
        fn(i)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        fn(i)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / iters


def _iters_for(m: int, n: int, k: int) -> int:
    # Aim for roughly a quarter-second per repeat at a pessimistic 100 TFLOP/s,
    # clamped so M=1 cells still run enough calls to average launch jitter.
    est = flops(m, n, k) / 100e12
    return int(min(2000, max(20, 0.25 / max(est, 1e-7))))


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    props = torch.cuda.get_device_properties(0)
    rows = []
    for name, n, k, m in plan(args.m, not args.no_control):
        for dtype in args.dtypes:
            copies = copies_needed(weight_bytes(n, k, dtype), args.rotate_bytes)
            row: dict[str, Any] = {"shape": name, "n": n, "k": k, "m": m, "dtype": dtype, "copies": copies}
            try:
                fn = _build(torch, dtype, m, n, k, copies)
                iters = _iters_for(m, n, k)
                if args.mode == "graph":
                    iters = graph_calls(iters, m * n * (4 if dtype == "int8" else 2))
                    samples = [_time_graph(torch, fn, iters, warmup=max(3, copies)) for _ in range(args.repeats)]
                else:
                    samples = [_time(torch, fn, iters, warmup=max(3, copies)) for _ in range(args.repeats)]
                med = statistics.median(samples)
                row.update(status="ok", iters=iters, repeats=args.repeats)
                row["cv_pct"] = 100.0 * statistics.pstdev(samples) / statistics.mean(samples)
                row.update(cell_metrics(m, n, k, dtype, med))
            except Exception as exc:  # a shape the kernel refuses is a result
                row.update(status="unsupported", error=f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
            finally:
                fn = None
                torch.cuda.empty_cache()
            rows.append(row)
            print(json.dumps(row), file=sys.stderr, flush=True)

    return {
        "tool": "gemm_decode_shapes.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": props.name,
        "arch": getattr(props, "gcnArchName", None),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "mode": args.mode,
        "rotate_bytes": args.rotate_bytes,
        "repeats": args.repeats,
        "rows": rows,
        "summary": summarize(rows),
    }


def render(result: dict[str, Any]) -> str:
    """Markdown tables for a human; the JSON is the record."""
    out = [
        f"# GEMM at decode shapes — {result['device']} ({result['arch']}), torch {result['torch']}",
        "",
        f"mode={result.get('mode', 'eager')}, rotate_bytes={result['rotate_bytes']} (0 = hot weights), repeats={result['repeats']}, median of repeats.",
        "",
        "| shape | N x K | M | dtype | status | us/call | TFLOP/s | weight GB/s | % of 3.76 TB/s | x bf16 | cv % |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    speed = {(s["shape"], s["m"], s["dtype"]): s["speedup_vs_bf16"] for s in result["summary"]["speedups"]}
    for r in result["rows"]:
        if r["status"] != "ok":
            out.append(
                f"| {r['shape']} | {r['n']}x{r['k']} | {r['m']} | {r['dtype']} | {r['status']} | | | | | | |"
                f" <!-- {r.get('error', '')} -->"
            )
            continue
        x = speed.get((r["shape"], r["m"], r["dtype"]))
        out.append(
            f"| {r['shape']} | {r['n']}x{r['k']} | {r['m']} | {r['dtype']} | ok | {r['us_per_call']:.1f} |"
            f" {r['tflops']:.1f} | {r['weight_gbps']:.0f} | {r['pct_measured_bw']:.0f} |"
            f" {'' if x is None else f'{x:.2f}'} | {r['cv_pct']:.1f} |"
        )
    out += [
        "",
        "## Decode GEMM time per token (sum over all E2B GEMMs, weighted by count)",
        "",
        "| M | dtype | us/token | x bf16 |",
        "| ---: | --- | ---: | ---: |",
    ]
    for p in result["summary"]["per_token"]:
        us = "incomplete — a shape did not run" if p["missing"] else f"{p['us_per_token']:.1f}"
        x = "" if p["speedup_vs_bf16"] is None else f"{p['speedup_vs_bf16']:.2f}"
        out.append(f"| {p['m']} | {p['dtype']} | {us} | {x} |")
    errors = [r for r in result["rows"] if r["status"] != "ok"]
    if errors:
        out += ["", "## Unsupported cells", ""]
        out += [f"- {r['shape']} M={r['m']} {r['dtype']}: `{r['error']}`" for r in errors]
    return "\n".join(out) + "\n"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--m", type=int, nargs="+", default=DECODE_MS, help="batch rows (decode concurrency)")
    p.add_argument("--dtypes", nargs="+", default=DTYPES, choices=DTYPES)
    p.add_argument("--rotate-bytes", type=int, default=1 << 30, help="cold-weight working set; 0 = hot")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--mode", choices=["graph", "eager"], default="graph", help="graph = replay as vLLM decodes; eager = diagnostic"
    )
    p.add_argument("--no-control", action="store_true", help="skip the 8192x8192 control rows")
    p.add_argument("--out", help="write the JSON record here; markdown goes to stdout")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    result = run(args)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    print(render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
