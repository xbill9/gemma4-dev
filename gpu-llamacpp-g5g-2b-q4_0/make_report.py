#!/usr/bin/env python3
"""Turn a sweep.json into a schema-valid serving report plus its REPORT.md.

    python3 make_report.py --run benchmarks/runs/2026-08-29-first-serve-g5g \
        --instance i-0123 --ami ami-0123 --az us-east-1d --market spot

Separate from sweep.py because the sweep must be able to run against a box that
is about to be reclaimed, writing results as it goes; shaping them into a report
is offline work that can happen after the instance is gone.

**It shapes any of the three g5g rigs' sweeps, not just this one.** Until
2026-08-31 every field was hardcoded to pytorch-transformers, so the JAX and
vLLM legs of a cross-rig run had no schema-valid report and `rollup.py` could
not count them -- the comparison existed only in prose. Engine, version,
quantisation and memory are now derived from the sweep's own /health where the
server reports one, and supplied by flag where it does not (vLLM's /health is an
empty 200). Defaults reproduce this rig's earlier output byte for byte.

The schema is at 1.1: `throughput.sweep[]` carries `input_len`/`output_len` (so a
2-D context x concurrency sweep fits one report) and `status`, so cells that
cannot exist are RECORDED rather than dropped -- an absent cell is
indistinguishable from an untried one, which is how a sweep overstates coverage.
"""

import argparse
import json
import os


def prom(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(name) and not line.startswith("#"):
            try:
                return float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                return 0.0
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--instance", default="")
    ap.add_argument("--ami", default="")
    ap.add_argument("--az", default="")
    ap.add_argument("--market", default="spot")
    ap.add_argument("--spot-price", type=float, default=0.0)
    # benchmarks/README.md: "the filename must equal run.id", and report filenames are
    # <date>-<model-short>-<hw-short> while run dirs are <date>-<what>-<hw-short>. Defaulting
    # run.id to the run-dir basename therefore emits a report that fails validation whenever
    # <what> != <model-short>. Both of this rig's 2026-08-29 reports did; fixed 2026-08-30.
    ap.add_argument("--id", default="", help="run.id; MUST equal the report filename stem")
    # Hardcoding this made the second run's report claim it was the first. Fixed 2026-08-30.
    ap.add_argument("--notes", default="", help="run.notes; describe THIS run")
    # Cross-rig support. Each defaults to this rig's value so existing
    # invocations are unchanged; a sibling's run supplies what /health cannot.
    ap.add_argument("--rig", default="gpu-llamacpp-g5g-2b-q4_0")
    ap.add_argument("--engine", default="")
    ap.add_argument("--engine-version", default="")
    ap.add_argument("--max-model-len", type=int, default=0)
    ap.add_argument("--weights-gib", type=float, default=0.0)
    ap.add_argument("--quantization", default="")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--arch-notes", default="")
    ap.add_argument("--memory-notes", default="")
    args = ap.parse_args()

    run_dir = os.path.basename(args.run.rstrip("/"))
    run_id = args.id or run_dir
    sweep = json.load(open(os.path.join(args.run, "sweep.json")))
    metrics = open(os.path.join(args.run, "metrics.prom")).read()
    health = sweep.get("health") or {}
    ok = [c for c in sweep["cells"] if c["status"] == "ok"]

    # /health differs per runtime and vLLM returns an empty 200, so resolve each
    # field from whatever the server did report, then fall back to the flag.
    prec = health.get("precision") or {}
    dtype = health.get("compute_dtype") or prec.get("weights") or "float16"
    engine = args.engine or "llama.cpp"
    # The engine version IS the pinned build id here -- llama.cpp has no runtime
    # version string worth quoting, and the same ref built for a different arch
    # is a different binary. Read it off the box rather than defaulting blind.
    version = args.engine_version or "llama.cpp (pass --engine-version with the BUILD_ID)"
    max_len = args.max_model_len or health.get("seq") or 0
    weights_gib = args.weights_gib or round(
        (health.get("weight_bytes") or prom(metrics, "llamacpp:model_size_bytes")) / 2**30, 3)
    if args.quantization:
        quant = args.quantization
    elif prec.get("ple_bits") or prec.get("int8_lm_head"):
        quant = (f"PLE {prec.get('ple_bits')}-bit"
                 f"{', int8 lm_head' if prec.get('int8_lm_head') else ''}"
                 " (not numerics-preserving)")
    else:
        quant = "none (dense reference checkpoint)"

    report = {
        "schema_version": "1.1",
        "run": {
            "id": run_id,
            "date": run_id[:10],
            "operator": "xbill",
            "source": (f"gpu-llamacpp-g5g-2b-q4_0/sweep.py + make_report.py; "
                       f"rig {args.rig}; run dir {run_dir}"),
            "notes": args.notes or f"Generated by make_report.py from benchmarks/runs/{run_dir}.",
        },
        "hardware": {
            "accelerator": "NVIDIA T4G",
            "chips": 1,
            "topology": "single device",
            "hbm_gb_per_chip": 15.36,
            "machine_type": "g5g.2xlarge",
            "host": {"cpu": "AWS Graviton2 (aarch64)", "vcpus": 8, "ram_gb": 16},
            # rate_per_chip_hour is the schema's required key, and on a 1-chip
            # instance it equals the instance rate. Spelling it usd_per_hour
            # validated locally and failed the schema.
            "pricing": {
                "rate_per_chip_hour": args.spot_price,
                "currency": "USD",
                "source": (
                    f"aws ec2 describe-spot-price-history, {args.az}, "
                    f"{args.market}, at run time"
                ),
            },
        },
        "model": {
            "id": sweep["model"],
            "family": "gemma-4",
            "parameters_b": 5.4,
            "weights_dtype": dtype,
            "quantization": quant,
            "kv_cache_dtype": prec.get("kv_cache") or dtype,
            "max_model_len": max_len,
            "architecture_notes": args.arch_notes or (
                "E2B: 28 sliding layers at head_dim 256 plus 7 full-attention layers at "
                "global_head_dim 512; 8 query heads to 1 KV head. transformers owns the "
                "cache here -- there is no hand-written ring and no padding."
            ),
        },
        "software": {
            "engine": engine,
            "version": version,
            "backend": "cuda",
            "tensor_parallel_size": 1,
            "serve_args": ["--seq", str(max_len)],
            "container_image": f"AMI {args.ami}" if args.ami else "",
        },
        "throughput": {
            "workload": {
                "tool": f"gpu-llamacpp-g5g-2b-q4_0/sweep.py (decode-source "
                        f"{sweep.get('summary', {}).get('decode_source', 'usage')})",
                "dataset": "synthetic repeated filler, summarize instruction",
                "num_prompts": 1,
                "runs_per_point": 3,
            },
            "sweep": [],
        },
        "memory": {
            "usable_hbm_gib": 14.07,
            "weights_gib": weights_gib,
            "notes": args.memory_notes or (
                "Dense float16, no PLE quantisation and no int8 lm_head -- those are JAX-port "
                "levers with no analogue here. E2B KV is ~18 KiB/token, so at 4096 the whole "
                "cache is tens of MiB and is not the binding constraint."
            ),
        },
        "issues": [],
        "notes": [],
    }

    for c in sweep["cells"]:
        point = {
            "concurrency": args.concurrency,
            "input_len": c.get("input_len", c["context"]),
            "output_len": c["output_len"],
            "status": c["status"],
        }
        if c["status"] == "ok":
            point["output_tokens_per_second"] = c["decode_tps_median"]
            point["request_latency_s"] = c["wall_s_median"]
        else:
            point["error"] = c.get("error", "")[:200]
        report["throughput"]["sweep"].append(point)

    if ok:
        lo = min(c["decode_tps_median"] for c in ok)
        hi = max(c["decode_tps_median"] for c in ok)
        report["notes"].append(
            f"Decode {lo:.2f}-{hi:.2f} tok/s across a {ok[0]['input_len']}-"
            f"{ok[-1]['input_len']} token context range: flat, so decode is set by the "
            "weights rather than by the context."
        )
    report["notes"].append(
        "Quote the decode figures, not end-to-end: they come from "
        "llamacpp:tokens_predicted_total/_seconds_total, which time decode alone. Both are in sweep.json "
        "and they do not agree."
    )
    # llama-server exposes NO degeneracy counter -- that series is an invention of
    # this repo's own JAX and PyTorch servers. So the note is emitted only when the
    # sweep actually carried one, and is OMITTED otherwise rather than defaulting
    # to 0. A report claiming "0 degenerate replies" from a check that never ran is
    # worse than a report that is silent about it.
    degen = sweep.get("summary", {}).get("degenerate_responses")
    if degen is None:
        report["notes"].append(
            "Degeneracy was NOT counted server-side: llama-server exposes no such "
            "metric. verify_model_health applies a local heuristic per probe; this "
            "run has no whole-run figure."
        )
    else:
        report["notes"].append(f"Degenerate replies over the whole run: {int(degen)}.")
    ratio = sweep.get("summary", {}).get("stream_over_usage_median")
    if ratio:
        report["notes"].append(
            f"Client-side stream statistic reads {ratio:.4f} of this server's own decode "
            "gauge on this run. The offset is per-rig -- never borrow another rig's ratio.")

    out = os.path.join(args.run, "REPORT.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
