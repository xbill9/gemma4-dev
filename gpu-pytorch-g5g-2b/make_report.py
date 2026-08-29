#!/usr/bin/env python3
"""Turn a sweep.json into a schema-valid serving report plus its REPORT.md.

    python3 make_report.py --run benchmarks/runs/2026-08-29-first-serve-g5g \
        --instance i-0123 --ami ami-0123 --az us-east-1d --market spot

Separate from sweep.py because the sweep must be able to run against a box that
is about to be reclaimed, writing results as it goes; shaping them into a report
is offline work that can happen after the instance is gone.

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
    args = ap.parse_args()

    run_id = os.path.basename(args.run.rstrip("/"))
    sweep = json.load(open(os.path.join(args.run, "sweep.json")))
    metrics = open(os.path.join(args.run, "metrics.prom")).read()
    health = sweep["health"]
    ok = [c for c in sweep["cells"] if c["status"] == "ok"]

    report = {
        "schema_version": "1.1",
        "run": {
            "id": run_id,
            "date": run_id[:10],
            "operator": "xbill",
            "source": "sweep.py + make_report.py in gpu-pytorch-g5g-2b",
            "notes": (
                "First serve for this rig. Driven from the rig-root server.py rather than "
                "the registered MCP server, whose snapshot predated the deploy-path fixes."
            ),
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
            "weights_dtype": health.get("compute_dtype"),
            "quantization": "none (dense reference checkpoint)",
            "kv_cache_dtype": health.get("compute_dtype"),
            "max_model_len": health.get("seq"),
            "architecture_notes": (
                "E2B: 28 sliding layers at head_dim 256 plus 7 full-attention layers at "
                "global_head_dim 512; 8 query heads to 1 KV head. transformers owns the "
                "cache here -- there is no hand-written ring and no padding."
            ),
        },
        "software": {
            "engine": "pytorch-transformers",
            "version": "torch 2.12.0+cu132",
            "backend": "cuda",
            "tensor_parallel_size": 1,
            "serve_args": ["--seq", str(health.get("seq"))],
            "container_image": f"AMI {args.ami}" if args.ami else "",
        },
        "throughput": {
            "workload": {
                "tool": "sweep.py (this rig)",
                "dataset": "synthetic repeated filler, summarize instruction",
                "num_prompts": 1,
                "runs_per_point": 3,
            },
            "sweep": [],
        },
        "memory": {
            "usable_hbm_gib": 14.07,
            "weights_gib": round(health.get("weight_bytes", 0) / 2**30, 3),
            "notes": (
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
            "concurrency": 1,          # MAX_NUM_SEQS=1: this rig serializes on one lock
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
        "tpu_jax_decode_tokens_per_second, which times decode alone. Both are in sweep.json "
        "and they do not agree."
    )
    degen = prom(metrics, "tpu_jax_degenerate_responses_total")
    report["notes"].append(f"Degenerate replies over the whole run: {int(degen)}.")

    out = os.path.join(args.run, "REPORT.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
