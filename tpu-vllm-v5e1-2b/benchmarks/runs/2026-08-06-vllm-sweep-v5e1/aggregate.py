#!/usr/bin/env python3
"""Aggregate this run's raw `vllm bench serve` dumps into summary.json, tables.md, and the
schema-1.1 report at ../../reports/2026-08-06-gemma4-e2b-v5e1.json.

Everything in STACK below was read off the running deployment (container logs, /version,
gcloud describe, the Cloud Billing catalog) — nothing is estimated. Fields that were not
measured are absent rather than guessed, which is why there is no machine_type.

Stdlib only.
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
REPORT = os.path.join(HERE, "..", "..", "reports", "2026-08-06-gemma4-e2b-v5e1.json")

# The matrix that was planned. The 32768 row is recorded as infeasible rather than dropped:
# it exceeds the configured max_model_len, so it cannot exist on this deployment. It was
# determined analytically from max_model_len, NOT by attempting it — see REPORT.md.
CONCURRENCIES = [1, 4, 16, 64]
CONTEXTS = [128, 1024, 8192, 32768]
MAX_MODEL_LEN = 16384

STACK = {
    "engine_version": "0.26.1rc1.dev125+ga7a204cc6",
    "container_image": "vllm/vllm-tpu:nightly",
    "backend": "tpu-inference (JAX, flax_nnx)",
    "zone": "us-west4-a",
    "instance": "tpu-2B-v5e1-devops-agent",
    "provisioning": "spot",
    # Cloud Billing Catalog, Compute Engine service 6F81-5844-456A, SKU
    # "TpuV5e attached to Spot Preemptible VMs running in Las Vegas", usageType Preemptible.
    "rate_per_chip_hour": 0.5779,
    # From the container logs at steady state.
    "hbm_total_gib": 15.75,
    "hbm_cap_gib": 14.49,
    "weights_gib": 8.97,
    "kv_blocks": 10043,
    "kv_layers": 15,
    "resident_kv_tokens": 321376,
    "time_to_healthy_s": 986,
    "engine_init_s": 814.01,
    "compile_s": 738.03,
}

SERVE_ARGS = [
    "--max-model-len 16384",
    "--tensor-parallel-size 1",
    "--disable_chunked_mm_input",
    "--max_num_batched_tokens 4096",
    '--limit-mm-per-prompt {"image":4,"audio":1}',
    "--enable-auto-tool-choice",
    "--tool-call-parser gemma4",
    "--reasoning-parser gemma4",
]


def stats(res, metric):
    out = {}
    for stat in ("mean", "median", "p90", "p99"):
        v = res.get(f"{stat}_{metric}_ms")
        if isinstance(v, (int, float)):
            out[stat] = round(v, 2)
    return out


def load_cells():
    """(ctx, conc) -> sweep point, measured cells only."""
    cells = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        name = os.path.basename(path)[: -len(".json")]
        ctx_s, conc_s = name.split("-")
        ctx, conc = int(ctx_s[1:]), int(conc_s[1:])
        with open(path) as fh:
            res = json.load(fh)
        point = {"concurrency": conc, "input_len": ctx, "output_len": 128, "status": "ok"}
        for key, src in (
            ("request_rate_rps", "request_throughput"),
            ("output_tok_per_s", "output_throughput"),
            ("total_tok_per_s", "total_token_throughput"),
        ):
            v = res.get(src)
            if isinstance(v, (int, float)):
                point[key] = round(v, 2)
        for key, metric in (("ttft_ms", "ttft"), ("tpot_ms", "tpot"), ("itl_ms", "itl")):
            s = stats(res, metric)
            if s:
                point[key] = s
        med = point.get("tpot_ms", {}).get("median")
        if med:
            point["per_stream_tok_per_s"] = round(1000 / med, 1)
        # Per-request arrays dropped; they are one element per prompt and dominate the size.
        point["raw"] = {k: v for k, v in res.items() if not isinstance(v, list)}
        cells[(ctx, conc)] = point
    return cells


def build_sweep(cells):
    sweep = []
    for ctx in CONTEXTS:
        for conc in CONCURRENCIES:
            if (ctx, conc) in cells:
                sweep.append(cells[(ctx, conc)])
            elif ctx + 128 > MAX_MODEL_LEN:
                sweep.append(
                    {
                        "concurrency": conc,
                        "input_len": ctx,
                        "output_len": 128,
                        "status": "infeasible",
                        "error": f"input_len {ctx} + output_len 128 exceeds configured "
                        f"max_model_len {MAX_MODEL_LEN}; not attempted",
                    }
                )
    return sweep


def cost_rows(sweep):
    rate = STACK["rate_per_chip_hour"]
    ok = [p for p in sweep if p["status"] == "ok" and p.get("output_tok_per_s")]
    if not ok:
        return []
    peak = max(ok, key=lambda p: p["output_tok_per_s"])
    single = min(ok, key=lambda p: (p["concurrency"], p["input_len"]))
    rows = []
    for label, p in (
        (f"saturation (ctx {peak['input_len']}, c={peak['concurrency']})", peak),
        (f"single-stream (ctx {single['input_len']}, c=1)", single),
    ):
        tok_s = p["output_tok_per_s"]
        rows.append(
            {
                "operating_point": label,
                "output_tok_per_s": tok_s,
                "usd": round(rate / (tok_s * 3600) * 1_000_000, 2),
            }
        )
    return rows


def write_summary(cells, sweep):
    summary = {
        "google/gemma-4-E2B-it": {
            "boot": {
                "model": "google/gemma-4-E2B-it",
                "max_model_len": MAX_MODEL_LEN,
                "time_to_healthy_s": STACK["time_to_healthy_s"],
            },
            "cells": {
                f"c{p['input_len']}-u{p['concurrency']}": {k: v for k, v in p.items() if k != "raw"} for p in sweep
            },
        }
    }
    path = os.path.join(RESULTS, "summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {os.path.relpath(path, HERE)} ({len(sweep)} cells)")


def write_tables(sweep):
    by = {(p["input_len"], p["concurrency"]): p for p in sweep}

    def grid(title, pick, nd=0):
        lines = [f"### {title}", "", "| ctx \\ users | " + " | ".join(str(u) for u in CONCURRENCIES) + " |"]
        lines.append("|---" * (len(CONCURRENCIES) + 1) + "|")
        for ctx in CONTEXTS:
            row = [f"| {ctx} "]
            for u in CONCURRENCIES:
                p = by.get((ctx, u))
                if p is None:
                    row.append("|— ")
                elif p["status"] != "ok":
                    row.append(f"|_{p['status']}_ ")
                else:
                    v = pick(p)
                    row.append(f"|{v:,.{nd}f} " if isinstance(v, (int, float)) else "|— ")
            lines.append("".join(row) + "|")
        lines.append("")
        return lines

    out = [
        "# 2026-08-06 v5e-1 sweep tables — gemma-4-E2B-it",
        "",
        "Generated by `aggregate.py`. Do not hand-edit.",
        "",
        "Cells marked _infeasible_ exceed the configured `max_model_len` and were not attempted.",
        "",
    ]
    out += grid("Aggregate output tok/s", lambda p: p.get("output_tok_per_s"))
    out += grid("Total (prefill+decode) tok/s", lambda p: p.get("total_tok_per_s"))
    out += grid("Median TTFT ms", lambda p: p.get("ttft_ms", {}).get("median"), nd=1)
    out += grid("p99 TTFT ms", lambda p: p.get("ttft_ms", {}).get("p99"), nd=1)
    out += grid("Median TPOT ms", lambda p: p.get("tpot_ms", {}).get("median"), nd=2)
    out += grid("Per-stream tok/s (1000 / median TPOT)", lambda p: p.get("per_stream_tok_per_s"), nd=1)
    path = os.path.join(HERE, "tables.md")
    with open(path, "w") as fh:
        fh.write("\n".join(out))
    print(f"wrote {os.path.relpath(path, HERE)}")


def write_report(sweep):
    kv_bytes_per_token = STACK["kv_layers"] * 2 * 256 * 2  # layers x (k,v) x head_dim x bf16
    kv_gib = round(STACK["kv_blocks"] * 32 * kv_bytes_per_token / 1024**3, 2)
    report = {
        "schema_version": "1.1",
        "run": {
            "id": "2026-08-06-gemma4-e2b-v5e1",
            "date": "2026-08-06",
            "source": "benchmarks/runs/2026-08-06-vllm-sweep-v5e1/",
            "notes": "Single run per cell, no repeats — treat differences under a few percent as "
            "noise. The 32768-context row is recorded infeasible from max_model_len, not attempted.",
        },
        "hardware": {
            "accelerator": "tpu-v5e",
            "chips": 1,
            "topology": "1x1",
            "hbm_gb_per_chip": STACK["hbm_total_gib"],
            "host": {
                "cloud": "gcp",
                "zone": STACK["zone"],
                "provisioning": STACK["provisioning"],
                "instance_name": STACK["instance"],
            },
            "pricing": {
                "currency": "USD",
                "rate_per_chip_hour": STACK["rate_per_chip_hour"],
                "source": "Cloud Billing Catalog, Compute Engine service 6F81-5844-456A, SKU "
                "'TpuV5e attached to Spot Preemptible VMs running in Las Vegas' "
                "(usageType Preemptible), read 2026-08-06",
            },
        },
        "model": {
            "id": "google/gemma-4-E2B-it",
            "family": "gemma-4",
            "parameters_b": 2,
            "weights_dtype": "bfloat16",
            "quantization": "none",
            "max_model_len": MAX_MODEL_LEN,
            "architecture_notes": "KV sharing: the engine allocated KV for 15 layers only, with a "
            "single 256-dim KV head per layer — 15 KiB per token in bf16.",
        },
        "software": {
            "engine": "vllm",
            "version": STACK["engine_version"],
            "container_image": STACK["container_image"],
            "backend": STACK["backend"],
            "tensor_parallel_size": 1,
            "serve_args": SERVE_ARGS,
        },
        "throughput": {
            "workload": {
                "tool": "vllm bench serve",
                "dataset": "random",
                "output_len": 128,
                "runs_per_point": 1,
            },
            "sweep": sweep,
        },
        "memory": {
            "usable_hbm_gib": STACK["hbm_cap_gib"],
            "weights_gib": STACK["weights_gib"],
            "kv_cache_gib": kv_gib,
            "kv_bytes_per_token": kv_bytes_per_token,
            "resident_kv_tokens": STACK["resident_kv_tokens"],
            "notes": f"Engine reported {STACK['hbm_total_gib']} GiB total HBM, capped at "
            f"{STACK['hbm_cap_gib']} GiB. kv_cache_gib is derived from the logged block count "
            f"({STACK['kv_blocks']} blocks x 32 tokens x {STACK['kv_layers']} layers), not read directly.",
        },
        "startup": {
            "time_to_healthy_s": STACK["time_to_healthy_s"],
            "engine_init_s": STACK["engine_init_s"],
            "compile_s": STACK["compile_s"],
            "notes": "From container start to 'Application startup complete'. Compilation is 91% of "
            "engine init on this stack — a cold start is dominated by XLA/JAX compile, not weight load "
            "(weights downloaded in 9.8 s).",
        },
        "cost": {
            "per_m_output_tokens": cost_rows(sweep),
            "notes": "Serving cost only, at the spot rate above. Excludes the cold start, which on "
            "spot capacity is repaid after every preemption.",
        },
        "notes": [
            "Provisioned as a spot tpu-vm, not a Queued Resource: spot takes no --max-run-duration, "
            "so the node bills until preempted or destroyed.",
            "Raw /v1/completions returns an empty completion on -it models; all measurements here use "
            "the OpenAI chat/completions path that `vllm bench serve` drives.",
        ],
    }
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print(f"wrote {os.path.relpath(REPORT, HERE)}")


if __name__ == "__main__":
    cells = load_cells()
    sweep = build_sweep(cells)
    write_summary(cells, sweep)
    write_tables(sweep)
    write_report(sweep)
    n_ok = sum(1 for p in sweep if p["status"] == "ok")
    print(f"\n{n_ok} measured, {len(sweep) - n_ok} infeasible, {len(sweep)} cells total")
