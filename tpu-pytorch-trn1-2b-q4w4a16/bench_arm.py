#!/usr/bin/env python3
"""Measure one checkpoint on the Neuron engine: parity, decode speed, host memory.

    python3 bench_arm.py --model <id> --neff-dir /opt/neff --out arm.json \
        --batches 1,4,16 --steps 96

Loads the checkpoint once, then for each batch size traces both graphs at that
size and:

  parity  greedy decode on the device and on the CPU through the same wrappers,
          token for token, with prompt 0 repeated in the last slot so a mask
          that leaks between streams shows up as two different answers to one
          prompt (B >= 2 only; B=1 checks the device against the CPU).
  speed   a fixed `--steps` decode, no early exit, timed after a warm call at the
          same shape: ms per step and aggregate tokens per second. This is the
          method of benchmarks/runs/2026-07-27-inf2-qat-e2b-batch4 in
          tpu-pytorch-inf2-2b, so the numbers line up with its 47.3 / 157.1 /
          442.7 tok/s.

`--phase check` runs parity alone and is safe beside other work on the chip;
`--phase time` reloads the saved graphs and times alone. A sweep compiles and
checks two checkpoints at once, one per NeuronCore, and then times every one
with the chip otherwise idle: decode is bound by HBM bandwidth, which the two
cores share.

Writes one JSON. Text is recorded beside the ids: a matching pair of empty
outputs is a parity pass and a broken engine (docs/neuron-jax-quirks.md, quirk 1).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

from torch_generate import AUTOCAST_TYPE, PARITY_PROMPTS, NeuronGemmaEngine


def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return 0.0


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--neff-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batches", default="1,4,16")
    p.add_argument("--steps", type=int, default=96)
    p.add_argument("--max-total", type=int, default=256)
    p.add_argument("--prompt-bucket", type=int, default=32)
    p.add_argument("--parity-tokens", type=int, default=24)
    p.add_argument("--phase", choices=("all", "check", "time"), default="all",
                   help="check = compile + parity only (safe beside other device work); "
                        "time = decode timing only, from saved graphs, chip otherwise idle")
    p.add_argument("--repeats", type=int, default=3, help="timed decodes per cell; median kept")
    args = p.parse_args()

    result = {"model": args.model, "autocast": AUTOCAST_TYPE, "steps": args.steps,
              "max_total": args.max_total, "prompt_bucket": args.prompt_bucket,
              "cells": []}
    engine = NeuronGemmaEngine(model_id=args.model, batch=1, max_total=args.max_total,
                               prompt_bucket=args.prompt_bucket, neff_dir=args.neff_dir)
    t0 = time.monotonic()
    engine.load()
    result["load_s"] = round(time.monotonic() - t0, 1)
    result["rss_after_load_gb"] = round(rss_gb(), 2)
    result["quantization"] = engine.quantization

    for batch in [int(b) for b in args.batches.split(",")]:
        cell = {"batch": batch}
        engine.batch = batch
        engine.pre = engine.dec = None
        t0 = time.monotonic()
        engine.compile()
        cell["compile_s"] = round(time.monotonic() - t0, 1)
        pre_path, dec_path = engine._neff_paths()
        cell["neff_gb"] = round((os.path.getsize(pre_path) + os.path.getsize(dec_path)) / 1e9, 2)

        # parity: distinct prompts, prompt 0 repeated in the last slot
        layout = [i % len(PARITY_PROMPTS) for i in range(batch)]
        if batch >= 2:
            layout[-1] = 0
        prompts = [engine.encode_chat([{"role": "user", "content": PARITY_PROMPTS[i]}])
                   for i in layout]
        if args.phase != "time":
            cell["parity"] = parity(engine, prompts, batch, args.parity_tokens)
        if args.phase != "check":
            cell.update(timing(engine, [prompts[0]] * batch, batch, args.steps, args.repeats))
        cell["peak_rss_gb"] = round(peak_rss_gb(), 2)
        result["cells"].append(cell)
        print(json.dumps(cell), flush=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)

    result["peak_rss_gb"] = round(peak_rss_gb(), 2)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    ok = all(c["parity"]["seq_match"] and c["parity"]["dup_isolation"]
             and not c["parity"]["empty"] for c in result["cells"] if "parity" in c)
    print("PARITY_ALL:", ok)
    return 0 if ok else 1


def parity(engine, prompts, batch, tokens):
    """Device against CPU through the same wrappers, plus slot isolation."""
    dev = engine.generate_greedy(prompts, tokens)
    traced = (engine.pre, engine.dec)
    engine.pre, engine.dec = engine.pre_module, engine.dec_module
    try:
        cpu = engine.generate_greedy(prompts, tokens)
    finally:
        engine.pre, engine.dec = traced
    texts = [engine.tokenizer.decode([t for t in s if t not in engine.eos_ids],
                                     skip_special_tokens=True) for s in dev]
    return {
        "seq_match": dev == cpu,
        "slots_matching_cpu": sum(d == c for d, c in zip(dev, cpu, strict=True)),
        "dup_isolation": batch < 2 or dev[-1] == dev[0],
        "texts": texts[: min(batch, 4)],
        "empty": sum(1 for t in texts if not t.strip()),
    }


def timing(engine, prompts, batch, steps, repeats):
    """Fixed-length decode, no early exit, warmed at the same shape; median of repeats."""
    engine.generate_greedy(prompts, 2)
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        engine.generate_greedy(prompts, steps)
        runs.append(time.perf_counter() - t0)
    elapsed = sorted(runs)[len(runs) // 2]
    return {
        "ms_per_step": round(elapsed / steps * 1000, 2),
        "ms_per_step_runs": [round(r / steps * 1000, 2) for r in runs],
        "aggregate_tok_s": round(batch * steps / elapsed, 1),
        "per_stream_tok_s": round(steps / elapsed, 1),
    }


if __name__ == "__main__":
    raise SystemExit(main())
