#!/usr/bin/env python3
"""Profile one decode step of the JAX engine: per-kernel GPU time, cost, memory.

Run it ON the instance (it needs the GPU and the serving payload):

    systemctl stop jax-g4dn            # the profiler needs the device to itself
    set -a; . /opt/jax-g4dn/env; set +a
    PYTHONPATH=/opt/jax-g4dn/app python3.14 profile_decode.py \\
        --model google/gemma-4-E2B-it --ple-bits 0

The per-kernel table is the useful part. Everything else here is context for it.

WHY THIS EXISTS: on 2026-08-23 this rig measured 12.4 tok/s and every plausible
explanation (memory bandwidth, dtype policy, XLA flags, dispatch overhead, KV
cache, buffer donation) was wrong. The kernel table answered it in one run --
55% of decode time was `wrapped_convert`, and every matmul kernel was `float`
rather than `__half`. See docs/profiling-recipes.md for the traps.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import shutil
import sys
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL_NAME", "google/gemma-4-E2B-it"))
    p.add_argument("--ple-bits", type=int, default=0, choices=[0, 4, 8])
    p.add_argument("--quant-mode", default="auto", choices=["auto", "w4a16", "fp16"])
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--prompt-tokens", type=int, default=16)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--logdir", default="/tmp/jaxtrace")
    p.add_argument("--top", type=int, default=25)
    # Added 2026-08-26: int8_lm_head could not be profiled at all before, which
    # mattered because the LM-head conversion (wrapped_convert_61) was 14.3% of
    # decode on its own -- the single kernel this flag exists to remove.
    p.add_argument("--int8-lm-head", action="store_true",
                   help="Quantize the LM head to int8. NOT numerics-preserving.")
    return p.parse_args()


def gpu_kernels(logdir):
    """(total_us, Counter(name->us), Counter(name->calls)) from a JAX trace dir.

    The trace is a Chrome-format JSON. Lane names arrive as separate metadata
    events, so the device lane has to be resolved by (pid, tid) -- filtering on
    the event name alone mixes host dispatch in with device execution and will
    make Python overhead look like GPU time.
    """
    paths = glob.glob(os.path.join(logdir, "**", "*.trace.json.gz"), recursive=True)
    events = []
    for path in paths:
        with gzip.open(path, "rt") as handle:
            events.extend(json.load(handle).get("traceEvents", []))
    lanes = {}
    for ev in events:
        if ev.get("ph") == "M" and ev.get("name") == "thread_name":
            lanes[(ev.get("pid"), ev.get("tid"))] = ev.get("args", {}).get("name", "")
    dur, calls = collections.Counter(), collections.Counter()
    for ev in events:
        if ev.get("ph") != "X" or not ev.get("dur"):
            continue
        lane = lanes.get((ev.get("pid"), ev.get("tid")), "")
        if "Stream" in lane and "Compute" in lane:
            dur[ev.get("name", "?")] += ev["dur"]
            calls[ev.get("name", "?")] += 1
    return sum(dur.values()), dur, calls


def classify(name):
    """Flag the kernels that answer 'why is this slow'."""
    low = name.lower()
    if "convert" in low:
        return "DTYPE CONVERSION"
    if "__half" in name or "fp16" in low or "hgemm" in low:
        return "fp16"
    if "sgemm" in low or "gemvx" in low or ", float," in name:
        return "fp32"
    return ""


def main():
    args = parse_args()
    import jax
    import jax.numpy as jnp

    from jax_engine import JaxGemmaEngine
    from ports.gemma4.jax_e_model import COMPUTE_DTYPE, pad_to_bucket

    engine = JaxGemmaEngine(
        model_id=args.model, kv_cache_dtype="auto", quant_mode=args.quant_mode,
        max_model_len=args.max_model_len, ple_bits=args.ple_bits,
        int8_lm_head=args.int8_lm_head,
    )
    engine.load()

    dtypes = collections.Counter()

    def walk(tree):
        for value in (tree.values() if isinstance(tree, dict) else []):
            if hasattr(value, "dtype"):
                dtypes[str(value.dtype)] += 1
            else:
                walk(value)

    walk(engine.params)
    print(f"model            {args.model}")
    print(f"COMPUTE_DTYPE    {jnp.dtype(COMPUTE_DTYPE).name}")
    print(f"param dtypes     {dict(dtypes)}")
    print(f"resident         {engine.weight_bytes / 1e9:.3f} GB")
    print(f"quant_mode       {engine.quant_mode}   window_kv={engine.window_kv}")

    ids = jnp.arange(args.prompt_tokens, dtype=jnp.int32)
    padded, valid_mask = pad_to_bucket(ids[None, :])
    bucket = int(padded.shape[1])
    _, caches, valid = jax.block_until_ready(engine._jit_prefill(
        model=engine.model, prompt_ids=padded, prompt_valid=valid_mask,
        params=engine.params, max_new_tokens=64, quant_mode=engine.quant_mode,
        cache_dtype=engine.cache_dtype, window_kv=engine.window_kv,
    ))
    tok = jnp.ones((1, 1), jnp.int32)
    lens = valid_mask.sum(axis=1).astype(jnp.int32)

    compiled = engine._decode_step.lower(
        engine.params, caches, valid, tok, lens, jnp.int32(bucket)).compile()
    cost = compiled.cost_analysis()
    if isinstance(cost, (list, tuple)):
        cost = cost[0]
    mem = compiled.memory_analysis()
    flops = cost.get("flops") or 0
    print(f"\nstatic: {flops / 1e9:.2f} GFLOP/token   "
          f"temp {getattr(mem, 'temp_size_in_bytes', 0) / 1e9:.3f} GB   "
          f"args {getattr(mem, 'argument_size_in_bytes', 0) / 1e9:.3f} GB")
    print("NOTE: 'bytes accessed' from cost_analysis ignores fusion and counts a "
          "one-row gather of a multi-GB table as a full read. It is an upper "
          "bound, not traffic. Trust the kernel table below instead.")

    # Buffer donation invalidates the caches, so each step must consume the
    # PREVIOUS step's outputs. Reusing `caches` raises "Donation requested for
    # invalid buffer" on the second call.
    cur_c, cur_v = caches, valid
    for i in range(5):
        cur_c, cur_v, _ = engine._decode_step(
            engine.params, cur_c, cur_v, tok, lens + i, jnp.int32(bucket + i))
    jax.block_until_ready((cur_c, cur_v))

    started = time.perf_counter()
    for i in range(args.steps):
        cur_c, cur_v, out = engine._decode_step(
            engine.params, cur_c, cur_v, tok, lens + i, jnp.int32(bucket + i))
        jax.block_until_ready((cur_c, cur_v, out))
    wall_ms = (time.perf_counter() - started) / args.steps * 1e3

    shutil.rmtree(args.logdir, ignore_errors=True)
    with jax.profiler.trace(args.logdir):
        for i in range(args.steps):
            cur_c, cur_v, out = engine._decode_step(
                engine.params, cur_c, cur_v, tok, lens + i, jnp.int32(bucket + i))
            jax.block_until_ready((cur_c, cur_v, out))

    total_us, dur, calls = gpu_kernels(args.logdir)
    if not total_us:
        print("\nNo GPU-lane events found; is another process holding the device?")
        return 1
    print(f"\nwall {wall_ms:.3f} ms/token -> {1000 / wall_ms:.2f} tok/s"
          f"   (GPU lane {total_us / 1e3 / args.steps:.3f} ms/token)")
    print(f"\n{'ms/token':>9} {'%':>6} {'calls':>7}  kernel")
    for name, us in dur.most_common(args.top):
        tag = classify(name)
        suffix = f"   <-- {tag}" if tag else ""
        print(f"{us / 1e3 / args.steps:9.3f} {us / total_us * 100:5.1f}% "
              f"{calls[name] / args.steps:7.1f}  {name[:64]}{suffix}")

    conv = sum(us for n, us in dur.items() if classify(n) == "DTYPE CONVERSION")
    f32 = sum(us for n, us in dur.items() if classify(n) == "fp32")
    f16 = sum(us for n, us in dur.items() if classify(n) == "fp16")
    print(f"\nconversion {conv / total_us * 100:5.1f}%   "
          f"fp32 kernels {f32 / total_us * 100:5.1f}%   "
          f"fp16 kernels {f16 / total_us * 100:5.1f}%")
    if conv / total_us > 0.15:
        print("\n>>> Conversion is a large share of decode. Check whether the "
              "parameter dtypes above match COMPUTE_DTYPE: bf16 weights on a "
              "pre-Ampere GPU are converted before every matmul.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
