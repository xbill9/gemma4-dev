#!/usr/bin/env python3
"""Name the transient allocation that makes a prefill OOM.

Run it with the serving process stopped -- the compile wants the machine to
itself, and on this rig that is the machine you are typing on:

    python3 profile_prefill.py --model google/gemma-4-E2B-it \\
        --sweep 512,1024,2048,4096

WHY THIS EXISTS: docs/larger-models-on-t4g.md records per-request transients that
scale with model size -- E2B 4.52 GiB, E4B 5.25 GiB, 12B 12.61 GiB -- and could
not attribute any of them to a tensor. The OOM message gives a size and no name.

The trick is that **the allocation does not have to succeed to be measured**. An
ahead-of-time compile runs buffer assignment and allocates nothing, so
`memory_analysis().temp_size_in_bytes` reports the transient for a shape that
cannot run, and the optimized HLO names every buffer that makes it up. So a shape
that 500s in production can be analysed here without ever OOMing.

`--sweep` is the part that identifies the culprit rather than merely sizing it:
a buffer quadratic in the bucket is an attention score matrix, one linear in it
is an activation or a logits tensor, and one flat in it is a weight materialised
in dense form. Reading the growth is usually faster than reading the HLO.

THIS SCRIPT IS THE ONE THAT GAINS THE MOST FROM RUNNING HERE, and the reason is
that it never needed a device in the first place. `memory_analysis()` reports
buffer assignment for a shape that cannot run, so the whole analysis is an
ahead-of-time compile -- on the cloud rigs that meant "you can size a shape that
OOMs without OOMing", and here it means the entire tool works with no accelerator
at all and no capacity cycle spent. A prefill sizing question about a GPU rig can
be answered from this rig, as long as you remember the ANSWER is the transient in
bytes, which is a property of the graph, and not the verdict about whether it
fits, which is a property of the device you did not use.

What DOES change is the budget line. A CPU JAX device exposes no allocator, so
there is no device limit to print; the honest comparison here is against host
RAM, and it is printed as such.
"""
from __future__ import annotations

import argparse
import os
import re
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL_NAME", "google/gemma-4-E2B-it"))
    p.add_argument("--quant-mode", default="auto", choices=["auto", "w4a16", "fp16"])
    p.add_argument("--ple-bits", type=int, default=4, choices=[0, 4, 8])
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Part of the compiled shape; the cache is sized bucket + this.")
    p.add_argument("--sweep", default="512,1024,2048",
                   help="Comma-separated prompt buckets to compile for.")
    p.add_argument("--top", type=int, default=15, help="Largest HLO buffers to list.")
    p.add_argument("--window-kv", default="auto", choices=["auto", "on", "off"])
    return p.parse_args()


# HLO element types that actually appear in this port, with their byte widths.
_WIDTH = {"pred": 1, "s8": 1, "u8": 1, "s16": 2, "u16": 2, "bf16": 2, "f16": 2,
          "s32": 4, "u32": 4, "f32": 4, "s64": 8, "u64": 8, "f64": 8, "c64": 8,
          "c128": 16, "f8e4m3fn": 1, "f8e5m2": 1}

# `%name = f32[1,8,2048,2048]{3,2,1,0} fusion(...)` -- capture name, type, dims.
_INSTR = re.compile(
    r"^\s*(?:%)?([\w.\-]+)\s*=\s*([a-z0-9]+)\[([\d,]*)\]", re.MULTILINE)


def hlo_buffers(text: str):
    """[(bytes, name, shape_str)] for every instruction output in the module.

    This is instruction OUTPUT size, not the allocator's view: XLA reuses buffers,
    so the sum far exceeds peak. What it is good for is spotting the one tensor
    whose size is the same order as the failed allocation, which is the question
    the OOM message leaves open.
    """
    out = []
    for name, dtype, dims in _INSTR.findall(text):
        width = _WIDTH.get(dtype)
        if width is None:
            continue
        n = 1
        for d in (dims.split(",") if dims else []):
            n *= int(d)
        out.append((n * width, name, f"{dtype}[{dims}]"))
    out.sort(reverse=True)
    return out


def human(nbytes: float) -> str:
    return f"{nbytes / 2 ** 30:8.3f} GiB" if nbytes >= 2 ** 30 else f"{nbytes / 2 ** 20:8.1f} MiB"


def analyse(engine, bucket: int, args, jnp, jax):
    from ports.gemma4.jax_e_model import pad_to_bucket

    ids = jnp.ones((1, bucket), jnp.int32)
    padded, valid = pad_to_bucket(ids, pad_token_id=0)
    if padded.shape[1] != bucket:
        # The ladder moved: compile for the bucket it actually chose, and say so,
        # rather than silently reporting a different shape than the one asked for.
        print(f"  (bucket {bucket} -> {padded.shape[1]} by the ladder)")
        bucket = padded.shape[1]

    lowered = engine._jit_prefill.lower(
        model=engine.model, prompt_ids=padded, prompt_valid=valid,
        params=engine.params, max_new_tokens=args.max_new_tokens,
        quant_mode=engine.quant_mode, cache_dtype=engine.cache_dtype,
        window_kv=engine.window_kv,
    )
    compiled = lowered.compile()
    mem = compiled.memory_analysis()
    temp = getattr(mem, "temp_size_in_bytes", 0)
    argb = getattr(mem, "argument_size_in_bytes", 0)
    outb = getattr(mem, "output_size_in_bytes", 0)
    print(f"\n=== bucket {bucket} (+{args.max_new_tokens} new) ===")
    print(f"  args   {human(argb)}   output {human(outb)}   TEMP {human(temp)}")
    return bucket, temp, compiled


def main():
    args = parse_args()
    import jax
    import jax.numpy as jnp

    from jax_engine import JaxGemmaEngine

    engine = JaxGemmaEngine(
        model_id=args.model, quant_mode=args.quant_mode,
        max_model_len=args.max_model_len, ple_bits=args.ple_bits,
        window_kv={"auto": None, "on": True, "off": False}[args.window_kv],
    )
    engine.load()
    print(f"model {args.model}  quant={engine.quant_mode}  ple_bits={args.ple_bits}")
    print(f"resident weights {engine.weight_bytes / 1e9:.3f} GB on {engine.device}")
    stats = engine.memory_stats()
    if stats.get("has_device_allocator"):
        print(f"device limit {stats['hbm_bytes_limit'] / 1e9:.3f} GB   "
              f"in use {stats['hbm_bytes_in_use'] / 1e9:.3f} GB")
    else:
        # No allocator to ask. Printing "limit 0.000 GB" would read as a device
        # holding nothing rather than as a device with no budget, and those are
        # different facts -- the same distinction /metrics makes by omitting the
        # HBM series here instead of exporting a zero.
        mem = {}
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    key, _, rest = line.partition(":")
                    parts = rest.split()
                    if parts:
                        mem[key] = int(parts[0]) * 1024
        except OSError:
            pass
        print(f"no device allocator (CPU): process RSS "
              f"{stats.get('host_rss_bytes', 0) / 1e9:.3f} GB, host RAM "
              f"{mem.get('MemTotal', 0) / 1e9:.3f} GB total / "
              f"{mem.get('MemAvailable', 0) / 1e9:.3f} GB available, swap free "
              f"{mem.get('SwapFree', 0) / 1e9:.3f} GB")
        print("A transient that exceeds available RAM here does NOT raise -- it "
              "swaps. Compare the numbers below against MemAvailable yourself.")

    rows, last = [], None
    for bucket in [int(b) for b in args.sweep.split(",") if b.strip()]:
        try:
            b, temp, compiled = analyse(engine, bucket, args, jnp, jax)
        except Exception as exc:                       # compile can still fail
            print(f"\n=== bucket {bucket} === COMPILE FAILED: {type(exc).__name__}: "
                  f"{str(exc)[:200]}")
            continue
        rows.append((b, temp))
        last = compiled

    if len(rows) > 1:
        print("\n--- how the transient scales ---")
        print(f"{'bucket':>8} {'temp':>14} {'x prev':>8} {'bytes/token':>12}")
        prev = None
        for b, t in rows:
            ratio = f"{t / prev:.2f}" if prev else "-"
            print(f"{b:8d} {human(t):>14} {ratio:>8} {t / b / 1e6:11.3f} MB")
            prev = t
        print("Quadratic growth points at an attention score matrix; linear at an "
              "activation or logits tensor; flat at a weight materialised dense.")

    if last is not None:
        print(f"\n--- largest instruction outputs in the optimized HLO (bucket {rows[-1][0]}) ---")
        try:
            text = last.as_text()
        except Exception as exc:
            print(f"  unavailable: {type(exc).__name__}: {exc}")
            return 0
        seen = set()
        shown = 0
        for nbytes, name, shape in hlo_buffers(text):
            if shape in seen:
                continue
            seen.add(shape)
            print(f"  {human(nbytes)}  {shape:<34} {name[:44]}")
            shown += 1
            if shown >= args.top:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
