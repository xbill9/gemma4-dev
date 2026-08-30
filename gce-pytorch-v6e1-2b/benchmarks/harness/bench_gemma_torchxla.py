#!/usr/bin/env python3.12
"""Greedy-decode benchmark for Gemma 4 on TPU through PyTorch/XLA.

Measures what this rig actually serves: transformers + torch_xla on one v6e chip.
Not comparable to a `vllm bench serve` number from a sibling rig — there is no
continuous batching and no paged KV here, so "concurrency" is a static batch
dimension, not a stream count.

XLA compiles per shape, so every point is run once to warm the graph and the
warmup is reported separately as compile time rather than folded into the
measurement. Shapes are held static across decode steps (a StaticCache and a
fixed-width prefill) because a changing shape recompiles mid-run and the cost
lands in whichever step happened to trigger it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:  # E2B's checkpoint declares Gemma4ForConditionalGeneration, a multimodal wrapper
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from transformers import StaticCache
except ImportError:  # older transformers
    StaticCache = None


def _sync() -> None:
    """Blocks until every queued XLA op has actually executed.

    mark_step() only *flushes* the graph; without the wait the host races ahead
    and every timing below collapses to the enqueue cost.
    """
    xm.mark_step()
    xm.wait_device_ops()


def load(model_id: str, dtype: torch.dtype, device) -> tuple:
    tok = AutoTokenizer.from_pretrained(model_id)
    t0 = time.perf_counter()
    # E2B ships as Gemma4ForConditionalGeneration (a multimodal wrapper), so the
    # causal-LM auto class does not always map. Fall back to the image-text-to-text
    # class and drive its text tower; this benchmark is text-only either way.
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except (ValueError, KeyError):
        if AutoModelForImageTextToText is None:
            raise
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    model = model.to(device).eval()
    _sync()
    to_device_s = time.perf_counter() - t0
    return tok, model, load_s, to_device_s


@torch.no_grad()
def run_point(model, device, batch: int, input_len: int, output_len: int, vocab: int, max_cache: int) -> dict:
    """One (batch, input_len, output_len) cell: prefill once, then decode step by step.

    Returns wall-clock splits plus the derived rates. Token ids are random within
    the vocab: greedy decode's cost does not depend on which tokens they are, and
    a fixed synthetic prompt keeps the shape exactly `input_len` wide.
    """
    ids = torch.randint(0, vocab, (batch, input_len), dtype=torch.long, device=device)

    # The attention mask is allocated ONCE at full cache width and only ever has its
    # values flipped, never its shape changed. Growing it with torch.cat each step —
    # the obvious way to write this — gives every decode step a new shape, so XLA
    # recompiles on every token and TPOT comes out ~1000x too high. Shape constant,
    # values variable, one compile.
    mask = torch.zeros((batch, max_cache), dtype=torch.long, device=device)
    mask[:, :input_len] = 1

    cache = None
    if StaticCache is not None:
        cache = StaticCache(
            config=model.config, max_batch_size=batch,
            max_cache_len=max_cache, device=device, dtype=model.dtype,
        )

    # --- prefill (this is TTFT) ---
    _sync()
    t0 = time.perf_counter()
    out = model(
        input_ids=ids,
        attention_mask=mask[:, :input_len],
        past_key_values=cache,
        use_cache=True,
        cache_position=torch.arange(input_len, device=device),
    )
    next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    _sync()
    prefill_s = time.perf_counter() - t0

    cache = out.past_key_values
    # --- decode ---
    # Everything that varies across steps lives in a device tensor. Indexing with a
    # Python int (mask[:, pos] = 1) or building a fresh torch.tensor([pos]) each step
    # bakes a different constant into each step's graph, and XLA recompiles every
    # token: measured at ~9.2 s/token before this was fixed, against ~10 ms after.
    # Same shapes AND same constants is what buys the single compile.
    step_s: list[float] = []
    cache_pos = torch.tensor([input_len], device=device)
    one = torch.ones((batch, 1), dtype=mask.dtype, device=device)
    for _ in range(output_len):
        mask.scatter_(1, cache_pos.unsqueeze(0).expand(batch, 1), one)
        t0 = time.perf_counter()
        out = model(
            input_ids=next_tok,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_pos,
        )
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        cache_pos += 1
        _sync()
        step_s.append(time.perf_counter() - t0)
        cache = out.past_key_values

    decode_s = sum(step_s)
    total_s = prefill_s + decode_s
    out_tokens = batch * output_len
    return {
        "concurrency": batch,
        "input_len": input_len,
        "output_len": output_len,
        "status": "ok",
        "ttft_ms": {"median": round(prefill_s * 1000, 1)},
        "tpot_ms": {
            "median": round(statistics.median(step_s) * 1000, 2),
            "p99": round(sorted(step_s)[max(0, int(len(step_s) * 0.99) - 1)] * 1000, 2),
        },
        "output_tok_per_s": round(out_tokens / decode_s, 1),
        "total_tok_per_s": round((batch * input_len + out_tokens) / total_s, 1),
        "per_stream_tok_per_s": round(output_len / decode_s, 1),
        "_prefill_s": round(prefill_s, 4),
        "_decode_s": round(decode_s, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "google/gemma-4-E2B-it"))
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--concurrency", default="1,4,8,16,32")
    ap.add_argument("--profile-batch", type=int, default=8, help="batch size to capture an xprof trace for; 0 disables")
    ap.add_argument("--profile-dir", default="/opt/profiles")
    ap.add_argument("--out", default="/opt/bench_result.json")
    args = ap.parse_args()

    device = xm.xla_device()
    # The profiler is a server the trace client connects back to, so it has to be
    # listening before any trace_detached call — starting it at capture time races
    # the capture and silently yields an empty trace.
    profiler_port = int(os.environ.get("XLA_PROFILER_PORT", "9012"))
    profiler_server = None
    try:
        profiler_server = xp.start_server(profiler_port)
    except Exception as e:  # profiling must never fail the benchmark
        print(f"WARNING: could not start the profiler server: {e}", flush=True)

    batches = [int(b) for b in args.concurrency.split(",") if b.strip()]
    max_cache = args.input_len + args.output_len + 8

    cfg = AutoConfig.from_pretrained(args.model)
    vocab = getattr(cfg, "vocab_size", None) or getattr(cfg.get_text_config(), "vocab_size", 262144)

    print(f"device={device} chips={torch_xla.runtime.global_runtime_device_count()} model={args.model}", flush=True)
    tok, model, load_s, to_device_s = load(args.model, torch.bfloat16, device)
    print(f"weights loaded in {load_s:.1f}s, moved to device in {to_device_s:.1f}s", flush=True)

    sweep, compile_s = [], {}
    for b in batches:
        try:
            # Warmup: this run pays for XLA compilation of both the prefill and
            # the decode graph at this shape. Its cost is reported, not measured.
            t0 = time.perf_counter()
            run_point(model, device, b, args.input_len, min(4, args.output_len), vocab, max_cache)
            compile_s[b] = round(time.perf_counter() - t0, 1)
            print(f"batch {b}: warmed in {compile_s[b]}s", flush=True)

            point = run_point(model, device, b, args.input_len, args.output_len, vocab, max_cache)
            point["_warmup_s"] = compile_s[b]
            sweep.append(point)
            print(f"  -> {json.dumps(point)}", flush=True)
        except RuntimeError as e:
            msg = str(e)[:300]
            print(f"batch {b}: FAILED {msg}", flush=True)
            sweep.append({
                "concurrency": b, "input_len": args.input_len, "output_len": args.output_len,
                "status": "infeasible" if "memory" in msg.lower() or "oom" in msg.lower() else "failed",
                "error": msg,
            })

    # One xprof trace, so the run leaves something tensorboard can open.
    profile_path = None
    if args.profile_batch and profiler_server is not None:
        try:
            os.makedirs(args.profile_dir, exist_ok=True)
            xp.trace_detached(
                f"localhost:{profiler_port}", args.profile_dir, duration_ms=6000
            )
            run_point(model, device, args.profile_batch, args.input_len, 32, vocab, max_cache)
            profile_path = args.profile_dir
            print(f"xprof trace written to {profile_path}", flush=True)
        except Exception as e:  # profiling must never fail the benchmark
            print(f"WARNING: profile capture failed: {e}", flush=True)

    result = {
        "model": args.model,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_xla": getattr(torch_xla, "__version__", "unknown"),
        "chips": torch_xla.runtime.global_runtime_device_count(),
        "weights_load_s": round(load_s, 1),
        "to_device_s": round(to_device_s, 1),
        "compile_s_by_batch": compile_s,
        "workload": {"input_len": args.input_len, "output_len": args.output_len, "runs_per_point": 1},
        "sweep": sweep,
        "profile_dir": profile_path,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
