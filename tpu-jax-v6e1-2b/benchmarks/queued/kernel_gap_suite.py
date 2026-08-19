"""Queued investigation of the ~5x attention gap. See kernel-gap.md.

Runs unattended on a v6e-1. Every test is independently guarded and results are
written to ~/kernel_gap_suite.json after each one, so a crash or a preemption
still leaves everything completed so far.

    python3.13 kernel_gap_suite.py

Tests, in the order they answer the most:
  0  calibrate    — what streaming bandwidth does this chip actually reach?
  1  donation     — the leading hypothesis: is the cache being copied per step?
  2  attention    — eager attention in isolation, against the pure read roofline
  3  overheads    — what do the mask and the f32 softmax cost?
  4  profile      — rank operations by device time, keyed on tensor SHAPE
"""
import os, sys, json, time, statistics, traceback, glob, gzip, collections

sys.path.insert(0, os.path.expanduser("~/gemma"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp
from ports.gemma4.jax_e_model import (Gemma4EConfig, Gemma4EModelJAX, init_kv_cache,
                                      make_cached_decode_step, eager_attention_jax,
                                      make_decode_mask)
from ports.gemma4.jax_e_benchmark_sweep import build_benchmark_params
from ports.gemma4.jax_e_benchmark_sweep_v2 import E2B_CONFIG

PUBLISHED_BW = 1640e9
OUT, CAL = {}, [PUBLISHED_BW]
cfg = Gemma4EConfig(**dict(E2B_CONFIG, sliding_window=512))
model = Gemma4EModelJAX(cfg)

n_slide = sum(1 for i in range(cfg.first_kv_shared_layer_idx)
              if cfg.layer_types[i] == "sliding_attention")
n_full = cfg.first_kv_shared_layer_idx - n_slide
KV_EL = (n_slide * cfg.num_key_value_heads * cfg.head_dim * 2
         + n_full * cfg.num_global_key_value_heads * cfg.global_head_dim * 2)


def timed(fn, n=5):
    for _ in range(2):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); jax.block_until_ready(fn())
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def record(name, fn):
    print(f"\n{'='*74}\n### {name}\n{'='*74}", flush=True)
    try:
        OUT[name] = {"ok": True, "detail": fn()}
    except Exception as e:
        OUT[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        traceback.print_exc()
    with open(os.path.expanduser("~/kernel_gap_suite.json"), "w") as f:
        json.dump(OUT, f, indent=1)


def decode_args(B, ctx, dt=jnp.bfloat16):
    total = ctx + 16
    caches = init_kv_cache(cfg, batch_size=B, max_seq_len=total, dtype=dt)
    valid = jnp.zeros((B, total), dtype=jnp.bool_).at[:, :ctx].set(True)
    tok = jnp.ones((B, 1), dtype=jnp.int32)
    lens = jnp.full((B,), ctx, dtype=jnp.int32)
    return caches, valid, tok, lens


# ---------------------------------------------------------------- 0 calibrate
@lambda f: record("0-calibrate", f)
def _():
    got = {}
    for gb in (2, 4):
        n = int(gb * 1e9 / 2)
        x = jnp.ones((n,), dtype=jnp.bfloat16)
        f = jax.jit(lambda a: jnp.sum(a, dtype=jnp.float32))
        ms = timed(lambda: f(x))
        bw = n * 2 / (ms / 1000)
        got[f"{gb}GB"] = bw / 1e9
        print(f"  sum {gb} GB: {ms:7.3f} ms -> {bw/1e9:7.1f} GB/s "
              f"({bw/PUBLISHED_BW*100:.1f}% of published)", flush=True)
        del x, f
    CAL[0] = max(got.values()) * 1e9
    print(f"  calibrated ceiling: {CAL[0]/1e9:.1f} GB/s", flush=True)
    return got


# ---------------------------------------------------------------- 1 donation
@lambda f: record("1-donation", f)
def _():
    """H2: without donation, dynamic_update_slice copies the whole cache.

    Marginal slopes only, and only across LARGE ctx deltas — a small delta makes
    the slope mostly fixed-cost noise, which is what produced the bogus 841 GB/s
    reading in the previous session.
    """
    params = build_benchmark_params(cfg)
    B, CTXS = 32, [2048, 4096, 8192]
    got = {}
    for donate in (False, True):
        kw = {"donate_argnums": (1, 2)} if donate else {}
        step = jax.jit(make_cached_decode_step(model, quant_mode="w4a16",
                                               window_kv=False), **kw)
        label = "donated" if donate else "copying"
        print(f"\n  -- {label} --", flush=True)
        prev, slopes = None, []
        for ctx in CTXS:
            caches, valid, tok, lens = decode_args(B, ctx)
            args = (params, caches, valid, tok, lens, jnp.int32(ctx))
            # Donation invalidates the donated buffers, so a donated step cannot
            # be re-run on the same inputs. Rebuild per call.
            def once():
                c, v, t, l = decode_args(B, ctx)
                return step(params, c, v, t, l, jnp.int32(ctx))
            ms = timed(once if donate else (lambda: step(*args)), n=3)
            kv = B * ctx * KV_EL * 2
            if prev and ms > prev[0] * 1.15:      # demand a real delta
                bw = (kv - prev[1]) / ((ms - prev[0]) / 1000)
                slopes.append(bw / 1e9)
                print(f"    ctx {ctx:>5}: {ms:>7.2f} ms | marginal {bw/1e9:>7.1f} GB/s"
                      f" = {bw/CAL[0]*100:>5.1f}% of calibrated", flush=True)
            else:
                print(f"    ctx {ctx:>5}: {ms:>7.2f} ms | (delta too small to slope)",
                      flush=True)
            prev = (ms, kv)
            del caches
        got[label] = sum(slopes) / len(slopes) if slopes else None
    if got.get("copying") and got.get("donated"):
        print(f"\n  copying {got['copying']:.1f} GB/s -> donated {got['donated']:.1f} GB/s"
              f"  = {got['donated']/got['copying']:.2f}x", flush=True)
        print("  H2 predicts a clear rise. A flat result refutes it.", flush=True)
    return got


# ---------------------------------------------------------------- 2 attention
@lambda f: record("2-attention-isolated", f)
def _():
    """Attention alone, stripped of every other per-step cost."""
    got = {}
    for B, ctx in ((32, 4096), (64, 4096), (32, 8192)):
        H, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q = jnp.ones((B, H, 1, D), dtype=jnp.bfloat16)
        k = jnp.ones((B, Hkv, ctx, D), dtype=jnp.bfloat16)
        v = jnp.ones((B, Hkv, ctx, D), dtype=jnp.bfloat16)
        mask = jnp.zeros((B, 1, 1, ctx), dtype=jnp.float32)
        f = jax.jit(lambda a, b, c, m: eager_attention_jax(a, b, c, mask=m, softcap=0.0))
        ms = timed(lambda: f(q, k, v, mask))
        by = (k.size + v.size) * 2
        bw = by / (ms / 1000)
        got[f"B{B}_ctx{ctx}"] = bw / 1e9
        print(f"  B={B:>3} ctx={ctx:>5}: {ms:>7.3f} ms | {by/1e9:>5.2f} GB read"
              f" | {bw/1e9:>7.1f} GB/s = {bw/CAL[0]*100:>5.1f}% of calibrated", flush=True)
        del q, k, v, mask, f
    return got


# ---------------------------------------------------------------- 3 overheads
@lambda f: record("3-overheads", f)
def _():
    """What do the per-step mask and the f32 softmax actually cost?"""
    B, ctx = 32, 8192
    total = ctx + 16
    valid = jnp.zeros((B, total), dtype=jnp.bool_).at[:, :ctx].set(True)
    fm = jax.jit(lambda v: make_decode_mask(v))
    ms_mask = timed(lambda: fm(valid))
    H, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    q = jnp.ones((B, H, 1, D), dtype=jnp.bfloat16)
    k = jnp.ones((B, Hkv, total, D), dtype=jnp.bfloat16)
    v = jnp.ones((B, Hkv, total, D), dtype=jnp.bfloat16)
    m = fm(valid)
    f_full = jax.jit(lambda a, b, c, mm: eager_attention_jax(a, b, c, mask=mm, softcap=0.0))
    f_nomask = jax.jit(lambda a, b, c: eager_attention_jax(a, b, c, mask=None, softcap=0.0))
    ms_full = timed(lambda: f_full(q, k, v, m))
    ms_nomask = timed(lambda: f_nomask(q, k, v))
    print(f"  mask construction alone : {ms_mask:7.3f} ms", flush=True)
    print(f"  attention with mask     : {ms_full:7.3f} ms", flush=True)
    print(f"  attention without mask  : {ms_nomask:7.3f} ms"
          f"  -> mask costs {ms_full-ms_nomask:+.3f} ms inside attention", flush=True)
    return dict(mask_ms=ms_mask, attn_masked_ms=ms_full, attn_unmasked_ms=ms_nomask)


# ---------------------------------------------------------------- 4 profile
@lambda f: record("4-profile", f)
def _():
    """Rank operations by device time. Keyed on SHAPE: op names are reused across
    compiled buckets, so name-level aggregation mixes different programs."""
    params = build_benchmark_params(cfg)
    B, ctx = 32, 8192
    step = jax.jit(make_cached_decode_step(model, quant_mode="w4a16", window_kv=False))
    caches, valid, tok, lens = decode_args(B, ctx)
    args = (params, caches, valid, tok, lens, jnp.int32(ctx))
    for _ in range(3):
        jax.block_until_ready(step(*args))
    outdir = os.path.expanduser("~/kgs_trace")
    os.system(f"rm -rf {outdir}")
    with jax.profiler.trace(outdir):
        for _ in range(10):
            jax.block_until_ready(step(*args))
    paths = sorted(glob.glob(f"{outdir}/**/*.trace.json.gz", recursive=True))
    if not paths:
        raise RuntimeError(f"no trace written under {outdir}")
    with gzip.open(paths[-1], "rt") as fh:
        trace = json.load(fh)
    agg = collections.defaultdict(lambda: [0, 0.0])
    for e in trace.get("traceEvents", []):
        if e.get("ph") == "X" and "dur" in e:
            key = e["name"]
            agg[key][0] += 1
            agg[key][1] += e["dur"]
    top = sorted(agg.items(), key=lambda kv: -kv[1][1])[:20]
    total_us = sum(v[1] for v in agg.values()) or 1.0
    rows = []
    for name, (n, tot) in top:
        print(f"  {name[:58]:<58} n={n:>5} tot={tot/1e3:>9.2f} ms "
              f"({tot/total_us*100:>5.1f}%)", flush=True)
        rows.append(dict(op=name, n=n, ms=tot / 1e3, pct=tot / total_us * 100))
    return rows


print("\n" + "=" * 74)
for k, v in OUT.items():
    print(f"  {'PASS' if v['ok'] else 'FAIL'}  {k:<24} {v.get('error','')}")
print("KERNEL_GAP_SUITE_DONE", flush=True)
