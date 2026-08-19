"""Cross-check: do the measured step times match the bytes we claim to move?

Every throughput number in this repo comes from a wall-clock timer. That is one
instrument. This is a second, independent one: count the bytes a decode step must
read (weights once, plus the whole resident KV), divide by the measured time, and
compare the implied bandwidth against the chip's published HBM figure. A
configuration claiming to move more bytes per second than the hardware can is
measuring something other than what it thinks.

It also accounts for where HBM actually goes at the ceiling. We know 18.0
KiB/token of KV and ~6.6 GB of weights; if free HBM divided by bytes-per-token
does not predict the measured ceiling, the difference is temporaries, and that
gap is the `g(B)` term that made every naive "bytes freed / bytes per token"
prediction in this project too optimistic.

Published v6e-1 figures used as reference (not measured here): 32 GB HBM at
1640 GB/s. Achieved-bandwidth percentages are only as good as that constant.
"""
import os, sys, time, json, statistics
sys.path.insert(0, os.path.expanduser("~/gemma"))
import jax, jax.numpy as jnp
from ports.gemma4.jax_e_model import (Gemma4EConfig, Gemma4EModelJAX, init_kv_cache,
                                      make_cached_decode_step)
from ports.gemma4.jax_e_benchmark_sweep import build_benchmark_params
from ports.gemma4.jax_e_benchmark_sweep_v2 import E2B_CONFIG

HBM_BYTES = 32e9
HBM_BW = 1640e9          # published v6e-1 HBM bandwidth, GB/s
cfg = Gemma4EConfig(**dict(E2B_CONFIG, sliding_window=512))
model = Gemma4EModelJAX(cfg)
params = build_benchmark_params(cfg)


def tree_bytes(p):
    tot, stack = 0, [p]
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            stack.extend(n.values())
        elif hasattr(n, "size"):
            tot += n.size * n.dtype.itemsize
    return tot


W = tree_bytes(params)
n_slide = sum(1 for i in range(cfg.first_kv_shared_layer_idx)
              if cfg.layer_types[i] == "sliding_attention")
n_full = cfg.first_kv_shared_layer_idx - n_slide
KV_EL = (n_slide * cfg.num_key_value_heads * cfg.head_dim * 2
         + n_full * cfg.num_global_key_value_heads * cfg.global_head_dim * 2)
print(f"weights {W/1e9:.2f} GB | KV {KV_EL} elements/token "
      f"({KV_EL*2/1024:.2f} KiB/token bf16, {KV_EL*1.008/1024:.2f} KiB int8)", flush=True)
print(f"reference: {HBM_BYTES/1e9:.0f} GB HBM at {HBM_BW/1e9:.0f} GB/s\n", flush=True)

ROWS = []
CASES = [
    ("bf16", jnp.bfloat16, 2, [(512, 512), (512, 1296), (8192, 32), (8192, 87)]),
    ("int8", jnp.int8, 1.008, [(512, 512), (512, 2432), (8192, 32), (8192, 172)]),
]
for name, dt, bpe, points in CASES:
    step = jax.jit(make_cached_decode_step(model, quant_mode="w4a16", window_kv=False))
    for ctx, B in points:
        try:
            total = ctx + 16
            caches = init_kv_cache(cfg, batch_size=B, max_seq_len=total, dtype=dt)
            valid = jnp.zeros((B, total), dtype=jnp.bool_).at[:, :ctx].set(True)
            tok = jnp.ones((B, 1), dtype=jnp.int32)
            lens = jnp.full((B,), ctx, dtype=jnp.int32)
            args = (params, caches, valid, tok, lens, jnp.int32(ctx))
            for _ in range(2):
                jax.block_until_ready(step(*args))
            ts = []
            for _ in range(5):
                t0 = time.perf_counter(); jax.block_until_ready(step(*args))
                ts.append((time.perf_counter() - t0) * 1000)
            ms = statistics.median(ts)

            # Bytes a correct decode step must read: every weight once, plus the
            # KV of every resident sequence once. Activations are negligible at
            # one token per sequence.
            kv_bytes = B * ctx * KV_EL * bpe
            moved = W + kv_bytes
            achieved = moved / (ms / 1000)
            floor_ms = moved / HBM_BW * 1000
            ROWS.append(dict(cache=name, ctx=ctx, B=B, ms=ms,
                             kv_gb=kv_bytes/1e9, moved_gb=moved/1e9,
                             achieved_gbs=achieved/1e9, floor_ms=floor_ms,
                             pct=achieved/HBM_BW*100))
            print(f"  {name:>4} ctx{ctx:>5} B{B:>5}: {ms:>7.2f} ms | KV {kv_bytes/1e9:>5.2f} GB"
                  f" | moved {moved/1e9:>5.2f} GB | {achieved/1e9:>7.1f} GB/s"
                  f" = {achieved/HBM_BW*100:>5.1f}% of peak"
                  f" | roofline {floor_ms:>6.2f} ms", flush=True)
            del caches
        except Exception as e:
            print(f"  {name:>4} ctx{ctx:>5} B{B:>5}: FAILED {str(e)[:50]}", flush=True)

print("\n--- where HBM goes at the measured ceiling ---", flush=True)
for r in ROWS:
    if (r["cache"], r["ctx"], r["B"]) in {("bf16", 512, 1296), ("bf16", 8192, 87),
                                          ("int8", 512, 2432), ("int8", 8192, 172)}:
        acct = W + r["kv_gb"]*1e9
        print(f"  {r['cache']:>4} ctx{r['ctx']:>5} B{r['B']:>5}: weights {W/1e9:.2f}"
              f" + KV {r['kv_gb']:.2f} = {acct/1e9:.2f} GB of {HBM_BYTES/1e9:.0f}"
              f"  -> {(HBM_BYTES-acct)/1e9:.2f} GB unaccounted (temporaries)"
              f" = {(HBM_BYTES-acct)/HBM_BYTES*100:.0f}%", flush=True)

with open(os.path.expanduser("~/roofline.json"), "w") as f:
    json.dump(ROWS, f, indent=1)
print("\nROOFLINE_DONE", flush=True)
