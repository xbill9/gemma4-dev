"""Re-validate the counter-intuitive results, one configuration per process.

Several findings from 2026-07-29 are surprising enough to deserve an independent
measurement rather than a rerun:

  * int8 KV is FASTER than bf16 (1.22-1.78x). Quantization normally costs time.
  * int8 KV has slightly BETTER perplexity than bf16 (0.9891x).
  * Quantizing 53% of the model (the PLE table) changes aggregate throughput by
    +0.26% — capacity and step time cancel almost exactly.
  * int4 PLE is faster than int8 PLE at B=1 (112 vs 98 tok/s).
  * The decode ceiling varies ~7% with context rather than being a constant.

Three methodological weaknesses in the original measurements are fixed here:

  1. ONE CONFIG PER PROCESS. The originals ran every configuration sequentially
     in a single process, allocating and freeing multi-GB buffers throughout.
     HBM fragmentation and allocator state carry over, and the ordering happened
     to put the quantized configurations last. Invoke this script once per
     config (see the `--all` driver) so each starts from a clean device.
  2. ERROR BARS. The originals reported a median of 3-5 samples with no spread.
     A 3% difference between medians of 3 is not a result. This takes 15 samples
     and reports median, min, max and IQR, so "faster" can be distinguished from
     "noisier".
  3. BUFFER DONATION. Every prior benchmark built its own
     `jax.jit(make_cached_decode_step(...))` with no donation, so
     `dynamic_update_slice` may have been copying the whole cache each step. If
     that dominated, every step-time ratio measured so far is suspect. Both
     paths are measured.

Usage:
    python3.13 revalidate.py --config ple-bf16/kv-bf16/nodonate --ctx 8192 --batch 32
    python3.13 revalidate.py --all          # spawns one subprocess per config
"""
import argparse, itertools, json, os, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/gemma"))

PLE_OPTS = ["bf16", "8", "4"]
KV_OPTS = ["bf16", "int8"]
DONATE_OPTS = ["nodonate", "donate"]
N_SAMPLES = 15


def all_configs():
    for p, k, d in itertools.product(PLE_OPTS, KV_OPTS, DONATE_OPTS):
        yield f"ple-{p}/kv-{k}/{d}"


def run_one(config: str, ctx: int, batch: int) -> dict:
    import jax, jax.numpy as jnp
    from ports.gemma4.jax_e_model import (Gemma4EConfig, Gemma4EModelJAX, init_kv_cache,
                                          make_cached_decode_step, quantize_ple_table)
    from ports.gemma4.jax_e_benchmark_sweep import build_benchmark_params
    from ports.gemma4.jax_e_benchmark_sweep_v2 import E2B_CONFIG

    ple, kv, donate = config.split("/")
    ple_bits = 0 if ple == "ple-bf16" else int(ple.split("-")[1])
    cache_dt = jnp.bfloat16 if kv == "kv-bf16" else jnp.int8
    donating = donate == "donate"

    cfg = Gemma4EConfig(**dict(E2B_CONFIG, sliding_window=512))
    model = Gemma4EModelJAX(cfg)
    params = build_benchmark_params(cfg)
    if ple_bits:
        params = quantize_ple_table(params, bits=ple_bits,
                                    group_size=cfg.hidden_size_per_layer_input)

    def tree_bytes(p):
        tot, stack = 0, [p]
        while stack:
            n = stack.pop()
            if isinstance(n, dict):
                stack.extend(n.values())
            elif hasattr(n, "size"):
                tot += n.size * n.dtype.itemsize
        return tot

    step = jax.jit(make_cached_decode_step(model, quant_mode="w4a16", window_kv=False),
                   **({"donate_argnums": (1, 2)} if donating else {}))

    def fresh():
        total = ctx + 16
        caches = init_kv_cache(cfg, batch_size=batch, max_seq_len=total, dtype=cache_dt)
        valid = jnp.zeros((batch, total), dtype=jnp.bool_).at[:, :ctx].set(True)
        tok = jnp.ones((batch, 1), dtype=jnp.int32)
        lens = jnp.full((batch,), ctx, dtype=jnp.int32)
        return caches, valid, tok, lens

    # A donated buffer is invalidated by the call that consumes it, so a donating
    # step cannot be replayed on the same inputs. Both paths therefore rebuild
    # inputs per sample, which keeps the comparison fair rather than charging the
    # donating path an allocation the other one avoids.
    caches, valid, tok, lens = fresh()
    for _ in range(3):
        c, v, t, l = fresh()
        jax.block_until_ready(step(params, c, v, t, l, jnp.int32(ctx)))

    samples = []
    for _ in range(N_SAMPLES):
        c, v, t, l = fresh()
        jax.block_until_ready(c[0][0])
        t0 = time.perf_counter()
        jax.block_until_ready(step(params, c, v, t, l, jnp.int32(ctx)))
        samples.append((time.perf_counter() - t0) * 1000)

    samples.sort()
    q1, q3 = samples[len(samples) // 4], samples[3 * len(samples) // 4]
    med = statistics.median(samples)
    return dict(config=config, ctx=ctx, batch=batch,
                weights_gb=round(tree_bytes(params) / 1e9, 3),
                median_ms=round(med, 3), min_ms=round(samples[0], 3),
                max_ms=round(samples[-1], 3), iqr_ms=round(q3 - q1, 3),
                rel_iqr_pct=round((q3 - q1) / med * 100, 2),
                agg_tok_s=round(batch * 1000 / med, 1), n=len(samples))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=os.path.expanduser("~/revalidate.json"))
    a = ap.parse_args()

    if a.all:
        results = []
        for cfgname in all_configs():
            cmd = [sys.executable, os.path.abspath(__file__), "--config", cfgname,
                   "--ctx", str(a.ctx), "--batch", str(a.batch)]
            print(f"\n>>> {cfgname}", flush=True)
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            if line:
                r = json.loads(line[0][7:])
                results.append(r)
                print(f"    {r['median_ms']:>8.3f} ms  +/-{r['rel_iqr_pct']:>5.2f}% IQR"
                      f"  | {r['agg_tok_s']:>9.1f} tok/s | weights {r['weights_gb']} GB",
                      flush=True)
            else:
                print(f"    FAILED: {(p.stderr or p.stdout).strip().splitlines()[-1][:100]}",
                      flush=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)

        # Only claim a difference when it exceeds the measured noise.
        print("\n" + "=" * 78)
        print("DONATION EFFECT (same ple/kv, donate vs nodonate)")
        by = {r["config"]: r for r in results}
        for p, k in itertools.product(PLE_OPTS, KV_OPTS):
            nd, d = by.get(f"ple-{p}/kv-{k}/nodonate"), by.get(f"ple-{p}/kv-{k}/donate")
            if nd and d:
                ratio = nd["median_ms"] / d["median_ms"]
                noise = max(nd["rel_iqr_pct"], d["rel_iqr_pct"]) / 100
                verdict = "REAL" if abs(ratio - 1) > 2 * noise else "within noise"
                print(f"  ple-{p:<5} kv-{k:<5}: {nd['median_ms']:>8.3f} -> "
                      f"{d['median_ms']:>8.3f} ms = {ratio:>5.3f}x  [{verdict}]")
        print("\nint8 KV vs bf16 KV (same ple, same donation)")
        for p, dn in itertools.product(PLE_OPTS, DONATE_OPTS):
            b, i = by.get(f"ple-{p}/kv-bf16/{dn}"), by.get(f"ple-{p}/kv-int8/{dn}")
            if b and i:
                ratio = b["median_ms"] / i["median_ms"]
                noise = max(b["rel_iqr_pct"], i["rel_iqr_pct"]) / 100
                verdict = "REAL" if abs(ratio - 1) > 2 * noise else "within noise"
                print(f"  ple-{p:<5} {dn:<9}: {b['median_ms']:>8.3f} -> "
                      f"{i['median_ms']:>8.3f} ms = {ratio:>5.3f}x  [{verdict}]")
        print("=" * 78)
        print("REVALIDATE_DONE", flush=True)
    else:
        r = run_one(a.config, a.ctx, a.batch)
        print("RESULT " + json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
