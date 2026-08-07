"""Measure the W4A16 dequantize penalty ON THE 31B, directly.

The claim "decode is ~6x off because of the int4 unpack" currently rests on an
inference from E2B's dequant_at_load result (1.60x faster while reading 3.7x more
bytes => ~5.9x per byte). E2B is a different model, and the 31B cannot run
dequant_at_load at full depth because dense bf16 is ~62 GB against 33.55 GB of HBM.

But a TRUNCATED 31B can. At N layers the dense footprint is ~0.96 GB/layer plus the
2.82 GB embedding, so N=15 is ~17 GB and fits comfortably. Comparing w4a16 against
dense at the SAME truncated depth, on the same chip, isolates the dequantize cost
in this model's own geometry rather than E2B's.

Reports decode step latency both ways, plus the implied weight-bytes/s for each, so
the result can be checked against the ~1640 GB/s HBM roofline.

    python3.13 ports/gemma4/jax_31b_dequant_cost.py --layers 8,15
"""

import argparse
import copy
import json
import os
import statistics
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp

from ports.gemma4.jax_e_model import (
    Gemma4EModelJAX,
    dequantize_params_to_dense,
    make_cached_decode_step,
    pad_to_tpu_v6e_bucket,
    prefill_with_kv_cache,
)
from ports.gemma4.jax_31b_port import Streaming31BEngine

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"


def truncated(eng, n_layers: int):
    cfg = copy.copy(eng.config)
    cfg.num_hidden_layers = n_layers
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = list(eng.config.layer_types)[:n_layers]
    params = {k: v for k, v in eng.params.items() if not k.startswith("layer_")}
    for i in range(n_layers):
        params[f"layer_{i}"] = eng.params[f"layer_{i}"]
    return Gemma4EModelJAX(cfg), params


def tree_bytes(tree):
    return sum(int(x.size) * int(x.dtype.itemsize)
               for x in jax.tree_util.tree_leaves(tree))


def decode_ms(model, params, quant_mode, S=1024, repeats=7):
    ids = jnp.ones((1, S), dtype=jnp.int32)
    padded, valid = pad_to_tpu_v6e_bucket(ids)
    last_logits, caches, vmask = jax.block_until_ready(jax.jit(
        prefill_with_kv_cache,
        static_argnames=("model", "max_new_tokens", "quant_mode",
                         "cache_dtype", "window_kv"))(
        model=model, prompt_ids=padded, prompt_valid=valid, params=params,
        max_new_tokens=8, quant_mode=quant_mode, window_kv=True))
    step = jax.jit(make_cached_decode_step(model, quant_mode=quant_mode,
                                           window_kv=True))
    tok = jnp.argmax(last_logits, axis=-1, keepdims=True)
    lens = vmask[:, :S].sum(axis=1).astype(jnp.int32)

    def run():
        return step(params, caches, vmask, tok, lens, jnp.int32(S))

    jax.block_until_ready(run())
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(run())
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples), min(samples), max(samples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layers", default="8,15")
    ap.add_argument("--S", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--hbm-gbs", type=float, default=1640.0,
                    help="published v6e HBM bandwidth, for the roofline column")
    ap.add_argument("--json-out", default="gemma4_31b_dequant_cost.json")
    args = ap.parse_args()

    eng = Streaming31BEngine(model_id=args.model, quant_mode="w4a16",
                             max_model_len=2048, window_kv=True)
    eng.load()
    print(f"\nfull 31B w4a16 weights: {eng.weight_bytes / 1e9:.3f} GB\n")
    print(f"  {'layers':>6s} {'mode':>8s} {'weights':>10s} {'step':>10s} "
          f"{'tok/s':>8s} {'GB/s':>9s} {'% roofline':>11s}")

    rows = []
    for n in [int(x) for x in args.layers.split(",")]:
        model, qparams = truncated(eng, n)
        for mode in ("w4a16", "dense"):
            try:
                if mode == "dense":
                    params = jax.device_put(dequantize_params_to_dense(qparams))
                    jax.block_until_ready(params)
                    qm = "fp16"
                else:
                    params, qm = qparams, "w4a16"
                wb = tree_bytes(params)
                med, lo, hi = decode_ms(model, params, qm, S=args.S,
                                        repeats=args.repeats)
                gbs = wb / (med / 1000.0) / 1e9
                print(f"  {n:6d} {mode:>8s} {wb / 1e9:8.2f} GB {med:8.2f} ms "
                      f"{1000.0 / med:8.1f} {gbs:8.1f} {100 * gbs / args.hbm_gbs:10.1f}%")
                rows.append({"layers": n, "mode": mode,
                             "weight_gb": round(wb / 1e9, 3),
                             "step_ms": round(med, 3),
                             "step_min_ms": round(lo, 3), "step_max_ms": round(hi, 3),
                             "tok_s": round(1000.0 / med, 2),
                             "weight_gbs": round(gbs, 1),
                             "pct_roofline": round(100 * gbs / args.hbm_gbs, 1),
                             "status": "OK"})
                if mode == "dense":
                    del params
            except Exception as exc:
                msg = str(exc).split("\n")[0][:90]
                print(f"  {n:6d} {mode:>8s}  FAILED: {msg}")
                rows.append({"layers": n, "mode": mode, "status": "FAILED",
                             "error": msg})

    # Pair up and report the penalty
    print()
    for n in sorted({r["layers"] for r in rows}):
        q = next((r for r in rows if r["layers"] == n and r["mode"] == "w4a16"
                  and r.get("status") == "OK"), None)
        d = next((r for r in rows if r["layers"] == n and r["mode"] == "dense"
                  and r.get("status") == "OK"), None)
        if q and d:
            per_byte = (d["weight_gbs"] / q["weight_gbs"])
            print(f"  {n} layers: dense is {q['step_ms'] / d['step_ms']:.2f}x faster "
                  f"in wall time while reading {d['weight_gb'] / q['weight_gb']:.2f}x "
                  f"more bytes -> {per_byte:.2f}x better per byte")

    with open(args.json_out, "w") as fh:
        json.dump({"full_weight_gb": round(eng.weight_bytes / 1e9, 3),
                   "hbm_gbs": args.hbm_gbs, "rows": rows}, fh, indent=2)
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
