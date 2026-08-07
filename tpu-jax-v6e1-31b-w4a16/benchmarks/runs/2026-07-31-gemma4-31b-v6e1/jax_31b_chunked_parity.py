"""End-to-end validation of the donated chunked prefill path on real 31B weights.

The `donate_argnums=(1, 2)` fix was measured at COMPILE time on a hand-built step
(34.70 GB -> 8.013 GB of temp at total_len=4096). It was never run end to end. That
is the part that can actually be wrong: donated buffers are INVALIDATED by the call
that consumes them, so any lingering reference to `caches` or `valid` across chunk
iterations raises at runtime rather than at compile time — and a silent
mis-ordering would corrupt the cache instead.

So this runs the real `chunked_prefill_with_kv_cache` on the real checkpoint and
compares its `last_logits` against the one-shot `prefill_with_kv_cache` on identical
input. They should agree closely; they will NOT agree bitwise, for the same reason
the logits_at change does not (different contraction shapes tile differently on the
MXU), so the check is on argmax and max-abs-diff rather than equality.

    python3.13 ports/gemma4/jax_31b_chunked_parity.py --chunks 256,512,1024
"""

import argparse
import json
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np

from ports.gemma4.jax_e_model import (
    chunked_prefill_with_kv_cache,
    pad_to_tpu_v6e_bucket,
    prefill_with_kv_cache,
)
from ports.gemma4.jax_31b_port import Streaming31BEngine

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--S", type=int, default=2048)
    ap.add_argument("--chunks", default="256,512,1024")
    ap.add_argument("--json-out", default="gemma4_31b_chunked_parity.json")
    args = ap.parse_args()

    eng = Streaming31BEngine(model_id=args.model, quant_mode="w4a16",
                             max_model_len=args.S + 64, window_kv=True)
    eng.load()
    dev = jax.devices()[0]
    print(f"\nweights {eng.weight_bytes / 1e9:.3f} GB, S={args.S}\n")

    rng = np.random.default_rng(0)
    ids = jnp.asarray(rng.integers(3, 200000, size=(1, args.S)), dtype=jnp.int32)
    padded, valid = pad_to_tpu_v6e_bucket(ids)

    # Reference: one-shot prefill, windowed KV (the shipped default path).
    t0 = time.perf_counter()
    ref, _, _ = jax.block_until_ready(jax.jit(
        prefill_with_kv_cache,
        static_argnames=("model", "max_new_tokens", "quant_mode",
                         "cache_dtype", "window_kv"))(
        model=eng.model, prompt_ids=padded, prompt_valid=valid, params=eng.params,
        max_new_tokens=8, quant_mode="w4a16", window_kv=True))
    ref_np = np.asarray(ref, np.float32)
    print(f"  one-shot reference: {time.perf_counter() - t0:.1f}s, "
          f"top-1 token {int(ref_np.argmax(-1)[0])}")

    rows = []
    for chunk in [int(x) for x in args.chunks.split(",")]:
        if args.S % chunk:
            print(f"  chunk={chunk}: skipped (S not a multiple)")
            continue
        try:
            t0 = time.perf_counter()
            got, caches, vmask = jax.block_until_ready(
                chunked_prefill_with_kv_cache(
                    eng.model, padded, valid, eng.params, max_new_tokens=8,
                    chunk_size=chunk, quant_mode="w4a16",
                    cache_dtype=jnp.bfloat16))
            wall = time.perf_counter() - t0
            g = np.asarray(got, np.float32)
            same_argmax = bool(np.array_equal(g.argmax(-1), ref_np.argmax(-1)))
            maxdiff = float(np.max(np.abs(g - ref_np)))
            rel = maxdiff / (float(np.max(np.abs(ref_np))) or 1.0)
            hbm = dev.memory_stats().get("bytes_in_use", 0)
            print(f"  chunk={chunk:5d}  RAN ok  {wall:6.1f}s  "
                  f"argmax_match={same_argmax}  max|diff|={maxdiff:.4f} "
                  f"(rel {rel:.2e})  HBM {hbm / 1e9:.2f} GB")
            rows.append({"chunk": chunk, "status": "OK", "wall_s": round(wall, 1),
                         "argmax_match": same_argmax,
                         "max_abs_diff": maxdiff, "rel_diff": rel,
                         "hbm_gb": round(hbm / 1e9, 3),
                         "top1": int(g.argmax(-1)[0]),
                         "ref_top1": int(ref_np.argmax(-1)[0])})
            del got, caches, vmask
        except Exception as exc:
            msg = str(exc).split("\n")[0][:120]
            print(f"  chunk={chunk:5d}  FAILED: {msg}")
            rows.append({"chunk": chunk, "status": "FAILED", "error": msg})

    ok = [r for r in rows if r.get("status") == "OK"]
    print(f"\n  {len(ok)}/{len(rows)} chunk sizes ran; "
          f"argmax matched in {sum(1 for r in ok if r['argmax_match'])}/{len(ok)}")

    with open(args.json_out, "w") as fh:
        json.dump({"S": args.S, "weights_gb": round(eng.weight_bytes / 1e9, 3),
                   "ref_top1": int(ref_np.argmax(-1)[0]), "rows": rows}, fh, indent=2)
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
