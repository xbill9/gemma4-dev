"""Does query-chunking eliminate the 31B's S^2 prefill term?

Addendum 2/3 established the 8K ceiling is dominated by dense f32[16,2,S,S]
attention scores (8.59 GB per buffer at S=8192) plus a ~8.8 GB W4A16 working set.
`chunked_prefill_with_kv_cache` processes `chunk_size` queries at a time, which
should bound the score buffer to [16,2,chunk,S] — 16x smaller at chunk=512.

It was tried end-to-end earlier and OOMed at 8K for every chunk size (1024/512/256).
But that test conflated two things: the score buffer, and the fact that chunking
forces `window_kv=False`, which costs 7.4 GB of unwindowed KV at 8K. This separates
them by compiling the chunk STEP alone and reading its temp:

  temp collapses with chunk size -> chunking does kill the S^2 term, and the
    remaining blocker is only the unwindowed KV. That is an engineering problem
    (make chunked prefill work against a ring buffer), not a kernel rewrite.

  temp stays large -> chunking does not help and the score buffer is not what
    the chunk boundary bounds.

Compile-only, so it works at shapes that cannot execute.

    python3.13 ports/gemma4/jax_31b_chunk_temp.py
"""

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp

from ports.gemma4.jax_e_model import (
    init_kv_cache,
    make_chunked_prefill_step,
)
from ports.gemma4.jax_31b_port import Streaming31BEngine

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--total-lens", default="4096,8192,16384")
    ap.add_argument("--chunks", default="256,512,1024")
    ap.add_argument("--json-out", default="gemma4_31b_chunk_temp.json")
    args = ap.parse_args()

    eng = Streaming31BEngine(model_id=args.model, quant_mode="w4a16",
                             max_model_len=2048, window_kv=True)
    eng.load()
    limit = eng.memory_stats()["hbm_bytes_limit"]
    W = eng.weight_bytes
    print(f"\nweights {W / 1e9:.3f} GB   HBM limit {limit / 1e9:.3f} GB")
    print("\n  chunked prefill STEP, window_kv=False (required by chunking)\n")
    print(f"  {'total_len':>9s} {'chunk':>6s} {'temp':>10s} {'KV (args)':>11s} "
          f"{'W+KV+temp':>11s} {'verdict':>9s}   {'score buf if unchunked':>22s}")

    rows = []
    for total_len in [int(x) for x in args.total_lens.split(",")]:
        for chunk in [int(x) for x in args.chunks.split(",")]:
            try:
                caches = init_kv_cache(eng.config, batch_size=1,
                                       max_seq_len=total_len,
                                       dtype=jnp.bfloat16, window_kv=False)
                kv_bytes = sum(int(x.size) * int(x.dtype.itemsize)
                               for x in jax.tree_util.tree_leaves(caches))
                valid = jnp.zeros((1, total_len), dtype=jnp.bool_)
                ids = jnp.ones((1, chunk), dtype=jnp.int32)
                step = jax.jit(make_chunked_prefill_step(
                    eng.model, chunk, quant_mode="w4a16"))
                comp = step.lower(eng.params, caches, valid, ids,
                                  jnp.int32(0)).compile()
                m = comp.memory_analysis()
                temp = getattr(m, "temp_size_in_bytes", 0)
                total = W + kv_bytes + temp
                verdict = "FITS" if total < limit else "OVER"
                # what the score buffer would be without chunking, for reference
                unchunked = 16 * 2 * total_len * total_len * 4
                print(f"  {total_len:9d} {chunk:6d} {temp / 1e9:8.3f} GB "
                      f"{kv_bytes / 1e9:9.3f} GB {total / 1e9:9.3f} GB "
                      f"{verdict:>9s}   {unchunked / 1e9:19.2f} GB")
                rows.append({"total_len": total_len, "chunk": chunk,
                             "temp_gb": round(temp / 1e9, 4),
                             "kv_gb": round(kv_bytes / 1e9, 4),
                             "w_kv_temp_gb": round(total / 1e9, 4),
                             "fits": bool(total < limit),
                             "unchunked_score_buf_gb": round(unchunked / 1e9, 3),
                             "status": "OK"})
                del caches
            except Exception as exc:
                msg = str(exc).split("\n")[0][:80]
                print(f"  {total_len:9d} {chunk:6d}  FAILED: {msg}")
                rows.append({"total_len": total_len, "chunk": chunk,
                             "status": "FAILED", "error": msg})

    out = {"weights_gb": round(W / 1e9, 4),
           "hbm_limit_gb": round(limit / 1e9, 4), "rows": rows}
    with open(args.json_out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
