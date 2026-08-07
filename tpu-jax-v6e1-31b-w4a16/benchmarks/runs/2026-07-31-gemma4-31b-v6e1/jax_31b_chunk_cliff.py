"""Characterize the chunked-prefill compilation cliff, and test donation as a fix.

Addendum 5: at total_len=4096 the chunked prefill STEP costs 6.60 GB of temp at 30
layers and 34.70 GB at 60 — a 5.3x regression for a 2x in depth, while the one-shot
path grows 1.2x over the same step. Chunked is actually *better* than one-shot at 30
layers, so this is a scheduling regression rather than a property of chunking.

Two questions this answers:

  1. Is the cliff driven by depth alone, or by depth x total_len? Sweeping total_len
     at fixed 60 layers separates them. If 60 layers fits at a shorter total_len,
     the trigger is the combination, and long context is gated on the interaction
     rather than on depth.

  2. Does buffer donation fix it? The step takes `caches` and `valid` and returns
     updated copies. Without donation XLA must allocate a second full KV cache per
     step (3.69 GB at total_len=4096). `jax_engine` already uses donate_argnums=(1,2)
     on the decode step for exactly this reason and measured 1.62x there. The
     chunked step never got the same treatment.

Compile-only, so shapes that cannot execute still yield numbers.

    python3.13 ports/gemma4/jax_31b_chunk_cliff.py
"""

import argparse
import json
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp

from ports.gemma4.jax_e_model import init_kv_cache, make_chunked_prefill_step
from ports.gemma4.jax_31b_port import Streaming31BEngine

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"
_NEEDS_RE = re.compile(r"temporaries \(([^)]+)\)")


def measure(model, params, cfg, total_len, chunk, donate):
    caches = init_kv_cache(cfg, batch_size=1, max_seq_len=total_len,
                           dtype=jnp.bfloat16, window_kv=False)
    kv = sum(int(x.size) * int(x.dtype.itemsize)
             for x in jax.tree_util.tree_leaves(caches))
    valid = jnp.zeros((1, total_len), dtype=jnp.bool_)
    ids = jnp.ones((1, chunk), dtype=jnp.int32)
    fn = make_chunked_prefill_step(model, chunk, quant_mode="w4a16")
    step = jax.jit(fn, **({"donate_argnums": (1, 2)} if donate else {}))
    try:
        comp = step.lower(params, caches, valid, ids, jnp.int32(0)).compile()
        m = comp.memory_analysis()
        return {"kv_gb": round(kv / 1e9, 3),
                "temp_gb": round(getattr(m, "temp_size_in_bytes", 0) / 1e9, 4),
                "status": "OK"}
    except Exception as exc:
        s = str(exc)
        need = _NEEDS_RE.search(s)
        return {"kv_gb": round(kv / 1e9, 3), "status": "OOM",
                "needs": need.group(1) if need else s.split("\n")[0][:70]}
    finally:
        del caches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--total-lens", default="1024,2048,4096")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--json-out", default="gemma4_31b_chunk_cliff.json")
    args = ap.parse_args()

    eng = Streaming31BEngine(model_id=args.model, quant_mode="w4a16",
                             max_model_len=2048, window_kv=True)
    eng.load()
    print(f"\nfull 60-layer model, chunk={args.chunk}, window_kv=False\n")
    print(f"  {'total_len':>9s} {'donate':>7s} {'KV':>9s} {'step temp':>12s}")

    rows = []
    for T in [int(x) for x in args.total_lens.split(",")]:
        for donate in (False, True):
            r = measure(eng.model, eng.params, eng.config, T, args.chunk, donate)
            shown = (f"{r['temp_gb']:8.3f} GB" if r["status"] == "OK"
                     else f"OOM {r['needs']}")
            print(f"  {T:9d} {str(donate):>7s} {r['kv_gb']:7.2f} GB {shown:>12s}")
            rows.append({"total_len": T, "donate": donate, "chunk": args.chunk, **r})

    print()
    for T in sorted({r["total_len"] for r in rows}):
        off = next((r for r in rows if r["total_len"] == T and not r["donate"]), None)
        on = next((r for r in rows if r["total_len"] == T and r["donate"]), None)
        if off and on and off["status"] == "OK" and on["status"] == "OK":
            print(f"  total_len={T}: donation {off['temp_gb'] - on['temp_gb']:+.3f} GB "
                  f"({off['temp_gb']:.3f} -> {on['temp_gb']:.3f})")
        elif off and on and off["status"] == "OOM" and on["status"] == "OK":
            print(f"  total_len={T}: donation FIXES IT "
                  f"(OOM -> {on['temp_gb']:.3f} GB)")
        elif off and on:
            print(f"  total_len={T}: donation does not fix it "
                  f"({off['status']} -> {on['status']})")

    with open(args.json_out, "w") as fh:
        json.dump({"weights_gb": round(eng.weight_bytes / 1e9, 3), "rows": rows},
                  fh, indent=2)
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
