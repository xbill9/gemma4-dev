"""Ask XLA what the 8K prefill actually allocates, instead of inferring it.

The first pass at the 31B's prefill ceiling reasoned from resident-HBM growth to a
candidate buffer, predicted that removing it would clear the 8K OOM, and was wrong
— the removal was real (4.29 GB at B=2 x 2K, which became feasible) but not the
binding term. `lowered.compile().memory_analysis()` reports the compiled peak and
its breakdown directly, which is the measurement that should have come first.

Compiles prefill WITHOUT running it, so it works at shapes that OOM at execution.

    python3.13 ports/gemma4/jax_31b_hlo_memory.py --contexts 4096,8192
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

from ports.gemma4.jax_e_model import pad_to_tpu_v6e_bucket, prefill_with_kv_cache
from ports.gemma4.jax_31b_port import Streaming31BEngine

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--contexts", default="1024,2048,4096,8192")
    ap.add_argument("--batch-sizes", default="1,2")
    ap.add_argument("--json-out", default="gemma4_31b_hlo_memory.json")
    args = ap.parse_args()

    eng = Streaming31BEngine(model_id=args.model, quant_mode="w4a16",
                             max_model_len=4096, window_kv=True)
    eng.load()
    limit = eng.memory_stats()["hbm_bytes_limit"]
    print(f"\nweights {eng.weight_bytes / 1e9:.3f} GB   HBM limit {limit / 1e9:.3f} GB\n")

    jit_prefill = jax.jit(
        prefill_with_kv_cache,
        static_argnames=("model", "max_new_tokens", "quant_mode", "cache_dtype", "window_kv"),
    )

    rows = []
    for B in [int(x) for x in args.batch_sizes.split(",")]:
        for S in [int(x) for x in args.contexts.split(",")]:
            ids = jnp.ones((B, S), dtype=jnp.int32)
            padded, valid = pad_to_tpu_v6e_bucket(ids)
            try:
                comp = jit_prefill.lower(
                    model=eng.model, prompt_ids=padded, prompt_valid=valid,
                    params=eng.params, max_new_tokens=8, quant_mode="w4a16",
                    window_kv=True,
                ).compile()
                m = comp.memory_analysis()
                temp = getattr(m, "temp_size_in_bytes", 0)
                argsz = getattr(m, "argument_size_in_bytes", 0)
                out = getattr(m, "output_size_in_bytes", 0)
                alias = getattr(m, "alias_size_in_bytes", 0)
                total = temp + argsz + out - alias
                per_tok = temp / (B * S) / 1e6
                print(f"  B={B} S={S:6d}  temp {temp / 1e9:7.3f} GB  args {argsz / 1e9:7.3f} GB  "
                      f"out {out / 1e9:6.3f} GB  -> peak ~{total / 1e9:7.3f} GB "
                      f"({'FITS' if total < limit else 'OVER'})  temp/token {per_tok:.3f} MB")
                rows.append({"B": B, "S": S,
                             "temp_gb": round(temp / 1e9, 4),
                             "argument_gb": round(argsz / 1e9, 4),
                             "output_gb": round(out / 1e9, 4),
                             "alias_gb": round(alias / 1e9, 4),
                             "peak_gb": round(total / 1e9, 4),
                             "temp_mb_per_token": round(per_tok, 4),
                             "fits": bool(total < limit), "status": "OK"})
            except Exception as exc:
                msg = str(exc).split("\n")[0][:90]
                print(f"  B={B} S={S:6d}  COMPILE FAILED: {msg}")
                rows.append({"B": B, "S": S, "status": "FAILED", "error": msg})

    with open(args.json_out, "w") as fh:
        json.dump({"weights_gb": round(eng.weight_bytes / 1e9, 4),
                   "hbm_limit_gb": round(limit / 1e9, 4), "rows": rows}, fh, indent=2)
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
