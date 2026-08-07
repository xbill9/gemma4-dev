"""Locate what actually sets the 31B's prefill ceiling on a v6e-1.

The sweep OOMs above ~4K prompt tokens. Two candidates: attention temporaries, or
the `[B, S, vocab]` logits tensor that `prefill_with_kv_cache` materializes for the
whole sequence and then throws away all but one row of (jax_e_model.py:1306-1317).
With vocab=262144 the second is 1.07 MB *per token*, which would dominate.

This measures peak HBM after a prefill at several S and fits the growth. If the
slope is ~1.07 MB/token (f32) or ~0.54 (bf16) beyond weights and KV, the logits
tensor is the ceiling and slicing the hidden state before the lm_head removes it.

    python3.13 ports/gemma4/jax_31b_prefill_probe.py --contexts 512,1024,2048,4096
"""

import argparse
import gc
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
    ap.add_argument("--contexts", default="512,1024,2048,4096")
    ap.add_argument("--json-out", default="gemma4_31b_prefill_probe.json")
    args = ap.parse_args()

    contexts = [int(x) for x in args.contexts.split(",")]
    dev = jax.devices()[0]

    eng = Streaming31BEngine(model_id=args.model, quant_mode="w4a16",
                             max_model_len=max(contexts) + 64, window_kv=True)
    eng.load()
    base = dev.memory_stats()["bytes_in_use"]
    print(f"\nweights resident: {base / 1e9:.3f} GB")
    print(f"vocab={eng.config.vocab_size}  -> logits cost per token: "
          f"{eng.config.vocab_size * 4 / 1e6:.2f} MB (f32) / "
          f"{eng.config.vocab_size * 2 / 1e6:.2f} MB (bf16)\n")

    jit_prefill = jax.jit(
        prefill_with_kv_cache,
        static_argnames=("model", "max_new_tokens", "quant_mode", "cache_dtype", "window_kv"),
    )

    rows = []
    prev = None
    for S in contexts:
        ids = jnp.ones((1, S), dtype=jnp.int32)
        padded, valid = pad_to_tpu_v6e_bucket(ids)
        try:
            out = jax.block_until_ready(jit_prefill(
                model=eng.model, prompt_ids=padded, prompt_valid=valid,
                params=eng.params, max_new_tokens=8, quant_mode="w4a16",
                window_kv=True,
            ))
            peak = dev.memory_stats().get("peak_bytes_in_use", 0)
            live = dev.memory_stats()["bytes_in_use"]
            delta_per_tok = None
            if prev is not None:
                dS = S - prev[0]
                delta_per_tok = (peak - prev[1]) / dS / 1e6
            print(f"  S={S:6d}  peak {peak / 1e9:7.3f} GB  live {live / 1e9:7.3f} GB  "
                  f"above-weights {(peak - base) / 1e9:7.3f} GB"
                  + (f"  marginal {delta_per_tok:6.3f} MB/token" if delta_per_tok else ""))
            rows.append({"S": S, "peak_gb": round(peak / 1e9, 4),
                         "live_gb": round(live / 1e9, 4),
                         "above_weights_gb": round((peak - base) / 1e9, 4),
                         "marginal_mb_per_token": (round(delta_per_tok, 4)
                                                   if delta_per_tok else None),
                         "status": "OK"})
            prev = (S, peak)
            del out
        except Exception as exc:
            msg = str(exc).split("\n")[0][:80]
            print(f"  S={S:6d}  FAILED: {msg}")
            rows.append({"S": S, "status": "FAILED", "error": msg})
        gc.collect()

    with open(args.json_out, "w") as fh:
        json.dump({"weights_gb": round(base / 1e9, 4),
                   "vocab_size": eng.config.vocab_size, "rows": rows}, fh, indent=2)
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
