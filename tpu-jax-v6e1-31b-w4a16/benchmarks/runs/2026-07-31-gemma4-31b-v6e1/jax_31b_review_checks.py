"""Two checks the 31B report's claims rest on but never verified.

CHECK 1 — bitwise equality of the logits_at change ON TPU, ON REAL WEIGHTS.
  tests/test_prefill_logits_slice.py proves it on CPU with a 4-layer random model
  in fp16. The shipped path is 60 layers of W4A16 on a TPU, where the matmul
  [B,1,H]x[H,V] and [B,S,H]x[H,V] may tile and accumulate differently. The report
  says "bitwise identical"; this is what would earn that word.

CHECK 2 — does the DECODE step also upcast the embedding table to f32?
  Phase 1 of the buffer dump found f32[262144,5376] (5.637 GB) in the PREFILL
  program: the tied bf16 table converted for the lm_head. The lm_head runs every
  decode step too. If the same convert is there, then the decode traffic figure
  used for the bandwidth claim (weights only, 19.3 GB) is missing ~5.6 GB/step and
  the "15% of roofline" number is wrong in the denominator.

    python3.13 ports/gemma4/jax_31b_review_checks.py
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
import numpy as np

from ports.gemma4.jax_e_model import (
    make_cached_decode_step,
    pad_to_tpu_v6e_bucket,
    prefill_with_kv_cache,
)
from ports.gemma4.jax_31b_port import Streaming31BEngine
from ports.gemma4.jax_31b_buffers_ab import (
    largest_buffers,
    old_prefill_with_kv_cache,
)

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--shapes", default="1x512,1x1024,2x1024")
    ap.add_argument("--json-out", default="gemma4_31b_review_checks.json")
    args = ap.parse_args()

    eng = Streaming31BEngine(model_id=args.model, quant_mode="w4a16",
                             max_model_len=2048, window_kv=True)
    eng.load()
    out = {"weights_gb": round(eng.weight_bytes / 1e9, 4)}

    # ------------------------------------------------ CHECK 1: bitwise on TPU
    print("\n" + "=" * 92)
    print("CHECK 1 — logits_at bitwise equality, TPU + W4A16 + real weights")
    print("=" * 92)
    rows = []
    for shape in args.shapes.split(","):
        B, S = (int(x) for x in shape.split("x"))
        rng = np.random.default_rng(0)
        ids = jnp.asarray(rng.integers(3, 200000, size=(B, S)), dtype=jnp.int32)
        padded, valid = pad_to_tpu_v6e_bucket(ids)
        kw = dict(model=eng.model, prompt_ids=padded, prompt_valid=valid,
                  params=eng.params, max_new_tokens=8, quant_mode="w4a16",
                  window_kv=True)
        try:
            new_l, _, _ = jax.block_until_ready(jax.jit(
                prefill_with_kv_cache,
                static_argnames=("model", "max_new_tokens", "quant_mode",
                                 "cache_dtype", "window_kv"))(**kw))
            old_l, _, _ = jax.block_until_ready(jax.jit(
                old_prefill_with_kv_cache,
                static_argnames=("model", "max_new_tokens", "quant_mode",
                                 "cache_dtype", "window_kv"))(**kw))
            a, b = np.asarray(new_l, np.float32), np.asarray(old_l, np.float32)
            identical = bool(np.array_equal(a, b))
            maxdiff = float(np.max(np.abs(a - b)))
            ulps = int(np.sum(a != b))
            same_argmax = bool(np.array_equal(a.argmax(-1), b.argmax(-1)))
            print(f"  B={B} S={S:5d}  bitwise={identical}  max|diff|={maxdiff:.3e}  "
                  f"differing={ulps}/{a.size}  same_argmax={same_argmax}")
            rows.append({"B": B, "S": S, "bitwise_identical": identical,
                         "max_abs_diff": maxdiff, "differing_elements": ulps,
                         "total_elements": int(a.size), "same_argmax": same_argmax})
        except Exception as exc:
            msg = str(exc).split("\n")[0][:110]
            print(f"  B={B} S={S:5d}  FAILED: {msg}")
            rows.append({"B": B, "S": S, "status": "FAILED", "error": msg})
    out["check1_bitwise"] = rows

    # ------------------------------------- CHECK 2: decode-step embedding upcast
    print("\n" + "=" * 92)
    print("CHECK 2 — does the decode step materialize f32[262144,5376]?")
    print("=" * 92)
    try:
        S = 1024
        ids = jnp.ones((1, S), dtype=jnp.int32)
        padded, valid = pad_to_tpu_v6e_bucket(ids)
        last_logits, caches, vmask = jax.block_until_ready(jax.jit(
            prefill_with_kv_cache,
            static_argnames=("model", "max_new_tokens", "quant_mode",
                             "cache_dtype", "window_kv"))(
            model=eng.model, prompt_ids=padded, prompt_valid=valid,
            params=eng.params, max_new_tokens=8, quant_mode="w4a16", window_kv=True))
        step = jax.jit(make_cached_decode_step(eng.model, quant_mode="w4a16",
                                               window_kv=True))
        tok = jnp.argmax(last_logits, axis=-1, keepdims=True)
        lens = vmask[:, :S].sum(axis=1).astype(jnp.int32)
        comp = step.lower(eng.params, caches, vmask, tok, lens,
                          jnp.int32(S)).compile()
        m = comp.memory_analysis()
        temp = getattr(m, "temp_size_in_bytes", 0)
        text = comp.as_text()
        top, allrows = largest_buffers(text, top=12)
        emb_f32 = [r for r in allrows if "262144,5376" in r["shape"] and r["shape"].startswith("f32")]
        emb_bf16 = [r for r in allrows if "262144,5376" in r["shape"] and r["shape"].startswith("bf16")]
        print(f"  decode step temp: {temp / 1e9:.3f} GB")
        print(f"  f32[262144,5376] instructions : {len(emb_f32)}  "
              f"({sum(r['bytes'] for r in emb_f32) / 1e9:.3f} GB aggregate)")
        print(f"  bf16[262144,5376] instructions: {len(emb_bf16)}")
        print("  largest buffers in the decode step:")
        for r in top:
            print(f"     {r['bytes'] / 1e9:8.3f} GB  {r['opcode']:14s} {r['shape']}")
        out["check2_decode"] = {
            "temp_gb": round(temp / 1e9, 4),
            "embed_f32_instructions": len(emb_f32),
            "embed_f32_aggregate_gb": round(sum(r["bytes"] for r in emb_f32) / 1e9, 4),
            "embed_bf16_instructions": len(emb_bf16),
            "top": [{**r, "gb": round(r["bytes"] / 1e9, 4)} for r in top],
        }
    except Exception as exc:
        msg = str(exc).split("\n")[0][:140]
        print(f"  FAILED: {msg}")
        out["check2_decode"] = {"status": "FAILED", "error": msg}

    with open(args.json_out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
