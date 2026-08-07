"""Localize where a Gemma 4 forward pass goes wrong, layer by layer.

Runs ONE eager (un-jitted) prefill and records the RMS of the residual stream
after each sub-block, plus the RMS of every attention and MLP output. A healthy
stream drifts smoothly; a broken layer type shows up as a step change at the
layers of that type.

Compare a suspect checkpoint against a known-good one:

    python3.13 ports/gemma4/jax_12b_layer_probe.py --model google/gemma-4-12B-it-qat-w4a16-ct
    python3.13 ports/gemma4/jax_12b_layer_probe.py --model google/gemma-4-E2B-it-qat-w4a16-ct
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it-qat-w4a16-ct")
    ap.add_argument("--tokens", type=int, default=16)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from jax_engine import JaxGemmaEngine
    from ports.gemma4 import jax_e_model as M

    eng = JaxGemmaEngine(model_id=args.model, quant_mode="w4a16", max_model_len=256)
    eng.load()
    cfg = eng.config
    print(f"model={args.model}  layers={cfg.num_hidden_layers}  "
          f"k_eq_v={cfg.attention_k_eq_v}  n_global_kv={cfg.num_global_key_value_heads}")

    # Record sub-block output magnitudes by wrapping the attention / MLP callables.
    records = []

    def rms(x):
        return float(jnp.sqrt(jnp.mean(jnp.square(x.astype(jnp.float32)))))

    orig_attn = M.Gemma4EAttentionJAX.__call__
    orig_mlp = M.Gemma4EMLPJAX.__call__

    def attn_wrap(self, hidden_states, *a, **kw):
        out = orig_attn(self, hidden_states, *a, **kw)
        records.append(("attn", self.layer_type, rms(hidden_states), rms(out[0])))
        return out

    def mlp_wrap(self, x, *a, **kw):
        out = orig_mlp(self, x, *a, **kw)
        records.append(("mlp", "-", rms(x), rms(out)))
        return out

    M.Gemma4EAttentionJAX.__call__ = attn_wrap
    M.Gemma4EMLPJAX.__call__ = mlp_wrap

    ids = jnp.arange(2, 2 + args.tokens, dtype=jnp.int32)[None, :]
    pos = jnp.arange(args.tokens, dtype=jnp.int32)[None, :]
    valid = jnp.ones((1, args.tokens), dtype=bool)
    mask = M.make_prefill_causal_mask(valid)
    smask = M.make_prefill_causal_mask(valid, window=cfg.sliding_window)

    logits = eng.model(
        ids, eng.params, pos,
        attention_mask=mask, sliding_attention_mask=smask,
        quant_mode=eng.quant_mode,
    )
    if isinstance(logits, tuple):
        logits = logits[0]

    M.Gemma4EAttentionJAX.__call__ = orig_attn
    M.Gemma4EMLPJAX.__call__ = orig_mlp

    print(f"\n{'layer':>5} {'type':>18} {'attn_in':>12} {'attn_out':>12} "
          f"{'mlp_in':>12} {'mlp_out':>12}")
    per_layer = [records[i:i + 2] for i in range(0, len(records), 2)]
    for i, pair in enumerate(per_layer):
        (_, ltype, ain, aout) = pair[0]
        (_, _, min_, mout) = pair[1] if len(pair) > 1 else ("mlp", "-", 0.0, 0.0)
        flag = ""
        if aout > 1e4 or aout < 1e-4:
            flag = "  <== attn out degenerate"
        print(f"{i:>5} {ltype:>18} {ain:>12.4g} {aout:>12.4g} "
              f"{min_:>12.4g} {mout:>12.4g}{flag}")

    lg = logits.astype(jnp.float32)
    print(f"\nlogits: rms={rms(lg):.4g} min={float(lg.min()):.4g} "
          f"max={float(lg.max()):.4g}")
    top = jnp.argsort(lg[0, -1])[::-1][:5]
    print(f"top5 last-position token ids: {top.tolist()}")
    print(f"top5 logit values: {[round(float(lg[0, -1, t]), 3) for t in top]}")
    frac_capped = float(jnp.mean(jnp.abs(lg) > 0.99 * cfg.logit_softcapping))
    print(f"fraction of logits at softcap ({cfg.logit_softcapping}): {frac_capped:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
