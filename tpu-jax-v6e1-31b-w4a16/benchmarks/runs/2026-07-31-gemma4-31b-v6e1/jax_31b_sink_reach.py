"""Who can still see the attention sink at long context?

MODEL-INTEL.md §7: the massive activations sit on token positions 0-1 (`<bos>`,
`<|turn>`) in 78% of layers — the attention-sink signature. §4: the 10
full-attention layers produce 33-37% larger outputs than the 50 sliding ones and
are damped harder, i.e. they are load-bearing.

Those two facts predict a connection. A sliding layer has `sliding_window=1024`
and is masked to `(p - 1024, p]` REGARDLESS of `window_kv` — the ring buffer is a
memory optimization, the mask is the semantics. So once the prompt exceeds 1024
tokens, **50 of the 60 layers structurally cannot attend to position 0 at all.**
Only the 10 full-attention layers can.

If the sink matters, full-attention layers should put disproportionate attention
mass on positions 0-1 at long context, and that would explain why they are the
load-bearing ones.

This measures it directly: for the LAST query position, the attention distribution
over all keys, per layer, at a context longer than the window.

    python3.13 ports/gemma4/jax_31b_sink_reach.py --tokens 1536
"""

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", type=int, default=1536,
                    help="prompt length; must exceed sliding_window to be meaningful")
    ap.add_argument("--sink", type=int, default=2, help="how many leading positions count as the sink")
    ap.add_argument("--json-out", default="gemma4_31b_sink_reach.json")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer
    from ports.gemma4.jax_31b_port import Streaming31BEngine
    from ports.gemma4 import jax_e_model as M

    eng = Streaming31BEngine(model_id=args.model, quant_mode="w4a16",
                             max_model_len=args.tokens + 64, window_kv=True)
    eng.load()
    cfg = eng.config
    W = cfg.sliding_window
    print(f"\nsliding_window={W}, prompt={args.tokens} tokens "
          f"({'window binds' if args.tokens > W else 'WINDOW DOES NOT BIND — raise --tokens'})")

    tok = AutoTokenizer.from_pretrained(args.model)
    base = tok.apply_chat_template(
        [{"role": "user", "content": "Summarize the history of computing."}],
        tokenize=False, add_generation_prompt=True)
    ids_l = tok(base, add_special_tokens=False)["input_ids"]
    body = tok(" The quick brown fox jumps over the lazy dog.",
               add_special_tokens=False)["input_ids"]
    while len(ids_l) < args.tokens:
        ids_l = ids_l + body
    ids_l = ids_l[:args.tokens]
    ids = jnp.asarray(ids_l, dtype=jnp.int32)[None, :]
    S = ids.shape[1]

    # Capture the attention distribution of the LAST query position, per layer.
    rec = []
    orig = M.eager_attention_jax

    def patched(query, key, value, mask=None, scaling=1.0, softcap=30.0,
                key_scale=None, value_scale=None):
        # Recompute just the last query row's probabilities; cheap next to the
        # full attention, and avoids holding [B,H,S,S].
        q = jnp.asarray(query, jnp.float32)[:, :, -1:, :]          # [B,H,1,D]
        k = jnp.asarray(key, jnp.float32)
        if k.shape[1] != q.shape[1]:                                # GQA broadcast
            rep = q.shape[1] // k.shape[1]
            k = jnp.repeat(k, rep, axis=1)
        s = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scaling
        if softcap:
            s = jnp.tanh(s / softcap) * softcap
        if mask is not None:
            m = jnp.asarray(mask, jnp.float32)
            s = s + (m[:, :, -1:, :] if m.shape[2] > 1 else m)
        p = jax.nn.softmax(s, axis=-1)[0, :, 0, :]                  # [H, S_kv]
        rec.append(np.asarray(p))
        return orig(query, key, value, mask=mask, scaling=scaling, softcap=softcap,
                    key_scale=key_scale, value_scale=value_scale)

    M.eager_attention_jax = patched
    try:
        valid = jnp.ones((1, S), dtype=bool)
        pos = jnp.arange(S, dtype=jnp.int32)[None, :]
        M.Gemma4EModelJAX.__call__(
            eng.model, ids, eng.params, pos,
            attention_mask=M.make_prefill_causal_mask(valid), quant_mode="w4a16",
            sliding_attention_mask=M.make_prefill_causal_mask(valid, window=W))
    finally:
        M.eager_attention_jax = orig

    print(f"  captured {len(rec)} layers\n")
    print(f"  {'layer':>5} {'type':>7} {'sink mass':>11} {'reachable?':>11} "
          f"{'top key':>8} {'top mass':>9}")
    rows = []
    for i, p in enumerate(rec):
        lt = cfg.layer_types[i]
        mean_over_heads = p.mean(axis=0)                             # [S_kv]
        sink = float(mean_over_heads[:args.sink].sum())
        top = int(mean_over_heads.argmax())
        reach = (lt == "full_attention") or (S - 1 - args.sink < W)
        rows.append({"layer": i, "type": lt, "sink_mass": sink,
                     "top_key": top, "top_mass": float(mean_over_heads[top]),
                     "sink_reachable": bool(reach)})
        if i < 3 or lt == "full_attention" or i >= len(rec) - 2:
            print(f"  {i:>5} {lt[:4]:>7} {sink:11.6f} {str(reach):>11} "
                  f"{top:>8} {rows[-1]['top_mass']:9.4f}")

    sl = [r for r in rows if r["type"] == "sliding_attention"]
    fu = [r for r in rows if r["type"] == "full_attention"]
    sm_s = float(np.mean([r["sink_mass"] for r in sl]))
    sm_f = float(np.mean([r["sink_mass"] for r in fu]))
    print(f"\n  mean attention mass on positions 0..{args.sink-1}, last query:")
    print(f"    sliding (n={len(sl)}): {sm_s:.6f}")
    print(f"    full    (n={len(fu)}): {sm_f:.6f}"
          + (f"   ({sm_f/sm_s:.0f}x more)" if sm_s > 0 else "   (sliding is exactly 0)"))
    frac_full_sink = float(np.mean([r["top_key"] < args.sink for r in fu]))
    print(f"    full-attention layers whose TOP key is the sink: "
          f"{frac_full_sink*100:.0f}%")

    out = {"model": args.model, "tokens": S, "sliding_window": W,
           "sink_positions": args.sink, "layers": rows,
           "sliding_sink_mass": sm_s, "full_sink_mass": sm_f,
           "full_top_is_sink_frac": frac_full_sink}
    with open(args.json_out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
