"""Split a bad generation into 'prefill is wrong' vs 'decode/cache is wrong'.

Runs an eager un-jitted prefill over a real prompt and prints the top-5 next
tokens as text, then runs the engine's cached generate over the same prompt. If
the eager prefill's argmax is sensible but generation is not, the defect is in
the cached decode path, not the forward math.

    python3.13 ports/gemma4/jax_12b_firsttoken.py --model <hf-id>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it-qat-w4a16-ct")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    args = ap.parse_args()

    import jax.numpy as jnp
    from transformers import AutoTokenizer
    from jax_engine import JaxGemmaEngine
    from ports.gemma4 import jax_e_model as M

    tok = AutoTokenizer.from_pretrained(args.model)
    eng = JaxGemmaEngine(model_id=args.model, quant_mode="w4a16", max_model_len=256)
    eng.load()
    cfg = eng.config

    ids = tok(args.prompt, add_special_tokens=False)["input_ids"]
    if eng.bos_token_id is not None:
        ids = [eng.bos_token_id] + ids
    elif tok.bos_token_id is not None:
        ids = [tok.bos_token_id] + ids
    n = len(ids)
    print(f"model={args.model}")
    print(f"prompt={args.prompt!r}  ids={ids}")

    arr = jnp.array(ids, dtype=jnp.int32)[None, :]
    pos = jnp.arange(n, dtype=jnp.int32)[None, :]
    valid = jnp.ones((1, n), dtype=bool)
    logits = eng.model(
        arr, eng.params, pos,
        attention_mask=M.make_prefill_causal_mask(valid),
        sliding_attention_mask=M.make_prefill_causal_mask(valid, window=cfg.sliding_window),
        quant_mode=eng.quant_mode,
    )
    if isinstance(logits, tuple):
        logits = logits[0]

    last = logits[0, -1].astype(jnp.float32)
    top = jnp.argsort(last)[::-1][:5]
    print("\n--- EAGER PREFILL, next-token top5 -------------------------")
    for t in top.tolist():
        print(f"   {t:>8}  {float(last[t]):8.3f}  {tok.decode([t])!r}")

    print("\n--- ENGINE CACHED GENERATE ---------------------------------")
    out, _ = eng.generate(ids, max_new_tokens=args.max_new_tokens, temperature=0.0)
    print(f"   ids={out}")
    print(f"   text={tok.decode(out)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
