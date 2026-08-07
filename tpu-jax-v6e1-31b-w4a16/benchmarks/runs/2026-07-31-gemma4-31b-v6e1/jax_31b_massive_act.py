"""Find the 31B's massive-activation channels and test whether they are an attention sink.

MODEL-INTEL.md found `input_layernorm` weights averaging 23.5 with a single channel
at 1248, and `pre_feedforward_layernorm` at 40.8 / 860. In most large transformers
that signature means a handful of FIXED channels carry enormous activations, usually
concentrated on one or two token positions (typically BOS or a delimiter), and those
positions act as attention sinks. If that is what this is, it has direct consequences:

  * per-tensor activation quantization will be destroyed by those channels
  * KV-cache quantization must not clip the sink position
  * dropping/trimming the first token is dangerous in a way it looks like it isn't

STATIC  — which channel indices dominate each norm, and are they the SAME channels
          across all 60 layers? A shared index set is the signature; scattered
          indices would mean this is just heavy-tailed weight noise.

DYNAMIC — where the magnitude actually lands at run time: for each layer, the
          largest |activation| in the residual stream, which channel it sits in,
          and which TOKEN POSITION. Position concentration is the attention-sink
          test.

    python3.13 ports/gemma4/jax_31b_massive_act.py
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from ports.gemma4.jax_31b_model_intel import SafeReader, _snapshot

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"
NORMS = ["input_layernorm", "pre_feedforward_layernorm",
         "post_attention_layernorm", "post_feedforward_layernorm"]


def pass_static(model_id: str, topk: int, out: dict) -> None:
    path = _snapshot(model_id)
    shard = sorted(glob.glob(os.path.join(path, "*.safetensors")))[0]
    cfg = json.load(open(os.path.join(path, "config.json")))
    tcfg = cfg.get("text_config", cfg)
    n, H = tcfg["num_hidden_layers"], tcfg["hidden_size"]
    f = SafeReader(shard)
    keys = set(f.keys())
    pre = "model.language_model."

    print(f"\n=== STATIC: which channels dominate the norms? (H={H}) ===")
    out["static"] = {}
    for nm in NORMS:
        per_layer_top = {}
        allw = []
        for i in range(n):
            k = f"{pre}layers.{i}.{nm}.weight"
            if k not in keys:
                continue
            w = np.abs(f.get(k).ravel())
            allw.append(w)
            per_layer_top[i] = np.argsort(w)[::-1][:topk].tolist()
        if not per_layer_top:
            continue
        W = np.stack(allw)                                   # [L, H]
        med = np.median(W)
        # how concentrated is the mass?
        frac_10x = float((W > 10 * med).mean())
        # do the same channel indices recur across layers?
        cnt = Counter()
        for idx in per_layer_top.values():
            cnt.update(idx)
        shared = cnt.most_common(8)
        n_layers = len(per_layer_top)
        top1 = Counter(v[0] for v in per_layer_top.values()).most_common(3)
        print(f"\n{nm}")
        print(f"  median |w| {med:.3f}   channels >10x median: {100*frac_10x:.3f}%")
        print(f"  most recurrent top-{topk} channels (of {n_layers} layers): "
              + ", ".join(f"ch{c}x{k}" for c, k in shared[:6]))
        print(f"  most common layer-argmax channel: "
              + ", ".join(f"ch{c} in {k}/{n_layers} layers" for c, k in top1))
        out["static"][nm] = {
            "median_abs": float(med), "frac_gt_10x_median": frac_10x,
            "recurrent_top_channels": [[int(c), int(k)] for c, k in shared],
            "argmax_channel_counts": [[int(c), int(k)] for c, k in top1],
        }


def pass_dynamic(model_id: str, prompt: str, out: dict) -> None:
    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer
    from ports.gemma4.jax_31b_port import Streaming31BEngine
    from ports.gemma4 import jax_e_model as M

    print("\n=== DYNAMIC: where does the magnitude actually land? ===")
    eng = Streaming31BEngine(model_id=model_id, quant_mode="w4a16",
                             max_model_len=512, window_kv=True)
    eng.load()
    cfg = eng.config
    tok = AutoTokenizer.from_pretrained(model_id)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ids_np = tok(text, add_special_tokens=False, return_tensors="np")["input_ids"]
    ids = jnp.asarray(ids_np, dtype=jnp.int32)
    S = ids.shape[1]
    toks = [tok.decode([int(t)]) for t in ids_np[0]]

    rec = []
    orig = M.Gemma4EAttentionJAX.__call__

    def patch(self, hidden_states, *a, **k):
        h = np.asarray(jnp.asarray(hidden_states, jnp.float32))    # [B,S,H]
        absh = np.abs(h[0])
        flat = int(absh.argmax())
        pos, ch = divmod(flat, absh.shape[1])
        rec.append({"type": self.layer_type, "max": float(absh.max()),
                    "pos": int(pos), "ch": int(ch),
                    "median": float(np.median(absh)),
                    "pos0_max": float(absh[0].max()),
                    "rms_by_pos": absh.mean(axis=1).tolist()})
        return orig(self, hidden_states, *a, **k)

    M.Gemma4EAttentionJAX.__call__ = patch
    try:
        valid = jnp.ones((1, S), dtype=bool)
        pos_ids = jnp.arange(S, dtype=jnp.int32)[None, :]
        M.Gemma4EModelJAX.__call__(
            eng.model, ids, eng.params, pos_ids,
            attention_mask=M.make_prefill_causal_mask(valid), quant_mode="w4a16",
            sliding_attention_mask=M.make_prefill_causal_mask(
                valid, window=cfg.sliding_window))
    finally:
        M.Gemma4EAttentionJAX.__call__ = orig

    print(f"\n  prompt ({S} tokens): {toks}\n")
    print(f"  {'layer':>5} {'type':>7} {'max|h|':>10} {'ratio_to_med':>13} "
          f"{'at pos':>7} {'token':>14} {'ch':>6}")
    for i, r in enumerate(rec):
        if i < 4 or i % 10 == 0 or i >= len(rec) - 2:
            ratio = r["max"] / (r["median"] or 1e-9)
            t = toks[r["pos"]] if r["pos"] < len(toks) else "?"
            print(f"  {i:>5} {r['type'][:4]:>7} {r['max']:10.4g} {ratio:13.1f} "
                  f"{r['pos']:>7} {t!r:>14} {r['ch']:>6}")

    poscnt = Counter(r["pos"] for r in rec)
    chcnt = Counter(r["ch"] for r in rec)
    print(f"\n  token position holding the max, over {len(rec)} layers: "
          + ", ".join(f"pos{p} ({toks[p]!r}) x{c}" for p, c in poscnt.most_common(4)))
    print(f"  channel holding the max: "
          + ", ".join(f"ch{c} x{k}" for c, k in chcnt.most_common(4)))
    ratios = [r["max"] / (r["median"] or 1e-9) for r in rec]
    print(f"  max/median ratio: mean {np.mean(ratios):.1f}, peak {np.max(ratios):.1f} "
          f"at layer {int(np.argmax(ratios))}")

    # is position 0 special? compare its mean |h| against every other position
    per_pos = np.array([r["rms_by_pos"] for r in rec])          # [L, S]
    p0 = per_pos[:, 0].mean()
    rest = per_pos[:, 1:].mean()
    print(f"  mean |h| at position 0: {p0:.4g}   at positions 1..{S-1}: {rest:.4g} "
          f"  ({p0/rest:.1f}x)")

    out["dynamic"] = {
        "tokens": toks, "layers": rec,
        "pos_counts": [[int(p), int(c)] for p, c in poscnt.most_common()],
        "ch_counts": [[int(c), int(k)] for c, k in chcnt.most_common(10)],
        "pos0_vs_rest": {"pos0": float(p0), "rest": float(rest),
                         "ratio": float(p0 / (rest or 1e-9))},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--passes", default="static,dynamic")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--json-out", default="gemma4_31b_massive_act.json")
    args = ap.parse_args()

    out = {"model": args.model}
    p = [x.strip() for x in args.passes.split(",")]
    if "static" in p:
        pass_static(args.model, args.topk, out)
    if "dynamic" in p:
        pass_dynamic(args.model, args.prompt, out)
    with open(args.json_out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
