"""Characterize gemma-4-31B-it-qat-w4a16-ct itself: what the weights and the
activations say about how this model is built and where it is fragile.

Everything else in this directory measures the *engine*. This measures the
*model*. Three passes:

  STATIC   — read the checkpoint. Per-layer residual damping (`layer_scalar`),
             RMSNorm weight magnitudes, and the W4A16 group-scale distribution
             for every projection. The scale dynamic range (p99.9 / median) is
             the useful one: it says which projections have outlier channels and
             are therefore hardest to quantize, which is where a W4A16 checkpoint
             loses the most accuracy.

  DYNAMIC  — one eager prefill on a real prompt, recording the residual RMS
             around every attention and MLP block. Answers whether the 10
             full-attention layers behave like a different population from the
             50 sliding ones, and whether the stream grows or is held flat by
             the layer_scalar schedule.

  OUTPUT   — logit distribution at the final position: softcap saturation, top-k
             mass, entropy. Says how confident this model actually is, which is
             what makes greedy decoding stable or not.

    python3.13 ports/gemma4/jax_31b_model_intel.py --passes static,dynamic
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
NORMS = ["input_layernorm", "post_attention_layernorm",
         "pre_feedforward_layernorm", "post_feedforward_layernorm"]


def _snapshot(model_id: str) -> str:
    pat = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots/*")
    hits = glob.glob(pat)
    if not hits:
        raise FileNotFoundError(f"no local snapshot for {model_id}")
    return hits[0]


class SafeReader:
    """Minimal safetensors reader that returns float32 numpy for any dtype.

    Needed because this safetensors build cannot decode bfloat16 through numpy
    OR flax — both raise `data type 'bfloat16' not understood`, and torch is not
    installed on a bare JAX VM. The container format is trivial: an 8-byte
    little-endian header length, a JSON header of {name: {dtype, shape,
    data_offsets}}, then the raw buffer. bf16 -> f32 is a 16-bit left shift.
    """

    _NP = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "I64": "<i8", "I32": "<i4",
           "I16": "<i2", "I8": "|i1", "U8": "|u1", "BOOL": "|b1"}

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            self.header = json.loads(fh.read(n).decode("utf-8"))
        self.data_start = 8 + n
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")

    def keys(self):
        return [k for k in self.header if k != "__metadata__"]

    def get(self, name: str) -> np.ndarray:
        meta = self.header[name]
        lo, hi = meta["data_offsets"]
        raw = self._mm[self.data_start + lo: self.data_start + hi]
        dt = meta["dtype"]
        if dt == "BF16":
            u16 = raw.view("<u2")
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        elif dt in self._NP:
            f32 = raw.view(self._NP[dt]).astype(np.float32)
        else:
            raise ValueError(f"unhandled safetensors dtype {dt} for {name}")
        return f32.reshape(meta["shape"])


def _f32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def _stats(a) -> dict:
    f = _f32(a).ravel()
    f = f[np.isfinite(f)]
    if f.size == 0:
        return {}
    absf = np.abs(f)
    med = float(np.median(absf)) or 1e-12
    return {"mean": float(f.mean()), "absmean": float(absf.mean()),
            "median_abs": med, "max_abs": float(absf.max()),
            "p99_9_abs": float(np.percentile(absf, 99.9)),
            "dynrange": float(np.percentile(absf, 99.9) / med),
            "frac_zero": float((f == 0).mean())}


def pass_static(model_id: str, out: dict) -> None:
    path = _snapshot(model_id)
    shard = sorted(glob.glob(os.path.join(path, "*.safetensors")))[0]
    print(f"\n=== STATIC: {os.path.basename(shard)} ===\n")

    import json as _json
    cfg = _json.load(open(os.path.join(path, "config.json")))
    tcfg = cfg.get("text_config", cfg)
    layer_types = tcfg["layer_types"]
    n = tcfg["num_hidden_layers"]

    scalars, norms, scales = {}, defaultdict(dict), defaultdict(dict)
    f = SafeReader(shard)
    if True:
        keys = set(f.keys())
        pre = "model.language_model."
        for i in range(n):
            lp = f"{pre}layers.{i}"
            k = f"{lp}.layer_scalar"
            if k in keys:
                scalars[i] = float(_f32(f.get(k)).ravel()[0])
            for nm in NORMS:
                kk = f"{lp}.{nm}.weight"
                if kk in keys:
                    norms[nm][i] = _stats(f.get(kk))
            for p in PROJ:
                kk = f"{lp}.self_attn.{p}.weight_scale" if "proj" in p and p in (
                    "q_proj", "k_proj", "v_proj", "o_proj") else f"{lp}.mlp.{p}.weight_scale"
                if kk in keys:
                    scales[p][i] = _stats(f.get(kk))

    # --- residual damping schedule
    if scalars:
        vals = [scalars[i] for i in sorted(scalars)]
        print("layer_scalar (residual damping applied to the whole stream each layer)")
        print(f"  n={len(vals)}  mean={np.mean(vals):.4f}  min={min(vals):.4f} "
              f"(layer {int(np.argmin(vals))})  max={max(vals):.4f} (layer {int(np.argmax(vals))})")
        # print a coarse sparkline over depth
        lo, hi = min(vals), max(vals)
        bars = "".join("▁▂▃▄▅▆▇█"[min(7, int((v - lo) / ((hi - lo) or 1) * 7))] for v in vals)
        print(f"  depth 0->{len(vals)-1}: {bars}")
        sl = [scalars[i] for i in sorted(scalars) if layer_types[i] == "sliding_attention"]
        fu = [scalars[i] for i in sorted(scalars) if layer_types[i] == "full_attention"]
        print(f"  sliding mean {np.mean(sl):.4f}   full mean {np.mean(fu):.4f}")
        out["layer_scalar"] = {"values": vals, "sliding_mean": float(np.mean(sl)),
                               "full_mean": float(np.mean(fu))}

    # --- norm magnitudes
    print("\nRMSNorm weight magnitude by kind (mean of |w| over layers)")
    out["norms"] = {}
    for nm in NORMS:
        if not norms[nm]:
            continue
        m = float(np.mean([s["absmean"] for s in norms[nm].values()]))
        mx = float(np.max([s["max_abs"] for s in norms[nm].values()]))
        print(f"  {nm:28s} absmean {m:8.3f}   max {mx:8.3f}")
        out["norms"][nm] = {"absmean": m, "max_abs": mx}

    # --- quantization stress
    print("\nW4A16 group-scale dynamic range  (p99.9/median of |scale|; higher = "
          "more outlier channels = harder to quantize)")
    print(f"  {'projection':>12s} {'all':>9s} {'sliding':>9s} {'full':>9s} "
          f"{'early(0-19)':>12s} {'late(40-59)':>12s}")
    out["quant_stress"] = {}
    for p in PROJ:
        if not scales[p]:
            continue
        idx = sorted(scales[p])
        dr = {i: scales[p][i]["dynrange"] for i in idx}
        sl = [dr[i] for i in idx if layer_types[i] == "sliding_attention"]
        fu = [dr[i] for i in idx if layer_types[i] == "full_attention"]
        early = [dr[i] for i in idx if i < 20]
        late = [dr[i] for i in idx if i >= 40]
        row = {"all": float(np.mean(list(dr.values()))),
               "sliding": float(np.mean(sl)) if sl else None,
               "full": float(np.mean(fu)) if fu else None,
               "early": float(np.mean(early)) if early else None,
               "late": float(np.mean(late)) if late else None,
               "worst_layer": int(max(dr, key=dr.get)), "worst": float(max(dr.values()))}
        print(f"  {p:>12s} {row['all']:9.2f} "
              f"{(row['sliding'] if row['sliding'] else float('nan')):9.2f} "
              f"{(row['full'] if row['full'] else float('nan')):9.2f} "
              f"{row['early']:12.2f} {row['late']:12.2f}")
        out["quant_stress"][p] = row


def pass_dynamic(model_id: str, out: dict, prompt: str, max_layers: int) -> None:
    import jax
    import jax.numpy as jnp
    from ports.gemma4.jax_31b_port import Streaming31BEngine
    from ports.gemma4 import jax_e_model as M

    print("\n=== DYNAMIC: residual stream through the stack ===\n")
    eng = Streaming31BEngine(model_id=model_id, quant_mode="w4a16",
                             max_model_len=512, window_kv=True)
    eng.load()
    cfg = eng.config

    rec = []

    def rms(x):
        return float(jnp.sqrt(jnp.mean(jnp.asarray(x, jnp.float32) ** 2)))

    attn_orig = M.Gemma4EAttentionJAX.__call__
    mlp_orig = M.Gemma4EMLPJAX.__call__

    def attn_patch(self, hidden_states, *a, **k):
        o = attn_orig(self, hidden_states, *a, **k)
        first = o[0] if isinstance(o, tuple) else o
        rec.append(("attn", self.layer_type, rms(hidden_states), rms(first)))
        return o

    def mlp_patch(self, x, *a, **k):
        o = mlp_orig(self, x, *a, **k)
        rec.append(("mlp", "-", rms(x), rms(o)))
        return o

    M.Gemma4EAttentionJAX.__call__ = attn_patch
    M.Gemma4EMLPJAX.__call__ = mlp_patch
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        ids = jnp.asarray(tok(text, add_special_tokens=False,
                              return_tensors="np")["input_ids"], dtype=jnp.int32)
        S = ids.shape[1]
        pos = jnp.arange(S, dtype=jnp.int32)[None, :]
        valid = jnp.ones((1, S), dtype=bool)
        logits = M.Gemma4EModelJAX.__call__(
            eng.model, ids, eng.params, pos,
            attention_mask=M.make_prefill_causal_mask(valid), quant_mode="w4a16",
            sliding_attention_mask=M.make_prefill_causal_mask(
                valid, window=cfg.sliding_window))
    finally:
        M.Gemma4EAttentionJAX.__call__ = attn_orig
        M.Gemma4EMLPJAX.__call__ = mlp_orig

    attn = [r for r in rec if r[0] == "attn"]
    mlp = [r for r in rec if r[0] == "mlp"]
    print(f"  prompt tokens: {S}, recorded {len(attn)} attn / {len(mlp)} mlp blocks\n")
    print(f"  {'layer':>5} {'type':>18} {'resid_in':>11} {'attn_out':>11} "
          f"{'mlp_out':>11} {'attn/resid':>11}")
    rows = []
    shown = 0
    for i, (a, m) in enumerate(zip(attn, mlp)):
        ratio = a[3] / (a[2] or 1e-9)
        rows.append({"layer": i, "type": a[1], "resid_in": a[2],
                     "attn_out": a[3], "mlp_out": m[3], "attn_over_resid": ratio})
        near_end = i >= len(attn) - 3
        if i < 6 or a[1] == "full_attention" or near_end:
            if shown < max_layers or near_end:
                print(f"  {i:>5} {a[1]:>18} {a[2]:11.4g} {a[3]:11.4g} "
                      f"{m[3]:11.4g} {ratio:11.4f}")
                shown += 1

    sl = [r for r in rows if r["type"] == "sliding_attention"]
    fu = [r for r in rows if r["type"] == "full_attention"]
    print(f"\n  sliding layers (n={len(sl)}): attn/resid mean "
          f"{np.mean([r['attn_over_resid'] for r in sl]):.4f}")
    print(f"  full    layers (n={len(fu)}): attn/resid mean "
          f"{np.mean([r['attn_over_resid'] for r in fu]):.4f}")
    print(f"  residual RMS: layer0 in {rows[0]['resid_in']:.4g} -> "
          f"layer{len(rows)-1} in {rows[-1]['resid_in']:.4g} "
          f"({rows[-1]['resid_in'] / (rows[0]['resid_in'] or 1e-9):.2f}x over the stack)")

    lg = np.asarray(logits[:, -1, :], dtype=np.float32)[0]
    cap = cfg.logit_softcapping
    srt = np.sort(lg)[::-1]
    p = np.exp(lg - lg.max()); p /= p.sum()
    ent = float(-(p * np.log(np.clip(p, 1e-30, None))).sum())
    top1_mass = float(p[0] if False else p.max())
    print(f"\n  logits: rms {float(np.sqrt((lg**2).mean())):.4g}  "
          f"max {srt[0]:.4g}  top1-top2 margin {srt[0]-srt[1]:.4g}")
    print(f"  at softcap ({cap}): {float((np.abs(lg) >= cap*0.999).mean()):.6f}   "
          f"top-1 prob {top1_mass:.4f}   top-10 mass {float(np.sort(p)[::-1][:10].sum()):.4f}   "
          f"entropy {ent:.3f} nats")

    out["dynamic"] = {
        "prompt_tokens": int(S), "layers": rows,
        "sliding_attn_over_resid": float(np.mean([r["attn_over_resid"] for r in sl])),
        "full_attn_over_resid": float(np.mean([r["attn_over_resid"] for r in fu])),
        "resid_growth": float(rows[-1]["resid_in"] / (rows[0]["resid_in"] or 1e-9)),
        "logits": {"rms": float(np.sqrt((lg**2).mean())), "max": float(srt[0]),
                   "margin": float(srt[0]-srt[1]),
                   "frac_at_softcap": float((np.abs(lg) >= cap*0.999).mean()),
                   "top1_prob": top1_mass,
                   "top10_mass": float(np.sort(p)[::-1][:10].sum()),
                   "entropy_nats": ent},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--passes", default="static,dynamic")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--max-print-layers", type=int, default=18)
    ap.add_argument("--json-out", default="gemma4_31b_model_intel.json")
    args = ap.parse_args()

    out = {"model": args.model}
    passes = [p.strip() for p in args.passes.split(",")]
    if "static" in passes:
        pass_static(args.model, out)
    if "dynamic" in passes:
        pass_dynamic(args.model, out, args.prompt, args.max_print_layers)

    with open(args.json_out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
