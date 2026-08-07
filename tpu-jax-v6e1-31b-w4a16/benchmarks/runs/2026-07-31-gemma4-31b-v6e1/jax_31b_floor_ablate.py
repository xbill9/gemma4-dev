"""Locate the 31B's S-independent prefill temp floor by ablation.

Two previous attempts to name it failed: resident-memory inference (wrong), and
ranking HLO instructions by output shape (wrong — fusion means instruction size is
not allocation). A buffer-assignment dump would answer it directly but
`--xla_dump_hlo_pass_re=buffer-assignment` emits nothing on this TPU build.

So: ablate and watch `memory_analysis().temp_size_in_bytes`, which is the same
number that predicted every pass/fail in the sweep and is therefore trustworthy.

The decisive question is whether the ~9.2 GB floor at S=1024 is PER-LAYER working
set (dequantized W4A16 weights held live) or GLOBAL (the lm_head / 2.82 GB tied
embedding table). Truncating the layer count separates them cleanly:

  temp scales with layer count  -> per-layer working set
  temp flat vs layer count      -> global, i.e. the lm_head / embedding path

Also dumps everything XLA will emit, in case a buffer-assignment file appears
without the pass filter.

    python3.13 ports/gemma4/jax_31b_floor_ablate.py
"""

import argparse
import copy
import glob
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DUMP_ROOT = "/tmp/xladump_all"
if os.environ.get("ABLATE_DUMP", "0") == "1":
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + f" --xla_dump_to={DUMP_ROOT}").strip()

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402

from ports.gemma4.jax_e_model import (                            # noqa: E402
    Gemma4EModelJAX,
    pad_to_tpu_v6e_bucket,
    prefill_with_kv_cache,
)
from ports.gemma4.jax_31b_port import Streaming31BEngine          # noqa: E402

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"


def truncated(eng, n_layers: int):
    """A model + params restricted to the first n_layers, everything else intact.

    Keeps embed_tokens and final_norm, so the lm_head path is untouched — which is
    exactly what makes this a clean separator.
    """
    cfg = copy.copy(eng.config)
    cfg.num_hidden_layers = n_layers
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = list(eng.config.layer_types)[:n_layers]
    model = Gemma4EModelJAX(cfg)
    params = {k: v for k, v in eng.params.items()
              if not k.startswith("layer_")}
    for i in range(n_layers):
        params[f"layer_{i}"] = eng.params[f"layer_{i}"]
    return model, params


def temp_of(model, params, B, S):
    ids = jnp.ones((B, S), dtype=jnp.int32)
    padded, valid = pad_to_tpu_v6e_bucket(ids)
    comp = jax.jit(
        prefill_with_kv_cache,
        static_argnames=("model", "max_new_tokens", "quant_mode",
                         "cache_dtype", "window_kv"),
    ).lower(model=model, prompt_ids=padded, prompt_valid=valid, params=params,
            max_new_tokens=8, quant_mode="w4a16", window_kv=True).compile()
    m = comp.memory_analysis()
    return (getattr(m, "temp_size_in_bytes", 0),
            getattr(m, "argument_size_in_bytes", 0),
            getattr(m, "output_size_in_bytes", 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer-counts", default="1,2,6,15,30,60")
    ap.add_argument("--S", type=int, default=1024)
    ap.add_argument("--json-out", default="gemma4_31b_floor_ablate.json")
    args = ap.parse_args()

    eng = Streaming31BEngine(model_id=args.model, quant_mode="w4a16",
                             max_model_len=2048, window_kv=True)
    eng.load()
    print(f"\nweights {eng.weight_bytes / 1e9:.3f} GB    S={args.S}, B=1\n")
    print(f"  {'layers':>7s}  {'temp':>10s}  {'args':>10s}  {'output':>10s}  {'d(temp)/layer':>14s}")

    rows, prev = [], None
    for n in [int(x) for x in args.layer_counts.split(",")]:
        try:
            model, params = truncated(eng, n)
            t, a, o = temp_of(model, params, 1, args.S)
            slope = ""
            if prev is not None:
                dn = n - prev[0]
                slope = f"{(t - prev[1]) / dn / 1e6:10.1f} MB"
            print(f"  {n:7d}  {t / 1e9:8.3f} GB  {a / 1e9:8.3f} GB  "
                  f"{o / 1e9:8.3f} GB  {slope:>14s}")
            rows.append({"layers": n, "temp_gb": round(t / 1e9, 4),
                         "args_gb": round(a / 1e9, 4),
                         "output_gb": round(o / 1e9, 4), "status": "OK"})
            prev = (n, t)
        except Exception as exc:
            msg = str(exc).split("\n")[0][:100]
            print(f"  {n:7d}  FAILED: {msg}")
            rows.append({"layers": n, "status": "FAILED", "error": msg})

    out = {"weights_gb": round(eng.weight_bytes / 1e9, 4), "S": args.S,
           "by_layers": rows}

    ok = [r for r in rows if r.get("status") == "OK"]
    if len(ok) >= 2:
        lo, hi = ok[0], ok[-1]
        dn = hi["layers"] - lo["layers"]
        per_layer = (hi["temp_gb"] - lo["temp_gb"]) / dn if dn else 0.0
        intercept = lo["temp_gb"] - per_layer * lo["layers"]
        print(f"\n  linear fit over {lo['layers']}..{hi['layers']} layers:")
        print(f"    per-layer   {per_layer * 1000:8.1f} MB/layer")
        print(f"    intercept   {intercept:8.3f} GB   <- layer-independent floor")
        print(f"    => at 60 layers: {per_layer * 60:.3f} GB layers + "
              f"{intercept:.3f} GB fixed")
        out["fit"] = {"per_layer_gb": round(per_layer, 5),
                      "intercept_gb": round(intercept, 4)}

    if os.environ.get("ABLATE_DUMP", "0") == "1":
        files = sorted(glob.glob(os.path.join(DUMP_ROOT, "*")))
        print(f"\n  XLA dump: {len(files)} files")
        for f in files[:15]:
            print("   ", os.path.basename(f))
        out["dump_files"] = [os.path.basename(f) for f in files[:40]]

    with open(args.json_out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
