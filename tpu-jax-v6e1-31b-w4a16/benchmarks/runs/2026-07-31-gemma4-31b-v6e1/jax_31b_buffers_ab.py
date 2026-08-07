"""Two de-confounding measurements for the 31B prefill ceiling, one process.

Both exist because earlier conclusions in this investigation were inferred rather
than measured, and two of them were wrong.

PHASE 1 — name the buffers instead of inferring them.
  Compiles prefill at several S, parses the OPTIMIZED HLO, and ranks every
  instruction by materialized bytes. Answers two open questions:
    (a) what is the ~10 GB temp floor that is present already at S=1024?
    (b) which buffers scale with S^2, i.e. what would explode at 8K?
  8192 does not compile, so it cannot be dumped this way; the S-scaling of the
  named buffers at 1K/2K/4K is the evidence available.

PHASE 2 — same-chip, same-process A/B of the logits_at fix.
  The earlier pre/post comparison ran on two different VMs, one sample each, so
  "B=2 x 2K went OOM -> passing" was not attributable to the fix. Here the old
  prefill body is reconstructed locally (model without logits_at, slice after)
  and compiled back-to-back against the shipped one on the same chip. Compile-only
  via memory_analysis, so cells that cannot execute still yield numbers.

    python3.13 ports/gemma4/jax_31b_buffers_ab.py
"""

import argparse
import collections
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

from ports.gemma4.jax_e_model import (
    init_kv_cache,
    make_prefill_causal_mask,
    pad_to_tpu_v6e_bucket,
    prefill_with_kv_cache,
)
from ports.gemma4.jax_31b_port import Streaming31BEngine

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"

_DTYPE_BYTES = {"pred": 1, "s8": 1, "u8": 1, "s16": 2, "u16": 2, "bf16": 2, "f16": 2,
                "s32": 4, "u32": 4, "f32": 4, "s64": 8, "u64": 8, "f64": 8,
                "c64": 8, "c128": 16, "f8e4m3fn": 1, "f8e5m2": 1}

# e.g.  %fusion.3 = f32[1,32,4096,4096]{3,2,1,0} fusion(...), kind=kLoop, calls=...
_SHAPE_RE = re.compile(r"\b([a-z0-9]+)\[([\d,]+)\]")
_INSTR_RE = re.compile(r"^\s*(?:ROOT\s+)?%?([\w.\-]+)\s*=\s*([a-z0-9]+)\[([\d,]+)\][^=]*?\s([a-z-]+)\(")


def _bytes_of(dtype: str, dims: str) -> int:
    w = _DTYPE_BYTES.get(dtype)
    if w is None:
        return 0
    n = 1
    for d in dims.split(","):
        if not d:
            return 0
        n *= int(d)
    return n * w


def largest_buffers(hlo_text: str, top: int = 18):
    """Rank instructions in the optimized HLO by output bytes."""
    rows = []
    for line in hlo_text.splitlines():
        m = _INSTR_RE.match(line)
        if not m:
            continue
        name, dtype, dims, opcode = m.groups()
        nb = _bytes_of(dtype, dims)
        if nb:
            rows.append({"name": name, "opcode": opcode,
                         "shape": f"{dtype}[{dims}]", "bytes": nb})
    rows.sort(key=lambda r: -r["bytes"])
    return rows[:top], rows


def _compile_prefill(fn, eng, B, S, **extra):
    ids = jnp.ones((B, S), dtype=jnp.int32)
    padded, valid = pad_to_tpu_v6e_bucket(ids)
    jitted = jax.jit(fn, static_argnames=("model", "max_new_tokens", "quant_mode",
                                          "cache_dtype", "window_kv"))
    return jitted.lower(
        model=eng.model, prompt_ids=padded, prompt_valid=valid, params=eng.params,
        max_new_tokens=8, quant_mode="w4a16", window_kv=True, **extra
    ).compile()


# ---------------------------------------------------------------- old prefill

def old_prefill_with_kv_cache(model, prompt_ids, prompt_valid, params, max_new_tokens,
                              quant_mode="w4a16", cache_dtype=jnp.bfloat16,
                              window_kv=None):
    """The prefill body as it was BEFORE the logits_at change.

    Kept here rather than reverting the shipped file so both paths can be compiled
    in one process on one chip. Deliberately a verbatim copy of the old logic:
    full-sequence logits, then take_along_axis.
    """
    B, S = prompt_ids.shape
    total_len = S + max_new_tokens
    if window_kv is None:
        win = model.config.sliding_window
        window_kv = bool(win) and total_len > int(win)
    caches = init_kv_cache(model.config, batch_size=B, max_seq_len=total_len,
                           dtype=cache_dtype, window_kv=window_kv)
    position_ids = jnp.arange(S, dtype=jnp.int32)[None, :].repeat(B, axis=0)
    window = model.config.sliding_window
    mask = make_prefill_causal_mask(prompt_valid)
    sliding_mask = (make_prefill_causal_mask(prompt_valid, window=window)
                    if window is not None else None)
    logits, caches = model(
        prompt_ids, params, position_ids, attention_mask=mask, quant_mode=quant_mode,
        kv_caches=caches, cache_slot=jnp.int32(0), sliding_attention_mask=sliding_mask,
    )
    prompt_lens = prompt_valid.sum(axis=1).astype(jnp.int32)
    last_logits = jnp.take_along_axis(logits, (prompt_lens - 1)[:, None, None], axis=1)[:, 0, :]
    valid = jnp.concatenate(
        [prompt_valid, jnp.zeros((B, max_new_tokens), dtype=jnp.bool_)], axis=1)
    return last_logits, caches, valid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--buffer-contexts", default="1024,2048,4096")
    ap.add_argument("--ab-cells", default="1x2048,1x4096,2x2048,2x4096,1x8192")
    ap.add_argument("--json-out", default="gemma4_31b_buffers_ab.json")
    args = ap.parse_args()

    eng = Streaming31BEngine(model_id=args.model, quant_mode="w4a16",
                             max_model_len=4096, window_kv=True)
    eng.load()
    limit = eng.memory_stats()["hbm_bytes_limit"]
    out = {"weights_gb": round(eng.weight_bytes / 1e9, 4),
           "hbm_limit_gb": round(limit / 1e9, 4)}
    print(f"\nweights {eng.weight_bytes / 1e9:.3f} GB   HBM limit {limit / 1e9:.3f} GB")

    # ------------------------------------------------- PHASE 1: name the buffers
    print("\n" + "=" * 92)
    print("PHASE 1 — largest buffers in the optimized HLO")
    print("=" * 92)
    by_S = {}
    phase1 = {}
    for S in [int(x) for x in args.buffer_contexts.split(",")]:
        try:
            comp = _compile_prefill(prefill_with_kv_cache, eng, 1, S)
            m = comp.memory_analysis()
            temp = getattr(m, "temp_size_in_bytes", 0)
            text = comp.as_text()
            top, allrows = largest_buffers(text)
            print(f"\n--- B=1 S={S}   temp {temp / 1e9:.3f} GB   "
                  f"({len(allrows)} instructions parsed) ---")
            for r in top:
                print(f"   {r['bytes'] / 1e9:8.3f} GB  {r['opcode']:14s} {r['shape']}")
            by_S[S] = {r["name"]: r for r in allrows}
            phase1[S] = {"temp_gb": round(temp / 1e9, 4),
                         "top": [{**r, "gb": round(r["bytes"] / 1e9, 4)} for r in top]}
        except Exception as exc:
            msg = str(exc).split("\n")[0][:120]
            print(f"\n--- B=1 S={S}: FAILED: {msg}")
            phase1[S] = {"status": "FAILED", "error": msg}
    out["phase1_buffers"] = phase1

    # Which buffer shapes scale with S^2 between the two largest compiled sizes?
    sizes = sorted(k for k in by_S if by_S[k])
    if len(sizes) >= 2:
        lo, hi = sizes[-2], sizes[-1]
        ratio_S = hi / lo
        print(f"\n--- scaling of named buffers, S={lo} -> S={hi} (S ratio {ratio_S:.0f}x) ---")
        agg = collections.defaultdict(lambda: [0, 0])
        for name, r in by_S[lo].items():
            agg[r["opcode"]][0] += r["bytes"]
        for name, r in by_S[hi].items():
            agg[r["opcode"]][1] += r["bytes"]
        scaling = []
        for op, (b_lo, b_hi) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
            if b_hi < 5e8:
                continue
            growth = (b_hi / b_lo) if b_lo else float("inf")
            tag = ("S^2" if growth > ratio_S * 1.5 else
                   "S" if growth > 1.3 else "fixed")
            print(f"   {op:16s} {b_lo / 1e9:7.2f} -> {b_hi / 1e9:7.2f} GB   "
                  f"{growth:5.2f}x   [{tag}]")
            scaling.append({"opcode": op, "lo_gb": round(b_lo / 1e9, 3),
                            "hi_gb": round(b_hi / 1e9, 3),
                            "growth": round(growth, 3), "class": tag})
        out["phase1_scaling"] = {"lo_S": lo, "hi_S": hi, "by_opcode": scaling}

    # ------------------------------------------------ PHASE 2: same-chip A/B
    print("\n" + "=" * 92)
    print("PHASE 2 — logits_at A/B, same chip, same process, back to back")
    print("=" * 92)
    print(f"  {'cell':>10s}  {'OLD temp':>12s}  {'NEW temp':>12s}  {'saved':>10s}")
    ab = []
    for cell in args.ab_cells.split(","):
        B, S = (int(x) for x in cell.split("x"))
        row = {"B": B, "S": S}
        for label, fn in (("old", old_prefill_with_kv_cache), ("new", prefill_with_kv_cache)):
            try:
                comp = _compile_prefill(fn, eng, B, S)
                m = comp.memory_analysis()
                row[f"{label}_temp_gb"] = round(getattr(m, "temp_size_in_bytes", 0) / 1e9, 4)
                row[f"{label}_status"] = "COMPILED"
            except Exception as exc:
                row[f"{label}_status"] = "FAILED"
                row[f"{label}_error"] = str(exc).split("\n")[0][:110]
        o, n = row.get("old_temp_gb"), row.get("new_temp_gb")
        saved = f"{o - n:.3f} GB" if (o is not None and n is not None) else "—"
        print(f"  {f'B={B} S={S}':>10s}  "
              f"{(f'{o:.3f} GB' if o is not None else row['old_status']):>12s}  "
              f"{(f'{n:.3f} GB' if n is not None else row['new_status']):>12s}  {saved:>10s}")
        ab.append(row)
    out["phase2_ab"] = ab

    with open(args.json_out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
