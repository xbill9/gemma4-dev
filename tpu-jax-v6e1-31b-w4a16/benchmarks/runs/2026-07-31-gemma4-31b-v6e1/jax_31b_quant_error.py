"""Measure the 31B's ACTUAL W4A16 quantization error — after checking that is even
a meaningful thing to measure.

MODEL-INTEL.md section 5 ranks projections by group-scale dynamic range, which is a
*proxy* for quantization difficulty. The real number needs a reference: dequantize
`-qat-w4a16-ct` and compare against `-qat-q4_0-unquantized`, which ships QAT weights
in half precision.

THE CONTROL COMES FIRST. Those two checkpoints are QAT'd for *different* schemes
(int4 group-32 vs q4_0), so they may be separate training runs with genuinely
different weights. If so, a diff between them measures "distance between two QAT
variants", not "W4A16 round-trip error", and every per-layer number would be
meaningless.

The test: tensors that are UNQUANTIZED in both files — RMSNorm weights,
`layer_scalar`, `embed_tokens`. If those are bit-identical the two checkpoints share
a base and the comparison is valid. If they differ, stop and report that instead.

    python3.13 ports/gemma4/jax_31b_quant_error.py
"""

import argparse
import glob
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from ports.gemma4.jax_31b_model_intel import SafeReader, _snapshot

W4A16 = "google/gemma-4-31B-it-qat-w4a16-ct"
DENSE = "google/gemma-4-31B-it-qat-q4_0-unquantized"
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]


class MultiReader:
    """SafeReader over a sharded checkpoint."""

    def __init__(self, path: str):
        self.readers = [SafeReader(p) for p in
                        sorted(glob.glob(os.path.join(path, "*.safetensors")))]
        self.index = {}
        for r in self.readers:
            for k in r.keys():
                self.index[k] = r

    def keys(self):
        return list(self.index)

    def get(self, name):
        return self.index[name].get(name)

    def get_raw(self, name):
        return self.index[name].get_raw(name)


def unpack_w4a16(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """compressed-tensors int4: packed int32 [out, in/8] -> dense [out, in].

    Eight BIASED 4-bit values per int32 (stored = value + 8), low nibble first,
    each group of 32 input
    columns sharing one bf16 scale. Mirrors the engine's unpack; done in numpy here
    so this runs without a TPU.
    """
    # packed arrives as native int32; going through float32 would round any word
    # above 2**24 and silently corrupt the nibbles.
    assert packed.dtype == np.int32, f"packed must be int32, got {packed.dtype}"
    p = packed.astype(np.int64) & 0xFFFFFFFF
    out, k8 = p.shape
    # compressed-tensors uses a BIASED encoding, not two's complement: the stored
    # nibble is value+8, so 0 -> -8, 8 -> 0, 15 -> +7. Sign-extending instead
    # (0->0, 8->-8) scrambles the weights and yields SNR around -8 dB, which is
    # how this was caught. Matches jax_e_model.py:305.
    nib = np.stack([((p >> (4 * i)) & 0xF).astype(np.int16) - 8 for i in range(8)],
                   axis=-1).astype(np.float32)                    # [out, in/8, 8]
    dense = nib.reshape(out, k8 * 8)                              # [out, in]
    g = dense.shape[1] // scale.shape[1]                          # group size (32)
    s = np.repeat(scale.astype(np.float32), g, axis=1)
    return dense * s


def control(a: MultiReader, b: MultiReader, n_layers: int, out: dict) -> bool:
    """Are the two checkpoints the same base model?"""
    print("\n=== CONTROL: do the unquantized tensors match? ===\n")
    pre = "model.language_model."
    names = [f"{pre}norm.weight"]
    for i in (0, 30, 59):
        for nm in ("input_layernorm", "post_attention_layernorm",
                   "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            names.append(f"{pre}layers.{i}.{nm}.weight")
        names.append(f"{pre}layers.{i}.layer_scalar")
    names.append(f"{pre}embed_tokens.weight")

    rows, all_same = [], True
    for nm in names:
        if nm not in a.index or nm not in b.index:
            print(f"  {nm.replace(pre,''):46s} MISSING in one file")
            all_same = False
            continue
        x, y = a.get(nm), b.get(nm)
        if x.shape != y.shape:
            print(f"  {nm.replace(pre,''):46s} SHAPE {x.shape} vs {y.shape}")
            all_same = False
            continue
        ident = bool(np.array_equal(x, y))
        den = float(np.abs(x).max()) or 1.0
        rel = float(np.abs(x - y).max()) / den
        all_same &= ident
        rows.append({"tensor": nm.replace(pre, ""), "identical": ident, "rel_max": rel})
        print(f"  {nm.replace(pre,''):46s} identical={ident!s:5s} rel_max={rel:.3e}")
    out["control"] = {"identical": all_same, "tensors": rows}
    print(f"\n  => base weights {'MATCH' if all_same else 'DIFFER'}")
    return all_same


def measure(a: MultiReader, b: MultiReader, layers, out: dict) -> None:
    print("\n=== W4A16 round-trip error vs the half-precision QAT weights ===\n")
    pre = "model.language_model."
    print(f"  {'layer':>5} {'projection':>11} {'shape':>18} {'rel_fro':>10} "
          f"{'rel_max':>10} {'SNR dB':>8}")
    rows = []
    for i in layers:
        for p in ATTN + MLP:
            sub = "self_attn" if p in ATTN else "mlp"
            pk = f"{pre}layers.{i}.{sub}.{p}.weight_packed"
            sk = f"{pre}layers.{i}.{sub}.{p}.weight_scale"
            dk = f"{pre}layers.{i}.{sub}.{p}.weight"
            if pk not in a.index or dk not in b.index:
                continue
            q = unpack_w4a16(a.get_raw(pk), a.get(sk))
            d = b.get(dk).astype(np.float32)
            if q.shape != d.shape:
                print(f"  {i:>5} {p:>11} shape {q.shape} vs {d.shape} — skipped")
                continue
            err = q - d
            fro = float(np.linalg.norm(err) / (np.linalg.norm(d) or 1e-12))
            rmx = float(np.abs(err).max() / (np.abs(d).max() or 1e-12))
            snr = float(20 * np.log10(1.0 / fro)) if fro > 0 else float("inf")
            print(f"  {i:>5} {p:>11} {str(q.shape):>18} {fro:10.5f} {rmx:10.5f} {snr:8.2f}")
            rows.append({"layer": i, "proj": p, "shape": list(q.shape),
                         "rel_fro": fro, "rel_max": rmx, "snr_db": snr})
    out["error"] = rows
    if rows:
        import collections
        by = collections.defaultdict(list)
        for r in rows:
            by[r["proj"]].append(r["rel_fro"])
        print("\n  mean relative Frobenius error by projection:")
        for p, v in sorted(by.items(), key=lambda kv: -float(np.mean(kv[1]))):
            print(f"    {p:>11} {float(np.mean(v)):.5f}")
        out["by_projection"] = {p: float(np.mean(v)) for p, v in by.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,10,30,50,59")
    ap.add_argument("--json-out", default="gemma4_31b_quant_error.json")
    ap.add_argument("--force", action="store_true",
                    help="measure even if the control fails (results will be meaningless)")
    args = ap.parse_args()

    a = MultiReader(_snapshot(W4A16))
    b = MultiReader(_snapshot(DENSE))
    cfg = json.load(open(os.path.join(_snapshot(W4A16), "config.json")))
    n = cfg.get("text_config", cfg)["num_hidden_layers"]
    out = {"w4a16": W4A16, "dense": DENSE}

    ok = control(a, b, n, out)
    if ok or args.force:
        measure(a, b, [int(x) for x in args.layers.split(",")], out)
    else:
        print("\n  Control failed: these are different weights, so a per-tensor diff\n"
              "  would measure the distance between two QAT runs rather than the\n"
              "  W4A16 round-trip. Not measuring. Re-run with --force to override.")

    with open(args.json_out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
