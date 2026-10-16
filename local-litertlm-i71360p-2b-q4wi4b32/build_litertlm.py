"""Build a .litertlm of Gemma 4 E2B-it that keeps the QAT Q4_0 grid.

    python3 build_litertlm.py SRC_HF_DIR OUT_DIR [--cache-length N] [--prefill 128,1024]

SRC_HF_DIR is google/gemma-4-E2B-it-qat-q4_0-unquantized. Every FULLY_CONNECTED and
EMBEDDING_LOOKUP weight is stored as LiteRT's dynamic int4 in blocks of 32
(ai_edge_quantizer `dynamic_wi4b32_afp32`), text only.

What makes it a repack rather than a re-quantization: ai_edge_quantizer sets each block's
scale to max|w| / 7, which moves every group whose extreme value sits on level -8 and
re-rounds the rest. Here each block's min/max is replaced by +-7 * the block's trained
step (recovered as in gguf_exact.py), so the scale IS the trained step and every level is
the QAT level. LiteRT stores blockwise scales at bf16 precision, which is the one
rounding left; grid_scales.py in benchmarks/runs/2026-10-15-qat-grid-bf16-scales-i71360p
measures what that costs. A block off the 4-bit grid keeps the default scale and is
counted in build_report.json; the build does not stop for it, the report says how many.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

QK, REL_TOL = 32, 2.0**-7


def grid_step(g):
    """g [..., 32] float32 -> (step [..., 1], on_grid [..., 1] bool)."""
    amax = np.abs(g).max(-1, keepdims=True)
    nz = amax > 0
    tol = amax * REL_TOL
    d = np.where(nz, amax / 7, 0).astype(np.float32)
    ok = ~nz
    for m in range(1, 9):
        cand = np.where(nz, amax / m, 1).astype(np.float32)
        q = np.clip(np.rint(g / cand), -8, 7)
        good = (np.abs(g - q * cand).max(-1, keepdims=True) <= tol) & nz & ~ok
        d = np.where(good, cand, d)
        ok |= good
    q = np.clip(np.rint(g / np.where(d > 0, d, 1)), -8, 7)
    den, num = (q * q).sum(-1, keepdims=True), (q * g).sum(-1, keepdims=True)
    refined = np.where(den > 0, num / np.where(den > 0, den, 1), d).astype(np.float32)
    return np.where(ok & nz, refined, d), ok


def grid_step_chunked(g, rows=4096):
    """grid_step over the leading axis in chunks. Each candidate pass makes several
    temporaries the size of its input; on the whole per-layer embedding table (9.4 GB
    float32) that took 25+ GB and thrashed swap."""
    step = np.empty(g.shape[:-1] + (1,), np.float32)
    ok = np.empty(g.shape[:-1] + (1,), bool)
    for i in range(0, g.shape[0], rows):
        step[i : i + rows], ok[i : i + rows] = grid_step(g[i : i + rows])
    return step, ok


REPORT = {"calls": []}


def patch_quantizer():
    from ai_edge_quantizer import qtyping
    from ai_edge_quantizer.algorithms.uniform_quantize import common_quantize, uniform_quantize_tensor

    original = common_quantize.init_tensor_min_max

    def init_tensor_min_max(tensor_data, op_info):
        out = original(tensor_data, op_info)
        cfg = op_info.op_quant_config.weight_tensor_config
        if (
            tensor_data is None
            or cfg is None
            or cfg.num_bits != 4
            or cfg.granularity != qtyping.QuantGranularity.BLOCKWISE_32
        ):
            return out
        blocks, axis = uniform_quantize_tensor.reshape_data_for_blockwise(
            np.asarray(tensor_data, np.float32), op_info.op_name, cfg.granularity
        )
        g = np.moveaxis(blocks, axis, -1)
        assert g.shape[-1] == QK, g.shape
        step, ok = grid_step_chunked(g)
        bound = np.moveaxis(7 * step, -1, axis).squeeze(axis)
        okb = np.moveaxis(ok, -1, axis).squeeze(axis)
        assert bound.shape == out["max"].shape, (bound.shape, out["max"].shape)
        dflt = np.maximum(np.abs(out["min"]), np.abs(out["max"]))
        bound = np.where(okb, bound, dflt).astype(out["max"].dtype)
        REPORT["calls"].append(
            {
                "op": str(op_info.op_name),
                "index": op_info.subgraph_op_index,
                "shape": list(tensor_data.shape),
                "blocks": int(okb.size),
                "off_grid": int((~okb).sum()),
            }
        )
        return {"min": -bound, "max": bound}

    common_quantize.init_tensor_min_max = init_tensor_min_max

    # ai_edge_quantizer keys everything by tensor name and requires names unique across
    # ALL subgraphs. Converter V2 gives each signature (decode, prefill_128, prefill_1024)
    # its own subgraph with the same input names, and the in-place KV update names its
    # output like its input (kv_cache_k_0 six times in one model). Uniquify the file
    # before the quantizer reads it; see uniquify_tensor_names.
    from ai_edge_quantizer import quantizer as quantizer_lib

    original_init = quantizer_lib.Quantizer.__init__

    def __init__(self, float_model, *args, **kwargs):
        if isinstance(float_model, (str, os.PathLike)):
            REPORT.setdefault("renamed", {})[os.path.basename(str(float_model))] = uniquify_tensor_names(
                str(float_model)
            )
        original_init(self, float_model, *args, **kwargs)

    quantizer_lib.Quantizer.__init__ = __init__


# Never the first byte of a real tensor name here; one per repeat of a name.
_RENAME_MARKS = b"#$%&@!~^"


def uniquify_tensor_names(path):
    """Make every tensor name in a .tflite unique, in place, without moving a byte.

    The second and later occurrences of a name get their FIRST byte replaced by a mark
    (one per occurrence), so string lengths and every flatbuffer offset are unchanged.
    Refuses when a repeated name carries constant data: that would be two weights the
    quantizer could not tell apart, which renaming would hide rather than fix.
    Signatures bind inputs and outputs by tensor index, not by tensor name.
    Returns the number of names changed.
    """
    import mmap

    from ai_edge_litert import schema_py_generated as fb

    with open(path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        model = fb.Model.GetRootAs(mm, 0)
        seen, changed = {}, 0
        for si in range(model.SubgraphsLength()):
            sg = model.Subgraphs(si)
            for ti in range(sg.TensorsLength()):
                t = sg.Tensors(ti)
                name = t.Name()
                k = seen.get(name, 0)
                seen[name] = k + 1
                if k == 0:
                    continue
                b = model.Buffers(t.Buffer()) if t.Buffer() else None
                if b is not None and (b.DataLength() or b.Size()):
                    raise SystemExit(f"{path}: repeated tensor name {name!r} carries constant data")
                if k > len(_RENAME_MARKS):
                    raise SystemExit(f"{path}: {name!r} repeats more than {len(_RENAME_MARKS)} times")
                o = t._tab.Offset(10) + t._tab.Pos
                start = o + int.from_bytes(mm[o : o + 4], "little") + 4
                assert mm[start : start + len(name)] == name
                mm[start] = _RENAME_MARKS[k - 1]
                changed += 1
        mm.flush()
        mm.close()
    return changed


def recipe_json(path):
    from ai_edge_quantizer import recipe as r

    rec = r.dynamic_wi4b32_afp32(operation_name=r.TFLOperationName.FULLY_CONNECTED) + r.dynamic_wi4b32_afp32(
        operation_name=r.TFLOperationName.EMBEDDING_LOOKUP
    )
    with open(path, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    return path


def quantize_one(src, recipe, dst, report):
    """Quantize one .tflite in THIS process (the --quantize subprocess)."""
    patch_quantizer()
    from ai_edge_quantizer import quantizer as quantizer_lib

    qt = quantizer_lib.Quantizer(src)
    qt.load_quantization_recipe(recipe)
    qt.quantize().export_model(dst, overwrite=True)
    with open(report, "w") as f:
        json.dump(REPORT, f, indent=1)


def patch_export_quantization(reuse_dir):
    """Quantize each exported component in a fresh subprocess, or reuse a finished one.

    The exporter quantizes every component inside its own process, after the source
    model and the earlier components have been through it. On this 15 GiB host that was
    OOM-killed at the last and largest component, the per-layer embedder (9.4 GB float
    file, 2026-10-15). A fresh process per component starts from nothing. With
    --reuse DIR, a component whose `<name>_quantized.tflite` is already in DIR (built by
    this script) is hard-linked instead of quantized again.
    """
    import shutil
    import subprocess

    from litert_torch.generative.export_hf.core import export_lib

    def maybe_quantize_model(model_path, recipe):
        if not recipe or str(recipe).strip().lower() in ("none", "null", "false", ""):
            return model_path
        dst = model_path.removesuffix(".tflite").removesuffix("_quantized") + "_quantized.tflite"
        name = os.path.basename(dst)
        if reuse_dir and os.path.exists(os.path.join(reuse_dir, name)):
            os.link(os.path.join(reuse_dir, name), dst) if not os.path.exists(dst) else None
            REPORT.setdefault("components", {})[name] = {"reused": os.path.join(reuse_dir, name)}
            print(f"reused {name}", file=sys.stderr, flush=True)
            return dst
        report = dst + ".grid.json"
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--quantize", model_path, recipe, dst, report], check=True
        )
        with open(report) as f:
            r = json.load(f)
        REPORT["calls"] += r["calls"]
        REPORT.setdefault("components", {})[name] = {k: v for k, v in r.items() if k != "calls"}
        if reuse_dir:
            os.makedirs(reuse_dir, exist_ok=True)
            shutil.copy2(dst, os.path.join(reuse_dir, name))
        return dst

    export_lib.maybe_quantize_model = maybe_quantize_model


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--quantize":
        quantize_one(*sys.argv[2:6])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--cache-length", type=int, default=8192)
    ap.add_argument("--prefill", default="128,1024")
    ap.add_argument("--reuse", help="directory of finished *_quantized.tflite to reuse (and to fill)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    recipe = recipe_json(os.path.join(a.out_dir, "recipe_wi4b32.json"))
    patch_export_quantization(a.reuse)
    from litert_torch.generative.export_hf import export as export_lib

    t0 = time.time()
    export_lib.export(
        model=a.src,
        output_dir=a.out_dir,
        task="text_generation",
        prefill_lengths=[int(x) for x in a.prefill.split(",")],
        cache_length=a.cache_length,
        quantization_recipe=recipe,
        externalize_embedder=True,
        single_token_embedder=True,
        export_vision_encoder=False,
        export_audio_encoder=False,
        bundle_litert_lm=True,
        use_v2=True,
        delete_in_memory_params=True,
    )
    calls = REPORT["calls"]
    REPORT.update(
        src=a.src,
        cache_length=a.cache_length,
        prefill=a.prefill,
        seconds=round(time.time() - t0),
        quantized_tensors=len(calls),
        blocks=sum(c["blocks"] for c in calls),
        off_grid=sum(c["off_grid"] for c in calls),
    )
    with open(os.path.join(a.out_dir, "build_report.json"), "w") as f:
        json.dump(REPORT, f, indent=1)
    print(json.dumps({k: v for k, v in REPORT.items() if k != "calls"}), file=sys.stderr)


if __name__ == "__main__":
    main()
