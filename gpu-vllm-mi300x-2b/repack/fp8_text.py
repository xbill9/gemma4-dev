"""Build a text-only Gemma 4 FP8 (W8A8, dynamic per-token activations) checkpoint.

    python3 tools/fp8_text.py build  SRC_DIR TEXT_CONFIG TEXT_INDEX OUT_DIR
    python3 tools/fp8_text.py build-on SRC_DIR BASE_DIR OUT_DIR
    python3 tools/fp8_text.py verify SRC_DIR OUT_DIR

    Add --fnuz to any of them for E4M3FNUZ, the fp8 of AMD CDNA 3 (MI300X, gfx942).

build-on keeps a W4A16 base (e.g. an `-emb4` build with int4 embeddings and lm_head)
as it is except for its Linear layers: every `weight_packed` module that is not an
embedding or lm_head becomes FP8 from SRC_DIR, and the base's config gets an FP8
group_0 with its other config groups (the int4 embeddings) unchanged.

SRC_DIR       Google's `-qat-q4_0-unquantized` checkpoint (bf16 QAT weights).
TEXT_CONFIG   config.json of the matching xbill9 `-ct-text` W4A16 repack: the
              text-only `Gemma4ForCausalLM` config this build reuses.
TEXT_INDEX    that repack's model.safetensors.index.json: its `weight_packed`
              modules are the Linear layers quantized here, and its other tensors
              are the bf16 ones copied.

Each quantized Linear gets an FP8 E4M3 `weight` and a float32 per-output-channel
`weight_scale` = max|row| / 448, so each row's largest value maps to E4M3's
largest finite value. Activations are quantized per token at run time
(compressed-tensors `float-quantized`, the FP8 twin of an int8 W8A8 build), which
vLLM runs on Ada's FP8 tensor cores. Everything else is copied byte for byte,
including the embeddings: vLLM has no FP8 embedding method.

--fnuz stores `float8_e4m3fnuz` (safetensors `F8_E4M3FNUZ`) with `weight_scale` =
max|row| / 240, that format's largest finite value. It has no infinities, no negative
zero and one NaN (0x80), and the same bits are half the value of E4M3FN. vLLM
allocates every fp8 weight as E4M3FN and casts what it loads into it, exactly for
values up to 240, then on gfx942 reinterprets the bits as FNUZ and doubles the scale;
so this build loads on stock vLLM, its 2**-10 subnormals round to 2**-9 on the way.

Mixture-of-experts (26B-A4B): the W4A16 repack splits each fused expert bank into
per-expert `experts.{i}.{gate,up,down}_proj` modules, so those are the modules
quantized here, each sliced out of the source's fused `experts.gate_up_proj` /
`experts.down_proj`. The routers stay bf16 on the ignore list the text config carries.

Needs numpy and ml_dtypes; the safetensors reader/writer are gemma4-dev's
(`gpu-vllm-t4-2b-w4a16/repack/repack_q4_0.py`), so no torch.
"""

import json
import os
import re
import shutil
import sys

import ml_dtypes
import numpy as np

sys.path.insert(0, os.path.expanduser("~/gemma4-dev/gpu-vllm-t4-2b-w4a16/repack"))
import repack_q4_0
from repack_q4_0 import bf16_to_f32, open_checkpoint, write_safetensors

repack_q4_0._DTYPES.setdefault("F8_E4M3", (np.uint8, 1))  # the reader predates FP8
repack_q4_0._DTYPES.setdefault("F8_E4M3FNUZ", (np.uint8, 1))

FNUZ = "--fnuz" in sys.argv
E4M3 = ml_dtypes.float8_e4m3fnuz if FNUZ else ml_dtypes.float8_e4m3fn
E4M3_TAG = "F8_E4M3FNUZ" if FNUZ else "F8_E4M3"
E4M3_MAX = float(ml_dtypes.finfo(E4M3).max)  # 448.0, or 240.0 with --fnuz
COPY_FILES = ("chat_template.jinja", "generation_config.json", "tokenizer.json", "tokenizer_config.json")

QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "quantization_status": "compressed",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "format": "float-quantized",
            "weights": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "channel",
                "group_size": None,
                "dynamic": False,
                "actorder": None,
                "block_structure": None,
                "observer": None,
                "observer_kwargs": {},
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "token",
                "group_size": None,
                "dynamic": True,
                "actorder": None,
                "block_structure": None,
                "observer": None,
                "observer_kwargs": {},
            },
            "output_activations": None,
        }
    },
    "ignore": ["lm_head"],
    "kv_cache_scheme": None,
    "sparsity_config": {},
}


def quantize_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float32 [out, in] -> (E4M3 [out, in], float32 scale [out, 1])."""
    amax = np.abs(w).max(axis=1, keepdims=True)
    scale = np.where(amax > 0, amax / E4M3_MAX, 1.0).astype(np.float32)
    return (w / scale).astype(E4M3), scale


EXPERT = re.compile(r"^(?P<base>.+\.experts\.)(?P<i>\d+)\.(?P<proj>gate|up|down)_proj$")


def _fused(ck, m: str):
    """(fused source tensor name, expert index, rows) for a per-expert module, else None."""
    e = EXPERT.match(m)
    if not e:
        return None
    name = e["base"] + ("down_proj" if e["proj"] == "down" else "gate_up_proj")
    if name not in ck:
        return None
    if e["proj"] == "down":
        return name, int(e["i"]), slice(None)
    f = ck[name].header[name]["shape"][1] // 2
    return name, int(e["i"]), slice(0, f) if e["proj"] == "gate" else slice(f, 2 * f)


def has_src(ck, m: str) -> bool:
    return m + ".weight" in ck or _fused(ck, m) is not None


def src_weight(ck, m: str) -> np.ndarray:
    """float32 [out, in] source weight of module m, sliced out of a fused expert bank if need be."""
    if m + ".weight" in ck:
        return bf16_to_f32(np.asarray(ck[m + ".weight"].raw(m + ".weight")))
    name, i, rows = _fused(ck, m)
    return bf16_to_f32(np.ascontiguousarray(ck[name].raw(name)[i, rows]))


def plan(text_index: str) -> tuple[list[str], list[str]]:
    names = json.load(open(text_index))["weight_map"]
    quantized = sorted(n[: -len(".weight_packed")] for n in names if n.endswith(".weight_packed"))
    q_suffixes = (".weight_packed", ".weight_scale", ".weight_shape", ".weight_zero_point")
    copied = sorted(n for n in names if not n.endswith(q_suffixes))
    return quantized, copied


def build(src: str, text_config: str, text_index: str, out: str) -> None:
    ck = open_checkpoint(src)
    quantized, copied = plan(text_index)
    missing = [m + ".weight" for m in quantized if not has_src(ck, m)] + [n for n in copied if n not in ck]
    if missing:
        raise SystemExit(f"{len(missing)} tensors not in {src}, e.g. {missing[:3]}")
    os.makedirs(out, exist_ok=True)
    tensors = []
    for m in quantized:
        w = src_weight(ck, m)
        if w.ndim != 2:
            raise SystemExit(f"{m}: expected a 2-D Linear weight, got {w.shape}")
        q, s = quantize_rows(w)
        tensors.append((m + ".weight", E4M3_TAG, q.view(np.uint8)))
        tensors.append((m + ".weight_scale", "F32", s))
    for n in copied:
        r = ck[n]
        tensors.append((n, r.header[n]["dtype"], np.asarray(r.raw(n))))
    write_safetensors(os.path.join(out, "model.safetensors"), tensors)
    cfg = json.load(open(text_config))
    ignore = (cfg.get("quantization_config") or {}).get("ignore") or []
    cfg["quantization_config"] = dict(QUANT_CONFIG, ignore=["lm_head"] + [i for i in ignore if i != "lm_head"])
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    for name in COPY_FILES:
        if os.path.exists(os.path.join(src, name)):
            shutil.copy2(os.path.join(src, name), os.path.join(out, name))
    print(f"wrote {out}: {len(quantized)} FP8 Linear modules, {len(copied)} tensors copied")


EMBED_MODULES = ("embed_tokens", "embed_tokens_per_layer", "lm_head")


def build_on(src: str, base: str, out: str) -> None:
    ck, bk = open_checkpoint(src), open_checkpoint(base)
    names = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
    packed = sorted(n[: -len(".weight_packed")] for n in names if n.endswith(".weight_packed"))
    linear = [m for m in packed if m.rsplit(".", 1)[-1] not in EMBED_MODULES]
    q_suffixes = (".weight_packed", ".weight_scale", ".weight_shape", ".weight_zero_point")
    lin = set(linear)
    keep = sorted(n for n in names if not (n.endswith(q_suffixes) and n.rsplit(".", 1)[0] in lin))
    missing = [m + ".weight" for m in linear if not has_src(ck, m)]
    if missing:
        raise SystemExit(f"{len(missing)} Linear weights not in {src}, e.g. {missing[:3]}")
    os.makedirs(out, exist_ok=True)
    tensors = []
    for m in linear:
        q, sc = quantize_rows(src_weight(ck, m))
        tensors.append((m + ".weight", E4M3_TAG, q.view(np.uint8)))
        tensors.append((m + ".weight_scale", "F32", sc))
    for n in keep:
        tensors.append((n, bk[n].header[n]["dtype"], np.asarray(bk[n].raw(n))))
    write_safetensors(os.path.join(out, "model.safetensors"), tensors)
    cfg = json.load(open(os.path.join(base, "config.json")))
    q = cfg["quantization_config"]
    q["config_groups"]["group_0"] = QUANT_CONFIG["config_groups"]["group_0"]
    q["ignore"] = [i for i in q.get("ignore", []) if i != "lm_head"]
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    for name in COPY_FILES:
        if os.path.exists(os.path.join(base, name)):
            shutil.copy2(os.path.join(base, name), os.path.join(out, name))
    print(f"wrote {out}: {len(linear)} FP8 Linear modules, {len(keep)} tensors kept from {base}")


def verify(src: str, out: str) -> dict:
    ck, fp8 = open_checkpoint(src), open_checkpoint(out)
    rep = {
        "modules": 0,
        "values": 0,
        "max_rel_err": 0.0,
        "sum_sq_err": 0.0,
        "sum_sq": 0.0,
        "copied": 0,
        "copied_identical": 0,
    }
    for n, r in fp8.items():
        if n.endswith(".weight_scale"):
            continue
        dtype = r.header[n]["dtype"]
        if dtype in ("F8_E4M3", "F8_E4M3FNUZ"):
            dt = ml_dtypes.float8_e4m3fnuz if dtype == "F8_E4M3FNUZ" else ml_dtypes.float8_e4m3fn
            m = n[: -len(".weight")]
            w = src_weight(ck, m)
            q = np.asarray(r.raw(n)).view(dt).astype(np.float32)
            s = np.asarray(fp8[m + ".weight_scale"].raw(m + ".weight_scale"))
            d = q * s
            amax = np.abs(w).max(axis=1, keepdims=True)
            rel = np.abs(d - w) / np.where(amax > 0, amax, 1.0)
            rep["modules"] += 1
            rep["values"] += w.size
            rep["max_rel_err"] = max(rep["max_rel_err"], float(rel.max()))
            rep["sum_sq_err"] += float(((d - w) ** 2).sum())
            rep["sum_sq"] += float((w**2).sum())
        else:
            if n not in ck:  # e.g. an int4-packed embedding kept from a build-on base
                continue
            rep["copied"] += 1
            rep["copied_identical"] += bool(np.array_equal(np.asarray(r.raw(n)), np.asarray(ck[n].raw(n))))
    rep["relative_rms_error"] = (rep["sum_sq_err"] / rep["sum_sq"]) ** 0.5
    for k in ("sum_sq_err", "sum_sq"):
        rep.pop(k)
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    sys.argv = [a for a in sys.argv if a != "--fnuz"]
    if sys.argv[1] == "build":
        build(*sys.argv[2:6])
    elif sys.argv[1] == "build-on":
        build_on(*sys.argv[2:5])
    elif sys.argv[1] == "verify":
        json.dump(
            verify(sys.argv[2], sys.argv[3]),
            open(os.path.join(sys.argv[3], "verify_report.json"), "w"),
            indent=1,
        )
    else:
        sys.exit(__doc__)
