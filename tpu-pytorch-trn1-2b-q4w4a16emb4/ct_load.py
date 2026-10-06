"""Load a compressed-tensors Gemma-4 text checkpoint into a bf16 model on the HOST.

    from ct_load import is_compressed, load_text_model
    if is_compressed(path):
        model, generation_config = load_text_model(path)

Why this exists. NeuronCore-v2 has no int4, int8 or fp8 matmul that the
`torch_neuronx` trace path can reach (HARDWARE.md, inf2; the 2026-07-27 int8 run
in tpu-pytorch-inf2-2b): an in-graph dequantize is either constant-folded back to
bf16 by the compiler or materialised through HBM every step. So the repack is
expanded to bf16 here, once, on the CPU, and the traced graph is the same dense
bf16 graph the reference checkpoint produces. What the encoding changes is the
VALUES the graph holds, which is what the suite measures, and the download.

The one place the encoding survives past load is the vocabulary tables of an
`emb4` checkpoint. `embed_tokens` and `embed_tokens_per_layer` are gathered on
the host (torch_generate.py, "WHY THE HOST DOES THE EMBEDDING LOOKUP"), never on
the device, so they can stay packed int4 and only the gathered rows are unpacked.
That is 1.5 GB of host RAM where bf16 tables take 5.5 GB.

Formats handled, all symmetric, as the repacks' config.json declares them:

  pack-quantized   int4 packed eight to an int32, one scale per group of 32 inputs
  int-quantized    int8, one scale per output channel (W8A8: the A8 half is a
                   runtime activation quantization that this engine does not
                   perform, so W8A8 runs here as int8 weights, bf16 activations)
  float-quantized  fp8 e4m3, one scale per output channel (same note as W8A8)

The int4 unpack is compressed-tensors' own `unpack_from_int32`, so the packing
order is the library's definition, not a reimplementation of it. The packed
embedding's row unpack is checked against that function at load.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("ct_load")

# Checkpoint tensors that belong to towers a text engine never runs.
_SKIP_PREFIXES = ("model.vision_tower.", "model.audio_tower.", "model.embed_vision.",
                  "model.embed_audio.")
_TABLES = ("embed_tokens", "embed_tokens_per_layer")


def local_path(source: str) -> str:
    if os.path.isdir(source):
        return source
    from huggingface_hub import snapshot_download

    return snapshot_download(source)


def read_config(path: str) -> dict[str, Any]:
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)


def quantization_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    return cfg.get("quantization_config") or (cfg.get("text_config") or {}).get(
        "quantization_config"
    )


def is_compressed(source: str) -> bool:
    qc = quantization_config(read_config(local_path(source)))
    return bool(qc) and qc.get("quant_method") == "compressed-tensors"


def _rename(key: str) -> str | None:
    """Checkpoint key -> Gemma4ForCausalLM key, or None to drop it."""
    if key.startswith(_SKIP_PREFIXES):
        return None
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    return key


def _iter_tensors(path: str):
    from safetensors import safe_open

    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    for name in files:
        with safe_open(os.path.join(path, name), framework="pt") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


def _group(path: str) -> dict[str, dict[str, Any]]:
    """{module key: {tensor suffix: tensor}} for every kept tensor."""
    groups: dict[str, dict[str, Any]] = {}
    for key, tensor in _iter_tensors(path):
        new = _rename(key)
        if new is None:
            continue
        module, _, leaf = new.rpartition(".")
        groups.setdefault(module, {})[leaf] = tensor
    return groups


def unpack_int4(packed, shape):
    """int32-packed int4 -> int8 values in [-8, 7], via the library's own unpack."""
    import torch

    try:  # compressed-tensors >= 0.15 moved it
        from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    except ImportError:
        from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
            unpack_from_int32,
        )

    return unpack_from_int32(packed, 4, torch.Size([int(s) for s in shape]))


def dequantize(parts: dict[str, Any], dtype):
    """One module's stored tensors -> its dense weight in `dtype`."""
    import torch

    scale = parts.get("weight_scale")
    if "weight_zero_point" in parts and bool((parts["weight_zero_point"] != 0).any()):
        raise ValueError("asymmetric checkpoint: a non-zero weight_zero_point is not handled")
    if "weight_packed" in parts:
        out, inp = (int(s) for s in parts["weight_shape"])
        q = unpack_int4(parts["weight_packed"], (out, inp)).to(torch.float32)
        groups = scale.shape[1]
        w = q.view(out, groups, inp // groups) * scale.to(torch.float32).unsqueeze(-1)
        return w.reshape(out, inp).to(dtype)
    weight = parts["weight"]
    if scale is None:
        return weight.to(dtype)
    return (weight.to(torch.float32) * scale.to(torch.float32)).to(dtype)


def _make_host_embedding():
    import torch

    class HostEmbedding(torch.nn.Module):
        """A scaled embedding gathered on the CPU, dense or packed int4.

        Reproduces Gemma4TextScaledWordEmbedding exactly, including its rounding:
        the scale is cast to the table dtype before the multiply, so sqrt(1536)
        becomes 39.25 in bf16, as the reference model computes it.
        """

        def __init__(self, scale: float, dtype, dense=None, packed=None, group_scale=None,
                     dim: int | None = None):
            super().__init__()
            self.out_dtype = dtype
            self.scale = torch.tensor(scale).to(dtype)
            self.dense = dense
            self.packed = packed
            self.group_scale = group_scale
            self.dim = dim if dense is None else dense.shape[1]
            self.shifts = torch.arange(8, dtype=torch.int32) * 4

        @property
        def weight(self):
            # The engine never reads a table whole; this exists so an accidental
            # caller fails loudly on a packed table rather than misreading it.
            if self.dense is None:
                raise AttributeError("packed int4 table: gather rows with forward()")
            return self.dense

        def unpack_rows(self, rows_packed, rows_scale):
            torch_ = torch
            nib = (rows_packed.unsqueeze(-1) >> self.shifts) & 0xF
            q = nib.reshape(*rows_packed.shape[:-1], self.dim).to(torch_.float32) - 8.0
            groups = rows_scale.shape[-1]
            q = q.view(*q.shape[:-1], groups, self.dim // groups)
            return (q * rows_scale.to(torch_.float32).unsqueeze(-1)).reshape(
                *rows_packed.shape[:-1], self.dim
            ).to(self.out_dtype)

        def forward(self, ids):
            if self.dense is not None:
                rows = torch.nn.functional.embedding(ids, self.dense)
            else:
                rows = self.unpack_rows(self.packed[ids], self.group_scale[ids])
            return rows * self.scale

    return HostEmbedding


def load_text_model(source: str, dtype=None):
    """Build Gemma4ForCausalLM from a compressed-tensors checkpoint.

    Returns (model, generation_config). The model is on the CPU in bf16 except
    for packed vocabulary tables, which stay int4 until gathered.
    """
    import torch
    from transformers import AutoConfig, Gemma4ForCausalLM, GenerationConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    dtype = dtype or torch.bfloat16
    path = local_path(source)
    config = AutoConfig.from_pretrained(path)
    text_cfg = config.get_text_config()
    for cfg in {id(config): config, id(text_cfg): text_cfg}.values():
        if hasattr(cfg, "quantization_config"):
            delattr(cfg, "quantization_config")
    text_cfg.torch_dtype = dtype

    # Built on the meta device: nothing is allocated until a real tensor is
    # assigned, so a packed table never exists in dense form on this host.
    with torch.device("meta"):
        model = Gemma4ForCausalLM(text_cfg)
    model.eval()

    groups = _group(path)
    HostEmbedding = _make_host_embedding()
    lang = model.model
    stats = {"int4": 0, "int8": 0, "fp8": 0, "dense": 0}

    for table in _TABLES:
        parts = groups.pop(f"model.{table}", None)
        if parts is None:
            continue
        orig = getattr(lang, table)
        scale = float(orig.scalar_embed_scale)
        if "weight_packed" in parts:
            vocab, dim = (int(s) for s in parts["weight_shape"])
            emb = HostEmbedding(scale, dtype, packed=parts["weight_packed"],
                                group_scale=parts["weight_scale"], dim=dim)
            # The row unpack is ours; the whole-table unpack is the library's.
            # They must agree, or every token embeds wrong without an error.
            probe = parts["weight_packed"][:4]
            want = unpack_int4(probe, (4, dim)).to(torch.float32)
            got = ((probe.unsqueeze(-1) >> emb.shifts) & 0xF).reshape(4, dim).float() - 8.0
            if not torch.equal(want, got):
                raise RuntimeError(f"{table}: packed-row unpack disagrees with compressed-tensors")
            stats["int4"] += 1
            logger.info("%s kept packed int4: %d x %d", table, vocab, dim)
        else:
            emb = HostEmbedding(scale, dtype, dense=dequantize(parts, dtype))
            stats["dense"] += 1
        setattr(lang, table, emb)

    loaded = set()
    for module_key, parts in groups.items():
        module = model.get_submodule(module_key)
        if "weight_packed" in parts or "weight_scale" in parts:
            w = dequantize(parts, dtype)
            kind = ("int4" if "weight_packed" in parts
                    else "fp8" if parts["weight"].dtype == torch.float8_e4m3fn else "int8")
            stats[kind] += 1
            module._parameters["weight"] = torch.nn.Parameter(w, requires_grad=False)
            loaded.add(f"{module_key}.weight")
            continue
        for leaf, tensor in parts.items():
            t = tensor.to(dtype) if tensor.is_floating_point() else tensor
            if leaf in module._parameters:
                module._parameters[leaf] = torch.nn.Parameter(t, requires_grad=False)
            elif leaf in module._buffers:
                module._buffers[leaf] = t
            else:
                raise KeyError(f"checkpoint tensor {module_key}.{leaf} has no home in the model")
            loaded.add(f"{module_key}.{leaf}")
            stats["dense"] += 1

    # lm_head: tied checkpoints omit it and share the dense input table.
    if "lm_head.weight" not in loaded:
        table = lang.embed_tokens
        if table.dense is None:
            raise RuntimeError("lm_head missing and embed_tokens is packed: nothing to tie to")
        model.lm_head._parameters["weight"] = torch.nn.Parameter(table.dense, requires_grad=False)
        loaded.add("lm_head.weight")

    # Non-persistent buffers are never in a checkpoint; rebuild the only module
    # that owns any (the tables' scales are handled by HostEmbedding).
    lang.rotary_emb = Gemma4TextRotaryEmbedding(text_cfg)

    # Shared-KV layers carry k/v projections and norms that they never run;
    # Google's exports omit them. Fill with zeros so nothing stays on meta, and
    # refuse anything else that is missing.
    shared = {i for i, layer in enumerate(lang.layers) if layer.self_attn.is_kv_shared_layer}
    for name, param in list(model.named_parameters()):
        if param.device.type != "meta":
            continue
        parts = name.split(".")
        layer_ok = (len(parts) > 3 and parts[1] == "layers" and int(parts[2]) in shared
                    and parts[4] in ("k_proj", "v_proj", "k_norm"))
        if not layer_ok:
            raise RuntimeError(f"{name} is missing from the checkpoint")
        owner = model.get_submodule(name.rsplit(".", 1)[0])
        owner._parameters[parts[-1]] = torch.nn.Parameter(
            torch.zeros(param.shape, dtype=dtype), requires_grad=False
        )
    meta = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if meta:
        raise RuntimeError(f"buffers left on the meta device: {meta[:5]}")

    logger.info("dequantized %s from %s", stats, path)
    gen = GenerationConfig.from_pretrained(path)
    return model, gen
