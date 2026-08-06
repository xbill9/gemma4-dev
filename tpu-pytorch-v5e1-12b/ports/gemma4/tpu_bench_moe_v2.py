#!/usr/bin/env python3.12
"""KV-cached decode bench on TPU: ports/gemma4 MoE port + v2 FUSED expert
kernel (dequant-inside-matmul Pallas, fused_experts_kernel.py).

Identical to tpu_bench_moe.py except the experts are swapped to
W4A16ExpertsV2 (quant_experts_v2.py): decode-sized dispatch (T*K = batch*8)
runs the fused Pallas kernel — packed int4 slices gathered in-graph, dequant
inside the matmul, no [T*K, out, in] bf16 weight temporaries — while
prefill-sized dispatch keeps the v1 in-graph-dequant path (bounded by
prefill_len; use 64 at batch=8, where the v1 42.13 GB temp OOM lived).

Usage: python3.12 tpu_bench_moe_v2.py [model_id] [batch] [steps_per_call] [prefill_len]
Run with LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536.
"""

import gc
import json
import sys
import time

MODEL = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
)
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 1
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 1
PREFILL_LEN_ARG = int(sys.argv[4]) if len(sys.argv) > 4 else 128
sys.argv = sys.argv[:1]  # imported modules must not see bench argv

import torch
import transformers
from huggingface_hub import hf_hub_download

import moe_model
import quant_experts_v2

PROMPT = "Explain in two sentences why TPUs are fast."
MAX_SEQ = 256      # KV cache length
PREFILL_LEN = PREFILL_LEN_ARG  # prompt padded to this (MXU-friendly static)
WARMUP_STEPS = 4
BENCH_STEPS = 64

_DROP_PREFIXES = (
    "model.vision_tower.", "model.audio_tower.", "model.embed_vision.",
    "model.embed_audio.", "model.multi_modal_projector.",
    "model.vision_embedder.", "model.audio_embedder.",
    "vision_tower.", "audio_tower.", "vision_embedder.", "audio_embedder.",
)


def to_port_name(hf_name: str):
    """HF Unified/ConditionalGeneration module path -> port path (or None)."""
    if hf_name.startswith(_DROP_PREFIXES):
        return None
    if hf_name.startswith("model.language_model."):
        return "model." + hf_name[len("model.language_model."):]
    if hf_name.startswith("language_model."):
        return "model." + hf_name[len("language_model."):]
    return hf_name


def mark_ram(stage: str):
    vals = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS", "VmHWM")):
                key, kb = line.split(":")
                vals[key] = int(kb.split()[0]) / 1e6  # kB -> GB
    print(f"MARKER RAM [{stage}]: VmRSS {vals.get('VmRSS', 0):.1f} GB "
          f"(peak {vals.get('VmHWM', 0):.1f} GB)", flush=True)


def model_bytes(model: torch.nn.Module) -> int:
    """Total unique param+buffer bytes (tied tensors counted once)."""
    seen = {}
    for t in list(model.parameters()) + list(model.buffers()):
        seen[t.data_ptr()] = t.numel() * t.element_size()
    return sum(seen.values())


def build_port(text_config: dict) -> moe_model.Gemma4MoEForCausalLM:
    """MoE port in bf16 with Linear/Embedding random-init skipped (every
    param is loaded from the HF state dict — asserted after the transfer)."""
    cfg = moe_model.Gemma4MoEConfig.from_text_config(text_config)
    saved = (torch.nn.Linear.reset_parameters,
             torch.nn.Embedding.reset_parameters)
    torch.nn.Linear.reset_parameters = lambda self: None
    torch.nn.Embedding.reset_parameters = lambda self: None
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        port = moe_model.Gemma4MoEForCausalLM(cfg).eval()
    finally:
        torch.set_default_dtype(old_dtype)
        (torch.nn.Linear.reset_parameters,
         torch.nn.Embedding.reset_parameters) = saved
    return port


def load_hf_model(model_id: str):
    """bf16 CPU load; robust to the ForConditionalGeneration architecture."""
    last = None
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                     "AutoModel"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            hf = cls.from_pretrained(model_id, dtype=torch.bfloat16).eval()
            print(f"loaded with {cls_name}: {type(hf).__name__}", flush=True)
            return hf
        except Exception as e:  # class resolution differs across 5.x
            print(f"{cls_name} failed: {type(e).__name__}: {e}", flush=True)
            last = e
    raise last


def main() -> int:
    device = torch.device("tpu")
    assert quant_experts_v2.fused_op_available(), \
        "backend Pallas bridge missing — this bench would silently run v1"
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    config = json.load(open(hf_hub_download(MODEL, "config.json")))
    text_config = config.get("text_config", config)

    mark_ram("start")
    t0 = time.monotonic()
    port = build_port(text_config)
    n_layers = len(port.model.layers)
    print(f"MARKER port built: {n_layers} layers, "
          f"{port.config.num_experts} experts top-{port.config.top_k_experts}, "
          f"bf16 {model_bytes(port) / 1e9:.1f} GB "
          f"in {time.monotonic() - t0:.0f}s", flush=True)
    mark_ram("port built")

    print(f"Loading {MODEL} (bf16, CPU)...", flush=True)
    t0 = time.monotonic()
    hf = load_hf_model(MODEL)
    print(f"MARKER hf loaded in {time.monotonic() - t0:.0f}s", flush=True)
    mark_ram("hf loaded")

    # ---- transfer: remap HF keys -> port keys, keep only keys the port has.
    t0 = time.monotonic()
    port_sd_keys = set(port.state_dict().keys())
    sd = {}
    dropped = []
    for key, value in hf.state_dict().items():
        pkey = to_port_name(key)
        if pkey is not None and pkey in port_sd_keys:
            sd[pkey] = value
        else:
            dropped.append(key)
    missing, unexpected = port.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:8]}"
    bad_missing = [k for k in missing if k != "lm_head.weight"]  # tied
    assert not bad_missing, f"missing port keys (would stay random): {bad_missing[:8]}"
    non_text = [k for k in dropped if k.startswith(_DROP_PREFIXES)]
    lost = [k for k in dropped if not k.startswith(_DROP_PREFIXES)]
    assert not lost, f"text keys with no port destination: {lost[:8]}"
    print(f"MARKER transfer: {len(sd)} tensors -> port, "
          f"{len(non_text)} non-text dropped, missing={missing} (tied) "
          f"in {time.monotonic() - t0:.0f}s", flush=True)

    del hf, sd
    gc.collect()
    mark_ram("hf deleted")

    # ---- DIY quantization: experts bf16 [E,2I,H]/[E,H,I] -> packed int4 g32,
    # V2 module (fused Pallas kernel on decode-sized dispatch).
    t0 = time.monotonic()
    for i, layer in enumerate(port.model.layers):
        assert isinstance(layer, moe_model.Gemma4MoEDecoderLayer), type(layer)
        assert isinstance(layer.experts, moe_model.Gemma4RefExperts)
        layer.experts = quant_experts_v2.W4A16ExpertsV2.from_hf(layer.experts)
        # fp32 group scales -> bf16 (README: 2.9 GB -> 1.4 GB; both the v1
        # in-graph path and the fused kernel take scales in any dtype).
        for bname in ("gate_up_scale", "down_scale"):
            buf = layer.experts._buffers[bname]
            layer.experts._buffers[bname] = buf.to(torch.bfloat16)
        gc.collect()  # free this layer's bf16 reference experts now
        if (i + 1) % 5 == 0 or i + 1 == n_layers:
            print(f"  quantized {i + 1}/{n_layers} layers "
                  f"({time.monotonic() - t0:.0f}s)", flush=True)

    expert_bytes = sum(
        b.numel() * b.element_size()
        for layer in port.model.layers for b in layer.experts.buffers()
    )
    total_bytes = model_bytes(port)
    print(f"MARKER quantized: {n_layers} layers swapped to W4A16ExpertsV2 "
          f"(fused decode <= {quant_experts_v2.FUSED_MAX_SLICES} slices, "
          f"v1 prefill); experts packed+scales {expert_bytes / 1e9:.2f} GB, "
          f"non-expert bf16 {(total_bytes - expert_bytes) / 1e9:.2f} GB, "
          f"model total {total_bytes / 1e9:.2f} GB (HBM weight footprint) "
          f"in {time.monotonic() - t0:.0f}s", flush=True)
    mark_ram("quantized")

    t0 = time.monotonic()
    port = port.to(device)
    print(f"MARKER to(tpu) in {time.monotonic() - t0:.0f}s", flush=True)
    mark_ram("on tpu")

    # ---- prompt, cache, compiled prefill/decode (same shape discipline as
    # tpu_bench_moe.py: static PREFILL_LEN, (1,) position tensors, one graph each).
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )["input_ids"]
    n_prompt = ids.shape[1]
    assert n_prompt < PREFILL_LEN <= MAX_SEQ
    pad_id = tokenizer.pad_token_id or 0
    padded = torch.full((BATCH, PREFILL_LEN), pad_id, dtype=torch.long)
    padded[:, :n_prompt] = ids[0]
    padded = padded.to(device)
    last_idx = torch.tensor([n_prompt - 1], dtype=torch.long).to(device)

    cache = port.new_kv_cache(BATCH, MAX_SEQ, device=device)
    kv_bytes = sum(t.numel() * t.element_size() for t in cache.k + cache.v)
    print(f"MARKER kv cache: {kv_bytes / 1e9:.2f} GB "
          f"(batch={BATCH}, MAX_SEQ={MAX_SEQ})", flush=True)

    def prefill_fn(ids2d, last_idx):
        hidden = port.model.prefill(ids2d, cache)           # (B, PRE, H)
        h = hidden.index_select(1, last_idx)                # (B, 1, H)
        logits = port._cap_logits(port.lm_head(h))[:, -1, :]
        return logits.argmax(-1, keepdim=True)              # (B, 1)

    def decode_fn(tok, pos):
        toks = []
        for _ in range(CHUNK):
            logits = port.decode_step(tok, pos, cache)      # (B, V)
            tok = logits.argmax(-1, keepdim=True)           # (B, 1)
            toks.append(tok)
            pos = pos + 1
        return torch.cat(toks, dim=1), tok, pos

    assert BENCH_STEPS % CHUNK == 0
    warmup_calls = max(1, WARMUP_STEPS // CHUNK)

    prefill_c = torch.compile(prefill_fn, backend="tpu", dynamic=False,
                              fullgraph=True)
    decode_c = torch.compile(decode_fn, backend="tpu", dynamic=False,
                             fullgraph=True)

    with torch.no_grad():
        t0 = time.monotonic()
        nxt = prefill_c(padded, last_idx)
        nxt[0].cpu()
        print(f"MARKER MOE V2 prefill (PREFILL_LEN={PREFILL_LEN}, "
              f"n_prompt={n_prompt}, incl compile) in "
              f"{time.monotonic() - t0:.1f}s", flush=True)
        t0 = time.monotonic()
        nxt = prefill_c(padded, last_idx)
        nxt[0].cpu()
        print(f"MARKER MOE V2 prefill exec (compiled) in "
              f"{1000 * (time.monotonic() - t0):.0f}ms", flush=True)
        generated = [nxt]
        pos = torch.tensor([n_prompt], dtype=torch.long).to(device)

        print(f"Warmup ({warmup_calls * CHUNK} steps, includes compile)...",
              flush=True)
        t0 = time.monotonic()
        for _ in range(warmup_calls):
            chunk_toks, nxt, pos = decode_c(nxt, pos)
            generated.append(chunk_toks)
        nxt[0].cpu()
        print(f"MARKER MOE V2 warmup done in {time.monotonic() - t0:.1f}s",
              flush=True)

        t0 = time.monotonic()
        for _ in range(BENCH_STEPS // CHUNK):
            chunk_toks, nxt, pos = decode_c(nxt, pos)
            generated.append(chunk_toks)
        nxt[0].cpu()
        elapsed = time.monotonic() - t0

        text = tokenizer.decode(
            torch.cat(generated, dim=1)[0].cpu(), skip_special_tokens=True)
    mark_ram("after bench")
    print("--- output " + "-" * 49)
    print(text.strip())
    print("-" * 60)
    print(f"MARKER MOE V2 CACHED decode (MAX_SEQ={MAX_SEQ}, batch={BATCH}, "
          f"steps_per_call={CHUNK}, static KV cache, FUSED int4 experts): "
          f"{BENCH_STEPS} steps in {elapsed:.2f}s = "
          f"{BENCH_STEPS / elapsed:.1f} tok/s/stream, "
          f"{BATCH * BENCH_STEPS / elapsed:.1f} tok/s aggregate "
          f"({1000 * elapsed / BENCH_STEPS:.1f} ms/step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
