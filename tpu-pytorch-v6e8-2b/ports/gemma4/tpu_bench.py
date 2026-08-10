#!/usr/bin/env python3.12
"""KV-cached decode bench on TPU: ports/gemma4 port + real w4a16-ct weights.

Loads the HF gemma-4 QAT w4a16-ct checkpoint PACKED (run_compressed=True) on
CPU, swaps every quantized Linear in the port for Int8W4A16Linear (the
plane-permuted int8 + w8a16 Pallas kernel from w4a16_int8_model_bench.py;
in-graph W4A16Linear fallback for non-tileable shapes), loads the remaining
bf16 params (embeddings, norms, layer_scalars) from the HF state dict, then
benches the static-KV-cache prefill()/decode_step() path compiled with
torch.compile(backend="tpu", dynamic=False).

Baselines to beat (no-KV-cache int8-fused full-seq decode, SEQ=256, batch=1):
12B: 44.3 tok/s; 31B: 11.4 tok/s.

Usage: python3.12 tpu_bench.py [model_id] [batch] [steps_per_call] [quant]

quant: "int8" (default) = plane-permuted int8 + w8a16 Pallas kernel
       (Int8W4A16Linear, 2x packed-int4 bytes, no in-kernel unpack);
       "int4" = packed int4 + fused unpack w4a16 Pallas kernel
       (FusedW4A16Linear, half the HBM traffic per step, unpack on VPU).

steps_per_call > 1 chains that many decode steps (argmax feeding the next
embed, pos advanced in-graph) into ONE compiled graph, amortizing the
per-invocation host dispatch that dominates a seq=1 step.
"""

import gc
import json
import sys
import time

MODEL = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-12B-it-qat-w4a16-ct"
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 1
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 1
QUANT = sys.argv[4] if len(sys.argv) > 4 else "int8"
assert QUANT in ("int8", "int4")
sys.argv = sys.argv[:1]  # the bench modules parse sys.argv at import

import torch
import transformers
from huggingface_hub import hf_hub_download

import model as port_model
import w4a16_int8_model_bench as kmod  # registers the w8a16 Pallas op

W4A16Linear = kmod.W4A16Linear
if QUANT == "int8":
    QuantLinear = kmod.Int8W4A16Linear
else:
    import w4a16_fused_model_bench as fmod  # registers the w4a16 Pallas op
    QuantLinear = fmod.FusedW4A16Linear
Int8W4A16Linear = kmod.Int8W4A16Linear

PROMPT = "Explain in two sentences why TPUs are fast."
MAX_SEQ = 256      # KV cache length
PREFILL_LEN = 128  # prompt padded to this (MXU/Pallas-friendly static shape)
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


def build_port(text_config: dict) -> port_model.Gemma4DenseForCausalLM:
    """Port model in bf16 with parameter random-init skipped (all params are
    either replaced by quantized modules or loaded from the HF state dict —
    asserted below), so construction is fast and allocation-light."""
    cfg = port_model.Gemma4DenseConfig.from_text_config(text_config)
    saved = (torch.nn.Linear.reset_parameters, torch.nn.Embedding.reset_parameters)
    torch.nn.Linear.reset_parameters = lambda self: None
    torch.nn.Embedding.reset_parameters = lambda self: None
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        port = port_model.Gemma4DenseForCausalLM(cfg).eval()
    finally:
        torch.set_default_dtype(old_dtype)
        torch.nn.Linear.reset_parameters, torch.nn.Embedding.reset_parameters = saved
    return port


def main() -> int:
    device = torch.device("tpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    text_config = json.load(open(hf_hub_download(MODEL, "config.json")))["text_config"]
    port = build_port(text_config)

    print(f"Loading {MODEL} packed (run_compressed=True, CPU)...")
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

    # Non-quantized params/buffers: remap HF keys -> port keys, keep only keys
    # the port has (quantized-Linear weight_* keys and vision/audio drop out).
    port_sd_keys = set(port.state_dict().keys())
    sd = {}
    for key, value in hf.state_dict().items():
        pkey = to_port_name(key)
        if pkey is not None and pkey in port_sd_keys:
            sd[pkey] = value

    # Swap every quantized HF Linear into the port.
    t0 = time.monotonic()
    port_modules = dict(port.named_modules())
    int8_n = fallback = skipped_nonport = 0
    int8_bytes = 0
    for name, mod in list(hf.named_modules()):
        if not hasattr(mod, "weight_packed"):
            continue
        pname = to_port_name(name)
        if pname is None or pname not in port_modules:
            skipped_nonport += 1
            continue
        old = port_modules[pname]
        out_f, in_f = (int(v) for v in mod.weight_shape)
        assert isinstance(old, torch.nn.Linear) and (
            old.out_features, old.in_features) == (out_f, in_f), (
            f"shape mismatch at {pname}: port {old} vs HF {out_f}x{in_f}")
        packed = mod.weight_packed.data
        scale = mod.weight_scale.data
        if scale.shape[0] != out_f:
            scale = scale.T.contiguous()
        bias = mod.bias.data if getattr(mod, "bias", None) is not None else None
        tileable = out_f % 128 == 0 and in_f % 8 == 0 and kmod._ck_for(in_f) is not None
        cls = QuantLinear if tileable else W4A16Linear
        new = cls(packed, scale, out_f, in_f, bias)
        parent = port.get_submodule(pname.rsplit(".", 1)[0]) if "." in pname else port
        setattr(parent, pname.rsplit(".", 1)[-1], new)
        int8_n += tileable
        fallback += not tileable
        if tileable:
            wbytes = new.w8.numel() if QUANT == "int8" else new.packed.numel() * 4
            int8_bytes += wbytes + new.scale_rep4.numel() * 2
        # Free the HF packed copy as we go.
        hf_parent = hf.get_submodule(name.rsplit(".", 1)[0]) if "." in name else hf
        setattr(hf_parent, name.rsplit(".", 1)[-1], torch.nn.Identity())
    print(f"MARKER swapped: {int8_n} {QUANT} kernel, {fallback} fallback (in-graph int4), "
          f"{skipped_nonport} skipped (non-text); weights+scale = {int8_bytes / 1e9:.2f} GB; "
          f"converted in {time.monotonic() - t0:.1f}s")
    del hf, port_modules
    gc.collect()

    missing, unexpected = port.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:8]}"
    bad_missing = [k for k in missing
                   if not k.endswith((".w8", ".scale_rep4", ".packed", ".scale", ".shifts"))
                   and k != "lm_head.weight"]  # tied to embed_tokens.weight
    assert not bad_missing, f"missing non-quant keys: {bad_missing[:8]}"
    del sd
    gc.collect()

    # Sanity: every port Linear was swapped or is intentionally bf16.
    n_int8 = n_fb = 0
    bf16_linears = []
    for name, m in port.named_modules():
        if isinstance(m, QuantLinear):
            n_int8 += 1
        elif isinstance(m, W4A16Linear):
            n_fb += 1
        elif isinstance(m, torch.nn.Linear):
            bf16_linears.append(name)
    assert bf16_linears == ["lm_head"], f"unswapped Linears: {bf16_linears}"
    print(f"MARKER port modules: {n_int8} {QuantLinear.__name__}, {n_fb} W4A16Linear, "
          f"bf16 Linears: {bf16_linears} (lm_head tied to embeddings, intentional)")

    port = port.to(device)

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )["input_ids"]
    n_prompt = ids.shape[1]
    assert n_prompt < PREFILL_LEN <= MAX_SEQ
    pad_id = tokenizer.pad_token_id or 0
    # Pad the prompt to a static PREFILL_LEN; take logits at the last REAL
    # position. Pad rows write garbage into cache rows n_prompt..PREFILL_LEN-1,
    # but each decode step index_copy_'s its own row before attending and the
    # decode masks hide all rows > pos, so garbage is overwritten/masked.
    padded = torch.full((BATCH, PREFILL_LEN), pad_id, dtype=torch.long)
    padded[:, :n_prompt] = ids[0]
    padded = padded.to(device)
    last_idx = torch.tensor([n_prompt - 1], dtype=torch.long).to(device)

    cache = port.new_kv_cache(BATCH, MAX_SEQ, device=device)

    def prefill_fn(ids2d, last_idx):
        hidden = port.model.prefill(ids2d, cache)           # (B, PRE, H)
        h = hidden.index_select(1, last_idx)                # (B, 1, H)
        logits = port._cap_logits(port.lm_head(h))[:, -1, :]
        return logits.argmax(-1, keepdim=True)              # (B, 1)

    def decode_fn(tok, pos):
        """CHUNK chained decode steps in one graph: argmax feeds the next
        embed, pos advances in-graph. Returns (all new tokens, last, pos)."""
        toks = []
        for _ in range(CHUNK):
            logits = port.decode_step(tok, pos, cache)      # (B, V)
            tok = logits.argmax(-1, keepdim=True)           # (B, 1)
            toks.append(tok)
            pos = pos + 1
        return torch.cat(toks, dim=1), tok, pos

    assert BENCH_STEPS % CHUNK == 0
    warmup_calls = max(1, WARMUP_STEPS // CHUNK)

    prefill_c = torch.compile(prefill_fn, backend="tpu", dynamic=False, fullgraph=True)
    decode_c = torch.compile(decode_fn, backend="tpu", dynamic=False, fullgraph=True)

    with torch.no_grad():
        t0 = time.monotonic()
        nxt = prefill_c(padded, last_idx)
        nxt[0].cpu()
        print(f"MARKER prefill (PREFILL_LEN={PREFILL_LEN}, n_prompt={n_prompt}, "
              f"incl compile) in {time.monotonic() - t0:.1f}s")
        # Same-content prefill again, now compiled: pure exec time. Rewrites
        # the same cache rows with identical values, so state is unchanged.
        t0 = time.monotonic()
        nxt = prefill_c(padded, last_idx)
        nxt[0].cpu()
        print(f"MARKER prefill exec (compiled) in {1000 * (time.monotonic() - t0):.0f}ms")
        generated = [nxt]
        pos = torch.tensor([n_prompt], dtype=torch.long).to(device)

        print(f"Warmup ({warmup_calls * CHUNK} steps, includes compile)...")
        t0 = time.monotonic()
        for _ in range(warmup_calls):
            chunk_toks, nxt, pos = decode_c(nxt, pos)
            generated.append(chunk_toks)
        nxt[0].cpu()
        print(f"MARKER warmup done in {time.monotonic() - t0:.1f}s")

        t0 = time.monotonic()
        for _ in range(BENCH_STEPS // CHUNK):
            chunk_toks, nxt, pos = decode_c(nxt, pos)
            generated.append(chunk_toks)
        nxt[0].cpu()
        elapsed = time.monotonic() - t0

        text = tokenizer.decode(
            torch.cat(generated, dim=1)[0].cpu(), skip_special_tokens=True)
    print("--- output " + "-" * 49)
    print(text.strip())
    print("-" * 60)
    print(f"MARKER CACHED decode (MAX_SEQ={MAX_SEQ}, batch={BATCH}, "
          f"steps_per_call={CHUNK}, static KV cache): "
          f"{BENCH_STEPS} steps in {elapsed:.2f}s = {BENCH_STEPS / elapsed:.1f} tok/s/stream, "
          f"{BATCH * BENCH_STEPS / elapsed:.1f} tok/s aggregate "
          f"({1000 * elapsed / BENCH_STEPS:.1f} ms/step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
