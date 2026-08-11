#!/usr/bin/env python3.12
"""Full-model decode bench: gemma-4 QAT w4a16-ct with int8-STORED weights.

Same swap harness as w4a16_fused_model_bench.py, but each Linear's int4
weights are dequantized once at load into int8 (q-8, unscaled) laid out in
the kernel's plane-permuted column order, so the in-kernel work is just
convert+scale+dot — no nibble unpacking. 2x the HBM of packed int4, still
2x under bf16. Kernel-level shootout (w4a16_kernel_debug.py) priced this
~25% faster than the fused int4 kernel; this bench prices it at model level.

Usage: python3.12 w4a16_int8_model_bench.py [model_id] [batch]
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
import transformers
from jax.experimental import pallas as pl
import importlib
import os

# The PyTorch TPU backend package is not publicly distributable, so its name is
# not committed here: set TPU_BACKEND_MODULE (e.g. in /etc/environment on the VM)
# to the installed backend package. Imported dynamically for the Pallas bridge.
_BACKEND = os.getenv("TPU_BACKEND_MODULE")
if not _BACKEND:
    raise RuntimeError(
        "TPU_BACKEND_MODULE is not set — export it to the installed PyTorch TPU "
        "backend package name so the Pallas bridge can be imported."
    )
pallas = importlib.import_module(f"{_BACKEND}._internal.pallas")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-12B-it-qat-w4a16-ct"
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 1
PROMPT = "Explain in two sentences why TPUs are fast."
GROUP = 32
SEQ = 256
WARMUP_STEPS = 4
BENCH_STEPS = 64


def _ck_for(k: int) -> int | None:
    for cand in range(1024, 0, -128):
        if k % cand == 0:
            return cand
    return k if k <= 1024 else None


def w4a16_int8_matmul_jax(x: jax.Array, w8: jax.Array, scale_rep4: jax.Array) -> jax.Array:
    """x: [S, K] bf16; w8: [O, K] int8 in plane-permuted column order;
    scale_rep4: [O, K//8] bf16 (group scales repeat_interleave(4))."""
    seq, k = x.shape
    out_f = w8.shape[0]
    blk = 256 if out_f % 256 == 0 else (128 if out_f % 128 == 0 else out_f)
    ck = _ck_for(k)
    ck8 = ck // 8

    j = np.arange(k)
    c, r = j // ck, j % ck
    perm = c * ck + 8 * (r % ck8) + r // ck8
    x = x[:, perm]

    def kernel(x_ref, w8_ref, scale_ref, out_ref):
        x_all = x_ref[...]
        wq = w8_ref[...]
        s4 = scale_ref[...]
        acc = jnp.zeros((seq, blk), jnp.float32)
        for ci in range(k // ck):
            w = wq[:, ci * ck:(ci + 1) * ck].astype(s4.dtype)
            sr = s4[:, ci * ck8:(ci + 1) * ck8]
            w = w * jnp.concatenate([sr] * 8, axis=1)
            acc += jax.lax.dot_general(
                x_all[:, ci * ck:(ci + 1) * ck], w,
                (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32,
            )
        out_ref[...] = acc.astype(jnp.bfloat16)

    return pl.pallas_call(
        kernel,
        grid=(out_f // blk,),
        in_specs=[
            pl.BlockSpec((seq, k), lambda i: (0, 0)),
            pl.BlockSpec((blk, k), lambda i: (i, 0)),
            pl.BlockSpec((blk, k // 8), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((seq, blk), lambda i: (0, i)),
        out_shape=jax.ShapeDtypeStruct((seq, out_f), jnp.bfloat16),
    )(x, w8, scale_rep4)


w8a16_matmul = pallas.jax_op("w8a16::matmul", w4a16_int8_matmul_jax)


def int8_plane_permuted(packed: torch.Tensor, out_f: int, in_f: int) -> torch.Tensor:
    """Unpack int4-in-int32 to int8 (q-8, unscaled) with columns in the
    kernel's plane order: within chunk c, column i*ck8 + m holds original
    weight column c*ck + 8m + i. Done once on CPU at swap time."""
    ck = _ck_for(in_f)
    ck8 = ck // 8
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((packed.unsqueeze(-1) >> shifts) & 0xF) - 8          # [O, K/8, 8] (m, i)
    chunks = []
    for c in range(in_f // ck):
        blk = q[:, c * ck8:(c + 1) * ck8, :]                  # [O, ck8, 8]
        chunks.append(blk.permute(0, 2, 1).reshape(out_f, ck))
    return torch.cat(chunks, dim=1).to(torch.int8).contiguous()


class Int8W4A16Linear(torch.nn.Module):
    def __init__(self, packed, scale, out_f, in_f, bias):
        super().__init__()
        self.out_f, self.in_f = out_f, in_f
        self.register_buffer("w8", int8_plane_permuted(packed, out_f, in_f))
        self.register_buffer("scale_rep4", scale.repeat_interleave(4, dim=1).contiguous())
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x):
        y = w8a16_matmul(x.reshape(-1, self.in_f), self.w8, self.scale_rep4)
        y = y.reshape(*x.shape[:-1], self.out_f)
        return y + self.bias if self.bias is not None else y


class W4A16Linear(torch.nn.Module):
    """In-graph dequant fallback for non-tileable layers (identical math)."""

    def __init__(self, packed, scale, out_f, in_f, bias):
        super().__init__()
        self.out_f, self.in_f = out_f, in_f
        self.register_buffer("packed", packed)
        self.register_buffer("scale", scale)
        self.register_buffer("shifts", torch.arange(0, 32, 4, dtype=torch.int32))
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x):
        w = (self.packed.unsqueeze(-1) >> self.shifts) & 0xF
        w = w.reshape(self.out_f, self.in_f) - 8
        w = w.to(self.scale.dtype).reshape(self.out_f, -1, GROUP) * self.scale.unsqueeze(-1)
        w = w.reshape(self.out_f, self.in_f).to(torch.bfloat16)
        y = x @ w.T
        return y + self.bias if self.bias is not None else y


def main() -> int:
    device = torch.device("tpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    print(f"Loading {MODEL} packed (run_compressed=True, CPU)...")
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

    t0 = time.monotonic()
    int8_n = fallback = 0
    int8_bytes = 0
    for name, mod in list(model.named_modules()):
        if not hasattr(mod, "weight_packed"):
            continue
        out_f, in_f = (int(v) for v in mod.weight_shape)
        packed = mod.weight_packed.data
        scale = mod.weight_scale.data
        if scale.shape[0] != out_f:
            scale = scale.T.contiguous()
        bias = mod.bias.data if getattr(mod, "bias", None) is not None else None
        tileable = out_f % 128 == 0 and in_f % 8 == 0 and _ck_for(in_f) is not None
        cls = Int8W4A16Linear if tileable else W4A16Linear
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        new = cls(packed, scale, out_f, in_f, bias)
        setattr(parent, name.rsplit(".", 1)[-1], new)
        int8_n += tileable
        fallback += not tileable
        if tileable:
            int8_bytes += new.w8.numel() + new.scale_rep4.numel() * 2
    print(f"MARKER swapped: {int8_n} int8, {fallback} fallback (in-graph int4); "
          f"int8+scale = {int8_bytes / 1e9:.2f} GB; converted in {time.monotonic() - t0:.1f}s")

    model = model.to(device).eval()

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )["input_ids"]
    n_prompt = ids.shape[1]
    pad_id = tokenizer.pad_token_id or 0
    tokens = torch.full((BATCH, SEQ), pad_id, dtype=torch.long)
    tokens[:, :n_prompt] = ids[0]
    tokens = tokens.to(device)
    mask = torch.zeros((BATCH, SEQ), dtype=torch.long)
    mask[:, :n_prompt] = 1
    mask = mask.to(device)
    one = torch.ones((BATCH, 1), dtype=torch.long).to(device)
    pos = torch.tensor([n_prompt], dtype=torch.long).to(device)

    def step(tokens, mask, last_idx):
        logits = model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
        return logits.index_select(1, last_idx).argmax(-1)

    step_c = torch.compile(step, backend="tpu", dynamic=False)

    def decode_one():
        nonlocal pos, tokens, mask
        nxt = step_c(tokens, mask, pos - 1)
        tokens = tokens.index_copy(1, pos, nxt)
        mask = mask.index_copy(1, pos, one)
        pos = pos + 1

    print(f"Warmup ({WARMUP_STEPS} steps, includes compile)...")
    t0 = time.monotonic()
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            decode_one()
        tokens[0, :1].cpu()
    print(f"MARKER warmup done in {time.monotonic() - t0:.1f}s")

    t0 = time.monotonic()
    with torch.no_grad():
        for _ in range(BENCH_STEPS):
            decode_one()
        tokens[0, :1].cpu()
    elapsed = time.monotonic() - t0

    n_total = n_prompt + WARMUP_STEPS + BENCH_STEPS
    text = tokenizer.decode(tokens[0, n_prompt:n_total].cpu(), skip_special_tokens=True)
    print("--- output " + "-" * 49)
    print(text.strip())
    print("-" * 60)
    print(f"MARKER INT8 w4a16 compiled decode (SEQ={SEQ}, batch={BATCH}, no KV cache): "
          f"{BENCH_STEPS} steps in {elapsed:.2f}s = {BENCH_STEPS / elapsed:.1f} tok/s/stream, "
          f"{BATCH * BENCH_STEPS / elapsed:.1f} tok/s aggregate "
          f"({1000 * elapsed / BENCH_STEPS:.1f} ms/step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
