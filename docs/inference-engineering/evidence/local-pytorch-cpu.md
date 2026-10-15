# First light — `local-pytorch-cpu-2b`, 2026-09-04

`transformers` 5.16.1 + `torch` 2.13.0+cu130, `google/gemma-4-E2B-it` in bf16 on the host CPU
(Intel i7-10750H, 6 physical cores, AVX2 only, no AVX512). No accelerator used.
Harness: `bench_first_light.py` at the rig root, which reads `tpu.env`.
Machine-readable: `first_light.json`.

**This rig was a verbatim copy of `gpu-jax-g4dn-2b` until the morning of this run** — it contained
no PyTorch at all. Retargeted and served the same day.

## Result

| | run 1 | run 2 |
| :--- | ---: | ---: |
| load (warm page cache) | 3.3 s | **3.1 s** |
| generate 128 tok | 27.9 s | **26.4 s** |
| **tok/s** | 4.587 | **4.853** |
| RSS after load | 1.20 GB | 1.20 GB |
| **peak RSS** | **5.71 GB** | **5.71 GB** |

5.8% spread on the rate, and the peak RSS is **identical to three digits** across both.

**One end-to-end figure.** `generate()` exposes no prefill/decode split, so this covers both and is
**not comparable** to a sibling's decode-only rate. The prompt is 14 tokens, so it is decode-dominated
in practice — but that is an argument, not a measurement.

## The finding: mmap gives transformers the lazy PLE for free

```
safetensors file                10.25 GB
minus embed_tokens_per_layer    -4.70 GB   (one BF16 tensor; @MODELS.md, per-tensor measurement)
= everything else                5.55 GB
MEASURED peak RSS                5.71 GB   delta +0.16 GB
```

**56% of the file is touched, and the untouched 44% is the PLE.** Safetensors mmaps the checkpoint
and the PLE is an indexed gather, so only rows for tokens actually seen are ever paged in.
transformers writes no code for this; the OS page cache does it.

**Third independent confirmation of the same mechanism**, and together they explain this checkpoint's
behaviour on every stack tried on this host:

| stack | treatment of the 4.7 GB PLE | outcome |
| :--- | :--- | :--- |
| llama.cpp | `TENSOR_READ_LAZY` + mmap, never on device | 1612 MiB on a 4 GiB GPU, 73.75 tok/s |
| **transformers** | **mmap + gather, only touched rows resident** | **5.71 GB of 10.25, serves** |
| vLLM | copied into `VocabParallelEmbedding` at `__init__` | **OOM: asks for exactly 4.38 GiB** |

The one that fails is the one that materialises it eagerly.

## Why it is slow, and why that is the right trade here

bf16 on an AVX2-only CPU is emulated — there is no AVX512-BF16 datapath on Comet Lake. **bf16 is a
memory decision, not a speed one**: fp32 would be 20.5 GB and does not fit this host at all, while
bf16's 10.25 GB fits inside the 11.3–12.0 GB available. 4.85 tok/s is what that costs.

## Gemma 4 answered directly here, unlike on the sibling rigs

At 128 output tokens the llama.cpp and Ollama sweeps measured **pure thinking** — `content_chunks: 0`
in every cell. The same prompt here returned a real answer with no thinking block:

```
"Here are three generations of Google's Tensor Processing Units (TPUs):
 1. **TPU v1:** ... 2. **TPU v2:** ..."
```

The difference is the chat template: this rig applies the checkpoint's own `chat_template.jinja`,
Ollama substitutes its Go renderer (`renderer=gemma4 parser=gemma4`), and llama.cpp was driven
through a server with reasoning enabled. **Do not generalise from one prompt** — `MAX_NEW_TOKENS`
stays at 1024.

## Two API details, one wasted run each

- **`apply_chat_template` returns a `BatchEncoding` in transformers 5.x**, not a tensor. `ids.shape`
  raises `KeyError: 'shape'` then a bare `AttributeError`, which reads as a broken tokenizer and is
  not. Use `return_dict=True`, pass `**enc` to `generate` so the attention mask travels with it.
- **`device_map` requires `accelerate`**, which transformers does not depend on. Without it,
  `from_pretrained` raises `ValueError` at load time.

## Caveats

- **Sizing must come from the file, not from RSS.** RSS right after load is 1.20 GB — mmap has
  materialised almost nothing. A capacity check reading RSS would clear a host that cannot hold the
  model.
- **The GPU in this box was not used and is not usable here.** 4096 MiB against 10.25 GB, and
  transformers has no way to keep the PLE off-device; `device_map="auto"` would measure PCIe.
- **Live desktop.** MemAvailable moved from 10.83 to 12.03 GB across the session. Re-check capacity
  immediately before a load rather than trusting a figure from earlier.
- **One prompt, one shape, two repeats.** This is first light, not a sweep. No context or
  concurrency axis exists yet.
