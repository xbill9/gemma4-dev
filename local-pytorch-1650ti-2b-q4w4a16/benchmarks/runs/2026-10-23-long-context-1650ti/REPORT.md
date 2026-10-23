# 2026-10-23 — long context on the 4 GB card (in progress)

Same host, card and stack as `../2026-10-23-close-gap-1650ti/`. Goal: the model's full 131,072-token context on the
GTX 1650 Ti beside the int4 model, with prompt reads and decode measured at depth against llama.cpp on the same card.

#### Why it should fit

E2B keeps a full-length K/V cache in only 3 layers (positions 4, 9, 14; one KV head of width 512, so 2 KiB per token
per layer). The 12 other layers that own K/V are sliding-window layers capped at 512 positions, and the last 20 layers
reuse earlier layers' K/V. At 131,072 tokens the cache is 3 × 131,072 × 2 KiB = 768 MiB (arithmetic). The 90 tok/s
build peaked at 2,793 MiB with a 384-token cache, against 3,732 MiB visible to PyTorch.

#### Attention kernels (`w4attn_long.py`)

- **Decode:** the keys are split into slices; each Triton program reads its slice of K and V once for all eight query
  heads and keeps an online softmax; a second kernel combines the slices. Full layers read slots 0..pos (slot =
  position); sliding layers read a 512-slot ring and mask by the position stored with each slot. All indices come from
  a device tensor, so the kernel sits inside the CUDA graph.
- **Prompt chunks:** PyTorch's memory-efficient SDPA with K/V expanded across heads (a view; 12 MiB extra at 16k keys
  rather than a copy), a bottom-right causal mask for full layers and an explicit window mask for sliding layers.

Against an fp32 reference (`attn_long_check.jsonl`): decode relative error 0–3.3e-4 at positions 0 to 131,071 (full)
and 10 to 5,000 (sliding); prompt chunks 3.1–4.3e-4.

Decode GPU time per full-attention layer, CUDA-graph timed (`dec_sweep.jsonl`, `dec_sweep2.jsonl`):

| Kernel variant | 32,768 tokens held | 131,072 tokens held |
| --- | ---: | ---: |
| fp16 `tl.dot`, best launch shape | 2.85 ms (23.6 GB/s) | 11.35 ms (23.7 GB/s) |
| broadcast multiply-sum over the 8 real heads, 128 slices of 8 keys, 4 warps | 1.13–1.16 ms (58–59 GB/s) | 4.60 ms (58.4 GB/s) |

The fp16 `tl.dot` variant stays at about 24 GB/s at every launch shape tried: this card has no tensor cores, and the
dot also pads 8 heads to 16 rows. Every fp32 `tl.dot` configuration failed to compile (shared memory). Seven layers
attend over the full-length cache each token (3 own it, 4 share it), so at 131,072 tokens decode attention costs about
32 ms per token with the broadcast kernel (arithmetic from the per-layer time).

#### Model script (`e2b_long.py`)

Keeps the K/V cache outside transformers and replaces Gemma 4's attention forward: full layers own 131,072 slots,
sliding layers a 512-slot ring; prompts are read in 512-token chunks, and a sliding layer's chunk K/V is held aside
until the whole forward pass is done, because the KV-shared layers still need the ring's older positions. Not yet
run. The benchmark fills the context with numbered records and asks for the code of one placed a quarter of the way in.

#### Not yet measured

The model at depth (prompt-read speed, decode speed, peak memory, recall), and llama.cpp at the same depths
(`llama-bench -p 512 -n 64 -d 0,4096,16384,32768,65536,131072`, running).
