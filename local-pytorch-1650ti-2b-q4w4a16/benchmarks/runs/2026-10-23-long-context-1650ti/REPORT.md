# 2026-10-23 — long context on the 4 GB card (in progress)

Same host, card and stack as `../2026-10-23-close-gap-1650ti/`. The model's full 131,072-token context on the GTX 1650 Ti
beside the int4 model, with prompt reads, decode and recall measured at depth against llama.cpp on the same card.

#### Memory

E2B keeps a full-length K/V cache in only 3 layers (one KV head of width 512: 2 KiB per token per layer). The 12 other
layers that own K/V are sliding-window layers capped at 512 positions; the last 20 layers reuse earlier layers' K/V.

| | MiB |
| --- | ---: |
| Model, every weight and vocabulary table int4 (one copy of `embed_tokens`/`lm_head`) | 2,483 |
| K/V cache, 131,072 tokens | 774 |
| Allocated after load | 3,257 |
| Peak while reading a 4,081-token prompt and decoding | 3,470 |
| Visible to PyTorch | 3,732 |

`-text-emb4` stores `embed_tokens` and the untied `lm_head` as the same int4 levels and scales twice
(`embed_lmhead_identical.txt`); the model was trained tied, so one copy serves both and saves 216 MiB. With both copies
loaded, the first prompt read ran out of memory.

#### Results

`long-short.jsonl` (prompt reads multiplied in fp16) and `prefill-profile-fp32.jsonl` (in fp32, the default now).
Decode is 63 greedy tokens after the prompt; prompt tok/s is the whole prompt read from position 0.

| Context | Prompt read, fp16 GEMM | Prompt read, fp32 GEMM | Decode | Recall |
| ---: | ---: | ---: | ---: | --- |
| 61 | 79.4 tok/s | 160.5 tok/s | 87.8–87.9 tok/s | ✅ |
| 4,081 | 97.7 tok/s | 226.5 tok/s | 80.4–80.5 tok/s | ✅ |
| 32,761 | 79.4 tok/s | | 51.5 tok/s | ❌ (319550 for 713468) |

llama.cpp `7182ef7` on the same card and model (`llamacpp-depth.json`, `llama-bench -p 512 -n 64`, 2 repeats):

| Context held | Prompt read (512 tokens) | Decode |
| ---: | ---: | ---: |
| 0 | 360.76 | 82.09 |
| 4,096 | 279.75 | 76.65 |
| 16,384 | 182.50 | 66.59 |
| 32,768 | 124.74 | 56.80 |
| 65,536 | 76.36 | 43.80 |
| 131,072 | 42.69 | 30.21 |

llama-bench reads 512 tokens at the stated depth; this run's prompt figure reads the whole prompt from position 0, so
the prompt columns are not the same measurement.

#### Recall

The benchmark fills the context with numbered records and asks for the code of the record a quarter of the way in.
`dump_prompts.py` rebuilds the exact prompts and `recall_llamacpp.py` sends them to llama-server with thinking off (its
default turns Gemma 4 thinking on): same token counts (61, 4,081, 32,761) and the same answers from both engines,
including the same wrong code at 32,761 tokens, 319550, which belongs to record 723 of 1,091; the target was record 272
(`llamacpp-recall.jsonl`).

#### Kernels (`w4attn_long.py`)

- **Decode:** keys split into 64 slices; each Triton program reads its slice of K and V once for all eight query heads,
  one head at a time over 2-D [keys, D] tiles, with an online softmax; a second kernel combines the slices. Full layers
  read slots 0..pos; sliding layers a 512-slot ring masked by the position stored per slot. Indices come from a device
  tensor, so it runs inside the CUDA graph. Per full-attention layer (`attn_long_check.jsonl`): 0.19 ms at 4,096 tokens
  held, 1.20 ms at 32,768, 3.74 ms at 131,072. fp16 `tl.dot` stayed at about 24 GB/s at every launch shape on this card,
  which has no tensor cores (`dec_sweep*.jsonl`).
- **Prompt chunks:** PyTorch's memory-efficient SDPA, K/V expanded across heads as a view (12 MiB extra at 16k keys), a
  bottom-right causal mask for full layers and an explicit window mask for sliding layers; 256-token chunks.
- Against an fp32 reference: relative error 0–3.3e-4 for decode, 3.1–4.3e-4 for prompt chunks.

#### Prompt-read matmuls

Reading a 2,041-token prompt with fp16 multiplies took 20.3 s of GPU time, 18.5 s of it in cuBLAS `s884gemm` kernels
(`prefill-profile.jsonl`). On this card fp16 GEMMs run at 0.34–0.42 TFLOP/s and fp32 at 1.57–1.88 (0.66–0.93 for the
deep down-projection) (`gemm_check.jsonl`). Unpacking the weights to fp32 for prompt reads: 8.3 s of GPU time for the
same prompt, 6.1 s of it in `sgemm` (`prefill-profile-fp32.jsonl`).

#### Not yet measured

32,768 and 131,000 tokens with fp32 prompt reads (running), 16k and 64k.
