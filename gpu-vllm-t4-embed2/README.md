# gpu-vllm-t4-embed2

**EmbeddingGemma 2** (`google/embeddinggemma-2`) served by vLLM as an OpenAI-compatible
`/v1/embeddings` endpoint on one NVIDIA Tesla T4 already attached to a Compute Engine VM,
with an MCP server (`gpu-vllm-t4-embed2`) for capacity checks, the Turing kernel patch,
lifecycle and embedding tools.

EmbeddingGemma 2 is Google's Gemma 4-derived multimodal embedding model (released
2026-10-06): 768-d vectors with Matryoshka truncation to 512/256/128, 8,192-token
context, text + image + audio + video into one space. This rig serves text and images.

**Measured 2026-10-07** (`benchmarks/runs/2026-10-07-first-serve-t4/REPORT.md`):
float32, 1−cos ≤ 7.0e-9 against sentence-transformers; ~3.9k input tok/s at
~500-token inputs; 96 s to healthy text only.

## Quick start

```bash
make capacity       # free arithmetic: disks, VRAM, host RAM
make arch           # torch SM 7.5 + a vLLM/transformers that know the model
make patch          # encoder-attention clamp for Turing (required)
make verify-patch
make serve          # stop any generative T4 rig first: same GPU, same port
make status && make verify
make stop
```

```bash
curl -s localhost:8000/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model": "google/embeddinggemma-2",
  "input": ["task: search result | query: what is a TPU?",
            "title: none | text: A TPU is an accelerator for matrix math."],
  "dimensions": 256}'
```

**Add the task prefix yourself** — vLLM embeds exactly the string it is sent. Queries
take `task: search result | query: `, documents `title: none | text: ` (the full table
is in the model's `config_sentence_transformers.json`). Images go through `messages`
with an `image_url` part.

## Why it needs a patch

Google says float16 is incompatible with this model and the T4 has no bfloat16, so it
serves in float32 — and vLLM's encoder-only Triton attention kernel does not fit a
float32 tile in Turing's 64 KiB of shared memory. `patch_triton_encoder_turing.py`
clamps the tile and runs only the 512-wide attention layers in fp16 with fp32
accumulation. See `CLAUDE.md`.

## Install

Into this rig's own user base, never the shared `/opt1/pyuser` (that is the generative T4
rigs' vLLM 0.29). `make capacity` prints the exact command; it installs the pinned lock
in `benchmarks/runs/2026-10-07-first-serve-t4/evidence/requirements-lock.txt`.
