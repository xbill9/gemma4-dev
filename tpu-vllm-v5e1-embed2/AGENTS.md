# AGENTS.md

Guidance for Codex and other coding agents working in this directory. **`CLAUDE.md` is authoritative**; read
it first. This file only summarizes.

This rig serves **`google/embeddinggemma-2`** (EmbeddingGemma 2, a bidirectional Gemma 4-derived embedding
model) as `/v1/embeddings` through vLLM on a Cloud TPU v5e-1 Flex-start Queued Resource. It is
**unprovisioned and unmeasured as of 2026-10-07**.

- It serves only on a **patched image**. The boot script pulls the pinned
  `vllm/vllm-tpu:nightly-20261007-865ce9b-5ab4e44` digest, applies `patches/` in `patches/ORDER` to
  `tpu_inference` (GQA support and a sliding-window off-by-one in the encoder-only flash kernel), and serves
  `vllm-tpu-embed2:patched`.
- The serve flags are `--runner pooling --convert embed`, and both are needed on TPU. `MAX_NUM_BATCHED_TOKENS` must be
  ≥ `MAX_MODEL_LEN` (8,192), because bidirectional pooling cannot chunk prefill.
- Clients must prepend the model's task prefixes (`EMBED_PROMPTS` in `server.py`).
- `tpu.env` is the source of truth. Tests are `python test_agent.py` (unittest), lint is `make lint`, and
  `make tools` regenerates `GemmaTools.md`.
- Never hardcode the endpoint, never use `shell=True`, and never destroy a queued resource unless asked.
