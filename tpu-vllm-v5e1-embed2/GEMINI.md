# Gemini Code Assistant Context

**`CLAUDE.md` is authoritative for this directory**; read it before changing anything.

This project is an MCP devops agent that serves **EmbeddingGemma 2** (`google/embeddinggemma-2`) as a vLLM
embedding endpoint (`/v1/embeddings`) on a Cloud TPU v5e-1 Flex-start Queued Resource. It runs on a patched
`vllm-tpu` image (see `patches/`). It is **unprovisioned and unmeasured as of 2026-10-07**.

The model embeds and does not generate. Tools that used to ask the self-hosted model to explain logs now return the
logs raw. Use `verify_model_health`, `embed_text` and `rank_documents` to exercise it. Discover the endpoint
with `get_vllm_endpoint`, and never hardcode an IP.

Register the MCP server under the key `tpu-vllm-v5e1-embed2` (see `mcp_config.json`, or run
`make mcp-config`).
