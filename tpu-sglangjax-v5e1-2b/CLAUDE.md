# tpu-sglangjax-v5e1-2b

SGLang-JAX (`sgl-project/sglang-jax` at `b3e3e16b2`, main of 2026-10-20) serving `google/gemma-4-E2B-it` bf16 on one
TPU v5e chip, with `patches/gemma4-e2b.diff` (sha256 `d6ac1d7a2f8f6e55…`). No MCP server, skill or plugin: `tpu.env`, `flight/`,
`patches/`, measurements. Added 2026-11-08.

## Why a patch

Unpatched, sglang-jax serves the Gemma 4 31B and 26B A4B only, and E2B fails at model load
(`tpu-vllm-v5e1-2b-q4w4a16emb4/benchmarks/runs/2026-11-08-ladder-sglang-v5e1/`). The patch adds:

- **config**: E2B's `"num_global_key_value_heads": null` fell through `getattr` as `None`; fall back to the sliding count.
- **per-layer embeddings**: `embed_tokens_per_layer` (262144 x 35·256), `per_layer_model_projection`,
  `per_layer_projection_norm`, and per layer `per_layer_input_gate` / `per_layer_projection` /
  `post_per_layer_input_norm`, applied after the MLP residual and before `layer_scalar`, as in Hugging Face's
  `Gemma4TextModel` and `~/jax-dev/tpu-jax-v5e1-2b/ports/gemma4/jax_e_model.py`.
- **double-wide MLP** on the KV-shared layers (`use_double_wide_mlp`).
- **KV sharing, by copy**: layers 15–34 take this pass's K/V from the last non-shared layer of the same type and store
  them again in their own cache slot. Same attention values, no change to the memory pool or attention backend, at the
  cost of cache for 35 layers instead of 15 (43,008 bytes per token across the pool, from the server log).

## Serving flags that matter

- `--page-size 128`: at the default 1, the FlashAttention backend caps running requests at 131072 / pages per
  request = 8 at a 16,384-token context. sglang-jax's own Gemma 2 recipe says the same.
- `JAX_COMPILATION_CACHE_DIR`: the 512-wide full-attention Pallas kernel takes minutes to compile per shape on v5e
  (about 300 s per prefill bucket at first start). The cache key includes page size.
- `--watchdog-timeout 7200`: the default 300 s kills the scheduler during a lazy compile.
- No Gemma 4 tool-call parser in sglang-jax, so no BFCL.

## Running

`flight/` is the v5e-1 ladder's driver (`tpu-vllm-v5e1-2b-q4w4a16emb4/flight/`) cut to the `sglang` arm.
On a flex-start v5e-1 (`tpu.env`), copy `flight/`, `patches/` and `jev-tpu-v5e1/` to `/opt/ladder`, then
`sudo systemd-run --unit=ladder-sgl bash /opt/ladder/flight/run-sgl.sh <QR>`: setup, patch, `one+acc+sweep`, upload
to `gs://aisprint-491218-ladder-v5e1/<RUN>/<QR>/`, delete the queued resource whatever the outcome.
`flight/parity.py` compares the six greedy replies with the bf16 JAX engine's hashes from the ladder.

Compare with the ladder's `jax-bf16` and `jaxrust` arms (same bf16 checkpoint), not with the int4 `vllm` arm.
