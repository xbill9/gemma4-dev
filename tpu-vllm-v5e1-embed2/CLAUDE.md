# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file MCP server (`server.py`, MCPServer) that acts as a devops agent for serving **EmbeddingGemma 2**
(`google/embeddinggemma-2`) with vLLM as an **embedding** endpoint (`/v1/embeddings`) on a Google Cloud TPU
v5e-1 Flex-start Queued Resource. Its tools shell out to `gcloud` and talk HTTP to vLLM on port 8000.

Forked 2026-10-07 from `tpu-vllm-v5e1-2b-w8a8`, which had the patched-image machinery this rig needs. Slot 4
of the name is `embed2` (the model family and version, not a parameter size); `@NAMING.md` records it.

**STATUS 2026-10-07: unprovisioned and unmeasured.** Nothing here has booted. Every serving value is from
source reading and arithmetic, and every claim below about what the TPU stack does is a reading of
upstream source at the pinned commits, not an observation. The first boot is the experiment.

**It is not an A/B twin of anything.** Every other rig here serves a generative checkpoint; a number from
this rig must never be differenced against one of theirs. It answers a different request (`/v1/embeddings`,
no decode) and is measured in requests/s and end-to-end latency, not output tok/s.

## The model

Facts from the checkpoint's `config.json`, `config_sentence_transformers.json`, `1_Pooling/config.json` and
Google's launch posts (2026-10-06). Model-level facts belong in `@MODELS.md` if they are needed elsewhere.

- Architecture `EmbeddingGemma2Model` (`model_type: embedding_gemma2`), a Gemma 4-derived **bidirectional**
  encoder. One 1.49 GB bf16 `model.safetensors` (1,488,915,288 bytes) holds every modality.
- **Text encoder**: 24 layers, hidden 512, MLP 2048, 4 query heads, vocabulary 262,144, per-layer
  embeddings (PLE) like E2B. Pattern is 5 sliding : 1 full. Sliding layers: head_dim 256, 2 KV heads,
  window 512. Full-attention layers (5, 11, 17, 23): head_dim 512, **1 KV head**. RoPE θ 10k sliding / 1M full.
- **Output**: mean pooling (prompt included) then L2 normalize, 768 dimensions. Matryoshka-trained:
  512 / 256 / 128 are the recommended truncations.
- **Context**: 8,192 tokens, shared across modalities.
- **Towers**: vision ~170M (16 layers, hidden 768) and audio ~300M (12-layer conformer). Text alone is ~270M
  parameters. Text + vision + audio is ~740M.
- **dtype**: bf16 native. Google's guide says **float16 is incompatible**; use bf16 or fp32. v5e computes bf16
  natively, so this rig does not have the T4's problem.
- **Task prefixes are part of the input.** `"task: search result | query: "` for queries and
  `"title: none | text: "` for documents. The full table is `EMBED_PROMPTS` in `server.py`, copied from
  `config_sentence_transformers.json`. vLLM does not apply them; the client must. An unprefixed input is a
  different and worse embedding.
- Text cosines run high, around 0.69 for unrelated sentences per Google's guide. Compare rankings, not
  absolute scores.
- Not gated on Hugging Face. The boot script still fetches `hf-token` like the siblings do.

## Support status: why there is a patched image

Read this before changing the image or the patches. Sources are linked so the claims can be re-checked
against later commits.

1. **vLLM support is one day old.** `EmbeddingGemma2Model` landed in vllm-project/vllm@02b83919aa
   (PR #60254, 2026-10-06 17:43Z) with a follow-up mypy fix (bb87d227d4). It is in no release; `v0.31.0` and
   everything before it lack it. On TPU, the first image carrying it is
   **`vllm/vllm-tpu:nightly-20261007-865ce9b-5ab4e44`** (vLLM `5ab4e44` is 12 commits past 02b83919aa;
   tpu_inference `865ce9b`). `server.py` pins it by digest,
   `sha256:914a081a6e003ad77360f63650ffd6f4b4fc6d4e2be73119bae05cbf7c6acfa6`. The earlier nightly that day
   (`68f06a9-fa423b6`) is 57 commits *behind* the support. **Do not move to `:nightly` or a release tag
   without re-checking.** The patches below are written against this digest's tpu_inference.
2. **tpu_inference has no JAX definition for it.** `_MODEL_REGISTRY` in
   `tpu_inference/models/common/model_loader.py` lists no `EmbeddingGemma2Model`, so the model runs through the
   **torchax ("vllm") path**: vLLM's own PyTorch module lowered to XLA. `VLLM_ENV` sets
   `MODEL_IMPL_TYPE=vllm` to skip the failed JAX lookup and its warning. Pooling on that path is real.
   `VllmModelWrapper.build_pooler_func` pools on **CPU** after a device-to-host copy. CI covers it only with
   `Qwen/Qwen3-Embedding-8B` (`.buildkite/models/Qwen_Qwen3-Embedding-8B.yml`, `soft_fail: true`), a
   *causal* decoder.
3. **Bidirectional attention exists on TPU, but only for MHA.** EmbeddingGemma 2's attention is
   `AttentionType.ENCODER_ONLY`. tpu_inference added that in PR #2753 (2026-06-30) for BERT and ModernBERT. Those
   are multi-head models, and the kernel shows it: `flash_attention()` in
   `tpu_inference/kernels/flash_attention/kernel.py` raises `ValueError: Head count mismatch` whenever query
   heads ≠ KV heads. EmbeddingGemma 2 is GQA (4 over 2, and 4 over 1). **Unpatched, the first forward would
   fail.** `patches/gqa.diff` repeats K/V across each query group in `encoder_only_flash_attention`.
4. **The TPU sliding-window mask is one token wider than vLLM's.** `embedding_gemma2.py` passes
   `sliding_window = 512 + 1`, because vLLM's encoder-only convention is |q−k| ≤ W−1 and HF's is ≤ W. The CUDA
   flash-attn backend implements that as `window_size=(W−1, W−1)` (`vllm/v1/attention/backends/flash_attn.py`).
   tpu_inference masks `abs(row − col) <= sliding_window`, which is ≤ 513 here. `patches/window.diff`
   makes it `<`. This is a parity fix, not a crash fix. It also changes ModernBERT on this image, which is
   correct by the same argument but unverified. Drop it from `patches/ORDER` if a parity run says otherwise.
5. **`--convert embed` is required on TPU even though the model is natively a pooling model.** vLLM resolves
   a native pooling architecture to `convert_type="none"`, and `TPUModelRunner.get_supported_tasks()` returns
   exactly `(convert_type,)`. So without the flag the engine advertises task `"none"` and `/v1/embeddings`
   never exists. With it, `as_embedding_model()` returns the class unchanged (`is_pooling_model(cls)`), so
   nothing else moves. On GPU the flag is unnecessary.
6. **The Matryoshka `dimensions` API field will not work.** vLLM honours it only when the HF config has
   `matryoshka_dimensions` or `is_matryoshka`, and this checkpoint's `config.json` has neither. `embed_text`
   and `rank_documents` truncate and renormalize client-side instead, the same as sentence-transformers'
   `truncate_dim` + `normalize_embeddings`.

Things that pass source reading but are **unknown until boot**: whether the Pallas flash kernel's default
block sizes fit VMEM at head_dim 512; whether EmbeddingGemma 2's PLE lowers cleanly through torchax (the
Gemma 4 torchax PLE fix, tpu_inference PR #3406, touched `Gemma4` classes, not this one); compile time for the
token buckets up to 8,192; and parity against the sentence-transformers reference. **Parity was validated
upstream on A100 only.**

## Serving parameters

All in `tpu.env`, read by `server.py`'s `_vllm_serve_flags()`, which the startup script and
`manage_vllm_docker` share:

```
vllm serve google/embeddinggemma-2 --runner pooling --convert embed --max-model-len 8192 \
  --tensor-parallel-size 1 --max_num_batched_tokens 8192 --gpu-memory-utilization 0.80 \
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'
```

- **`MAX_NUM_BATCHED_TOKENS` ≥ `MAX_MODEL_LEN` is required, not tuning.** vLLM disables chunked prefill for
  bidirectional and mean-pooled models (`ModelConfig.is_chunked_prefill_supported`), so a request must fit
  in one batch. A test enforces the inequality.
- **No KV cache.** `KVCacheManager.get_kv_cache_spec()` skips encoder-only layers, so there is nothing for
  `--gpu-memory-utilization` to size. What it bounds is headroom. The encoder-only kernel materializes a
  dense sliding-window bias of `[heads, tokens, tokens]`, which is ~512 MiB in bf16 at 8,192 tokens and 4
  heads, rebuilt on every sliding layer. Text weights are ~0.5 GiB. This is arithmetic. If long inputs OOM,
  this bias is the first suspect, and lowering `MAX_MODEL_LEN` is the lever.
- **Text only by default.** A 0 limit for a modality makes vLLM skip building that tower (`_mark_tower_model`)
  and its weights are ignored at load. Image/video/audio would need `LIMIT_MM_PER_PROMPT` raised, and the towers
  through torchax are untested. Audio additionally needs the image's `vllm[audio]` extras, which have not
  been checked.
- No `--enable-auto-tool-choice` / `--tool-call-parser` / `--reasoning-parser`: those are generative-rig flags
  and the tests assert they stay out.

## Commands

```
make install    # pip install -r requirements.txt
make run        # python server.py (stdio MCP server)
make test       # python test_agent.py — unittest, NOT pytest
make lint       # ruff check . && ruff format --check . && mypy .
make format     # apply ruff formatting and autofixes
make tools      # regenerate GemmaTools.md from the @mcp.tool() decorators
make startup-script   # render the boot script (embeds patches/) to startup.rendered.sh
make run-container    # print the docker run line, built from server.py
make query PROMPT="..."   # embed one query against the hand-provisioned tpu-vm path
```

There is no `make benchmark`. The parent's `benchmarking_suite.py`, sweep, plot and compare scripts measured
generation and were dropped. `run_vllm_benchmark` drives `vllm bench serve --backend openai-embeddings
--endpoint /v1/embeddings` on the VM and maps the result to a schema-1.1 sweep point. `output_len` is 0 there,
and e2el latency travels under `raw` because the schema has no field for it.

## Tools that changed from the parent

- `verify_model_health` embeds a prefixed query, a relevant document and an irrelevant one. It checks
  dimension 768, unit norm, and that the relevant document ranks higher. It does not use an absolute
  threshold (see the 0.69 note above).
- `embed_text(text, prompt_name, dimensions)` and `rank_documents(query, documents, dimensions)` replace
  `query_queued_gemma4` / `query_queued_gemma4_with_stats`.
- `analyze_cloud_logging` returns the errors raw. An embedding model cannot summarize them.
- The Ollama tools and `find_gpu` were removed. They had no business in a TPU rig.

## Gotchas carried from the parent (still true here)

**`startup_script_template.sh` is consumed by `str.format()`.** Placeholders are `{project_id}`, `{zone}`,
`{model_name}`, `{hf_secret_id}`, `{tensor_parallel_size}`, `{max_model_len}`, `{max_num_batched_tokens}`,
`{limit_mm_per_prompt}`, `{gpu_memory_utilization}`, `{vllm_base_image}`, `{vllm_serve_image}`,
`{vllm_env_flags}`, `{patches_b64}`. Any other literal `{` or `}` breaks the deploy at format time. Escape as
`{{` / `}}`.

**The startup script fetches the HF token itself.** Never add a `{hf_token}` placeholder. Tracing stays off
across the token section.

**Patches live in `patches/`, applied in `patches/ORDER`,** embedded in the boot script as a base64 tar.gz,
applied with `patch -p1 --forward` to the image's `tpu_inference` and committed as `vllm-tpu-embed2:patched`.
A patch that fails to apply stops the boot. Diffs are rooted at the package's parent directory
(`a/tpu_inference/...`).

**`create_tpu_queued_resource` is non-destructive. `manage_queued_resource` deletes every other Queued Resource
in the zone.** Keep that split. **Don't destroy a queued resource unless asked.** Flex-start capacity can take up to 2 hours
to come back.

**Flex-start `v5litepod-1` is only accepted in `us-west4-a`** (verified 2026-08-04 on the siblings).
`tpu_zones_status.md` is mutable state that `find_tpu` rewrites. It was seeded from the parent's table
because this rig requests the same accelerator type, and this rig has made no attempts of its own.

**Resource names derive from the rig directory.** `RIG_NAME` = `tpu-vllm-v5e1-embed2` names the MCP server,
the default Queued Resource id (node `<id>-node`) and the Makefile's `SERVICE_NAME`. Registration is
`mcp_config.json` and `.agents/mcp_config.json` (committed examples) plus `make mcp-config`, which writes the
gitignored `.mcp.json`. This rig family has no `.claude-plugin/`, `.codex/` or skill directory. Renaming the
directory orphans anything provisioned.

**Endpoint discovery is dynamic.** `_discover_vllm_node()` probes `/v1/models`, which an embedding server
answers like any other. Never hardcode an IP.

**`gcloud compute tpus tpu-vm ssh` does not work from the dev sandbox** (it crashes with
`ConnectionResetError`), which breaks the SSH-based tools (`manage_vllm_docker`, `get_vllm_docker_logs`,
`get_tpu_system_logs`, `run_vllm_benchmark`). Plain `ssh -i ~/.ssh/google_compute_engine <user>@<ip>` and
HTTP to `:8000` work.

## Style and tests

Ruff (`py313`, line-length 120, rules `E,F,B,I`) and non-strict mypy, per `pyproject.toml`. A PostToolUse
hook formats edited `.py` files. Every subprocess goes through `run_command()`, never `shell=True`. Tools are
`async def` returning markdown with emoji status prefixes.

`test_agent.py` mocks `mcp` and the Google Cloud clients before importing `server`. Keep it offline. It
covers the pooling flags, the batch ≥ context rule, text-only default, patch ORDER, client-side Matryoshka
truncation, the embedding sweep-point mapping, and the new tools' prefixes and ranking.

## Git

The git root is the monorepo (`~/gemma4-dev`). Commit to `main` per the root `CLAUDE.md`, and read
`git status` first. Measurements go under `benchmarks/runs/<date>-<what>-v5e1/`; there are none yet.
`AGENTS.md` and `GEMINI.md` are short pointers to this file. Where they disagree, this file is correct.
