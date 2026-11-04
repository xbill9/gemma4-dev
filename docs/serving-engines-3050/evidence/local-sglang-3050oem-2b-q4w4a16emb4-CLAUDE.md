# CLAUDE.md — local-sglang-3050oem-2b-q4w4a16emb4

Gemma 4 E2B served by **SGLang 0.5.23** on the RTX 3050 OEM in the desktop `gpu`, from the same
checkpoint and revision as `local-vllm-3050oem-2b-q4w4a16emb4`
(`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` @ `db715f83`), so the pair is an engine A/B on
identical weights. Created 2026-11-04 for the 3050 serving-engine comparison. No MCP server, no skill,
no plugin: `tpu.env`, `serve.py`, a Makefile, measurements. Same card as the other `local-*-3050oem-*`
rigs; they do not run beside each other.

## Read first

- **`tpu.env` decides**, and `serve.py` turns it into the `sglang.launch_server` command (`make command`).
- **SGLang runs on the system python 3.14 though it ships wheels for 3.11–3.13 only.** `make install`
  installs the cp313 wheel retagged cp314 with `--no-deps`, then only the requirements that were missing
  (also `--no-deps`, plus IPython with its dependencies). Installed packages keep their versions over
  SGLang's pins: transformers 5.16.1 (pin 5.19.0), compressed-tensors 0.17.0 (0.18.0), openai 3.43.0
  (2.8.0), flashinfer-python 0.7.2.post1, xgrammar 0.2.8, numba 0.65.0, tokenizers 0.23.3. No existing
  package changed version (freeze before: `~/.cache/sglang-install/freeze-before-2026-11-04.txt`).
  Left out: deep-ep, deep-gemm, flash-mla, flash-attn-4, tilelang, torchaudio, torchcodec, nvshmem,
  modelscope.
- **Its five Rust extensions are cp313 builds** (`srt/rust_extensions/*`, `rust_tree_core`) and do not load
  on 3.14; they serve gRPC, the Rust server, some multimodal processors and the unified radix tree, none
  of which the HTTP path here uses.
- **Two patches to the installed SGLang** (`patches/`, applied by `make install`; originals restore with
  `patch -R`):
  - `text-config-per-layer.diff` (`srt/utils/hf_transformers/config.py`): the Gemma 4 attention shim ran
    only for multimodal configs, and this text-only checkpoint's config is `gemma4_text`, so model load
    raised `AmbiguousGlobalPerLayerAttributeError` on `head_dim` (sliding layers 256 wide, full 512).
    The heterogeneity code is the same in transformers 5.16.1 and 5.19, so the older transformers is not
    the cause.
  - `int4-embeddings.diff` (`srt/models/gemma4_causal.py`): SGLang 0.5.23 quantizes the int4 `lm_head`
    from compressed-tensors but not `embed_tokens` or `embed_tokens_per_layer`; it allocated them as bf16
    and left them uninitialized (6.93 GB of weights, no room for the cache). The patch keeps both tables
    packed on the card and unpacks only the gathered rows: weights 3.25 GB. A dense table is built only if
    something asks for `.weight` (the drafter's shared embedding), 768 MiB bf16.
- **Decode CUDA graphs only** (`DISABLE_PREFILL_CUDA_GRAPH=true`): prefill graph capture refuses the 20
  KV-shared layers, which reuse an earlier layer's attention id. Decode capture takes about 6 minutes per
  start. `CUDA_GRAPH_MAX_BS_DECODE=64` captures up to `MAX_RUNNING_REQUESTS`; unset, SGLang
  sizes it from card memory and captures 1, 2, 4 and 8 only, so 16 requests and more would run without graphs.
- **Port 8030.** Radix (prefix) cache off. Drafter: `SPECULATIVE_MODEL` (Google's assistant), served as
  `--speculative-algorithm NEXTN`, which SGLang promotes to `FROZEN_KV_MTP` for
  `Gemma4AssistantForCausalLM`.

## Commands

`make install | command | serve | status | query | test | lint`. Tests are `unittest`, offline.
