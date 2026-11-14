# CLAUDE.md — local-vllm-3050oem-2b-q4w4a16emb4

Gemma 4 E2B served by **vLLM 0.33.0** on the RTX 3050 OEM in the desktop `gpu`, from
`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`: Google's QAT Q4_0 weights in compressed-tensors
W4A16 on the QAT grid, vocabulary tables int4 too, text only. Created 2026-11-03 for the 3050
serving-engine comparison. No MCP server, no skill, no plugin: `tpu.env`, `serve.py`, a Makefile, the
sweep harness, measurements. Same card as the `local-llamacpp-3050oem-*`, `local-pytorch-3050oem-*` and
`local-litertlm-3050oem-*` rigs; they do not run beside each other.

## Read first

- **`tpu.env` decides**, and `serve.py` turns it into the `vllm serve` command (`make command` prints it).
  `make serve` and every benchmark script start the server through it.
- **vLLM runs on the system python3 with its torch pin skipped.** vLLM 0.33.0 pins `torch==2.14.0`; the
  system has `2.14.1+cu130`, which the PyTorch rigs were measured on. `make install` installs vLLM's other
  requirements, then vLLM with `--no-deps`. Its kernels are built against libtorch's stable ABI
  (`_C_stable_libtorch.abi3.so`) and serve on 2.14.1. Its own pins downgraded numpy 2.5.4 → 2.4.6, numba
  0.68.1 → 0.65.0 and fastapi 0.162 → 0.136.3 in the shared interpreter; the state before is
  `~/.cache/vllm-install/freeze-before-2026-11-03.txt`. mcp stayed at 2.2.0.
- **Port 8010.** llama.cpp rigs use 8080, LiteRT 9379/9380.
- **It reserves `GPU_MEMORY_UTILIZATION` of the card up front** (0.85: 7,082 MiB at 64 requests, against
  llama.cpp's 2,612 MiB). Nothing else fits beside it.
- **First start compiles** (about 3.5 minutes; later starts read `~/.cache/vllm/torch_compile_cache`).
- Kernels in use: `MarlinLinearKernel` for the int4 linears, `TRITON_ATTN` attention. Model loading 2.85 GiB.

## Harness

`sweep.py` and `attest.py` are copies of `local-llamacpp-3050oem-2b-q4_0`'s, changed in two places:

- **`sweep.py` counts vLLM's thinking tokens.** vLLM 0.33 streams them as `delta.reasoning`; llama.cpp as
  `delta.reasoning_content`. Reading only the latter counts no tokens on short replies and only the
  visible answer on long ones, which inflates time to first token.
- **`attest.py` decides the device from CUDA libraries mapped into the server or any child process.**
  vLLM has no `-ngl`, and its GPU work runs in a child (`EngineCore`).

The request body is unchanged: thinking on (`enable_thinking`), greedy, a unique leading nonce per request
so vLLM's prefix cache cannot hit. `cache_prompt: false` is a llama.cpp field and vLLM ignores it.

## Commands

`make install | command | serve | status | query | test | lint`. Tests are `unittest` (`make test`),
offline.
