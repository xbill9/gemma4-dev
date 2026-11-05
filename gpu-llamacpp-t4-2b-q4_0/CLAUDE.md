# CLAUDE.md — gpu-llamacpp-t4-2b-q4_0

`llama-server` (llama.cpp `7182ef71e`, `/opt1/llama.cpp`, CUDA 13.4, sm_75) serving the exact Q4_0 E2B GGUF
(SHA-256 `419db9a6…`) and the exact Q4_0 QAT drafter (`060f5a01…`) on the T4. Forked from
`local-llamacpp-3050oem-2b-q4_0`; the same commit as that rig's 2026-11-03/04 runs.

- `-t 2 -tb 2` (2 vCPU). `per_layer_token_embd` on the card (`OVERRIDE_TENSORS`).
- `server.py` (MCP) came with the fork and is unregistered here; the benchmarks drive `llama-server` directly.
- First check 2026-11-05: with the drafter at 2 tokens, 2,879 MiB at `-c 8192`, acceptance 0.71 on a short reply.

## Shared by the six T4 engine rigs

- **Host:** GCE `n1-standard-2` (2 vCPU, 7.4 GiB RAM), Tesla T4 (TU104, sm_75, 15,360 MiB), driver 620.70.06.
  `sudo swapon /opt1/swapfile` after a reboot (16 GB, not in fstab); the 16-bit SGLang load and vLLM need it.
- **Interpreter:** `/usr/bin/python3.13` with `PYTHONUSERBASE=/opt1/pyuser-engines` (vLLM 0.33.0, SGLang
  0.5.23, litert-lm 0.19.0, torch 2.14.0+cu130, transformers 5.20.2). `python3` on this host is pyenv 3.12
  with a CPU torch. `/opt1/pyuser` (vLLM 0.29) belongs to `gpu-vllm-t4-2b{,-w4a16}` and is left alone.
- **One engine on the card at a time.** Ports: llama.cpp 8080, vLLM 8010, mistral.rs 8020, SGLang 8030,
  PyTorch 8100, LiteRT 9379; the older vLLM T4 rigs use 8000.
- **Forked 2026-11-05** from the `local-*-3050oem-2b-*` rig of the same engine, for the T4 run of
  `docs/serving-engines-3050`. Same weights by SHA-256 or revision, same engine versions, except where
  this file says otherwise. Code came across; measurements did not.
- Tests are `unittest`. Never a virtualenv. `tpu.env` decides and is committed.
