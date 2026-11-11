# CLAUDE.md — gpu-litertlm-t4-2b-int248

LiteRT-LM 0.19.0 on the T4 serving Google's `gemma-4-E2B-it.litertlm` (SHA-256 `18193810…`, revision
`cbb97a3e`), GPU backend = WebGPU (Dawn) on Vulkan through `nvidia-vulkan-icd` 620.70.06. Forked from
`local-litertlm-3050oem-2b-int248`; read that rig's CLAUDE.md for the runtime (`max_completion_tokens`,
the `pkill` gotcha: match `[l]itert-lm serve`).

- CLI: `PYTHONUSERBASE=/opt1/pyuser-engines /opt1/pyuser-engines/bin/litert-lm`, models under
  `LITERT_LM_DIR=/opt1/cache/gpu-litertlm-t4-2b-int248`.
- **No exact-Q4_0 sister on the T4**: the 3050's `local-litertlm-3050oem-2b-q4w4a16emb4` build needs ~45 GB of
  RAM to convert, and this host has 7.4 GiB. The file is not published.
- First check 2026-11-05: 1,282 MiB on the card, correct reply.

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
