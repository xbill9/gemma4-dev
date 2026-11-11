# CLAUDE.md — gpu-vllm-t4-2b-q4w4a16emb4

vLLM 0.33.0 on the T4 serving `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` @ `db715f83` (the 3050
rig's checkpoint and revision). Forked from `local-vllm-3050oem-2b-q4w4a16emb4`.

- **Not `gpu-vllm-t4-2b-w4a16`**, which serves the same checkpoint on this card with vLLM 0.29 from
  `/opt1/pyuser` and predates slot 5's fifth rule. This rig keeps its own user base, so neither moves the other.
- **Install:** torch 2.14.0 from the cu130 index alone, then every package of a `uv pip compile` lock of
  `vllm==0.33.0` (`/opt1/tmp/vllm033-lock.txt`) except torch/torchvision/torchaudio. A plain
  `pip install vllm==0.33.0` backtracked for over 20 minutes on `httpx2`/`httpcore2` (via `mcp`).
- **The Turing clamp is required** (`patch_triton_turing.py`, copied from `gpu-vllm-t4-2b-w4a16`): vLLM forces
  `TRITON_ATTN` for Gemma 4's 256/512 heads and its tile wants more than sm_75's 64 KiB. Applied to
  `/opt1/pyuser-engines/.../vllm/v1/attention/ops/triton_unified_attention.py` (original in
  `/opt1/tmp/triton_unified_attention.py.orig-0.33.0`); re-run after any vLLM reinstall.
- `serve.py` runs `$PYTHON_BIN -m vllm.entrypoints.cli.main serve` with `--dtype float16`.
- First start 2026-11-05: `MarlinLinearKernel`, `TRITON_ATTN`, model loading 2.85 GiB, KV 880,057 tokens at
  0.85, 340 s to healthy (compile, with a cargo build on both vCPUs).

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
