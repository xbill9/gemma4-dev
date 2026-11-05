# CLAUDE.md — gpu-sglang-t4-2b-q4_0

SGLang 0.5.23 on the T4. **Slot 5 differs from the 3050 rig**: it serves
`google/gemma-4-E2B-it-qat-q4_0-unquantized` @ `6befbaca` (the QAT Q4_0-grid values in bf16, served
float16), because SGLang's W4A16 path cannot run on sm_75 — its Marlin kernels compile to empty stubs below
sm_80. Same weights by value as the other T4 rigs; 16-bit storage, 10.00 GB on the card.
`benchmarks/runs/2026-11-05-first-light-t4/REPORT.md` has the evidence.

- **Three patches** (`patches/`): the 3050 rig's two, and `turing-smem.diff`, which shrinks the Triton
  decode and extend attention tiles on compute capability < 8 at head width ≥ 512 (they asked 98,304 and
  131,072 B of the 65,536 allowed). Originals are `*.orig-0.5.23` beside each patched file.
- **FlashInfer JIT on sm_75** needs `CUDA_HOME=/opt1/cuda-13.4` and `CPATH=/opt1/cuda-extra-include`
  (symlinks to only the curand headers of the pip CUDA 13.0 set; the whole set clashes with 13.4's).
  `serve.py` exports both from `tpu.env`.
- Startup is long on this host: the single 9.6 GB safetensors file swaps on 7.4 GiB of RAM.

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
