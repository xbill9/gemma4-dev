# CLAUDE.md — gpu-pytorch-t4-2b-q4w4a16

This repo's int4 PyTorch engine (`e2b_engine.py`, `w4gemv.py`, `w4attn.py`, `openai_server.py`,
byte-for-byte from `local-pytorch-3050oem-2b-q4w4a16` except `RIG_NAME`) on the T4, serving
`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text` @ `315aec06` with int4 vocabulary tables from `-text-emb4` @ `db715f83`.

- Needs `accelerate` (added `--no-deps` to `/opt1/pyuser-engines`). transformers is 5.20.2 here.
- First start 2026-11-05: weights 2,498 MiB, `prefill_dtype float16` (measured fp16 19.51 TFLOPS against fp32
  2.82), load 649.9 s with a cargo build running.
- Serves one request at a time, no drafter.

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
