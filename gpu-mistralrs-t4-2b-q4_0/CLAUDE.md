# CLAUDE.md — gpu-mistralrs-t4-2b-q4_0

mistral.rs v0.9.5 (`465191d`, `/opt1/mistral.rs`) on the T4 serving the exact Q4_0 GGUF llama.cpp serves,
with Google's mmproj (SHA-256 `021059cc…`, which mistral.rs always loads for the gemma4 GGUF architecture).
Forked from `local-mistralrs-3050oem-2b-q4_0`.

- **Built with `--features cuda` only**, `CUDA_COMPUTE_CAP=75`: flash-attn needs sm_80. Binary at
  `/opt1/mistralrs-v0.9.5/bin/mistralrs`, build log `/opt1/tmp/build-mistralrs.log`.
- Same fixed-cache sizing as the 3050 rig (`--kv-cache 8192tokens --max-seqs 4`), not re-derived for 15 GB.

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
