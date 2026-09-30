# CLAUDE.md — local-llamacpp-i71360p-4b-q4_0

The **E4B parallel of `local-llamacpp-i71360p-2b-q4_0`**, forked from it on 2026-09-30.
Same host, same llama.cpp binary (`fc07d781e`), same flags; **slot 4 is the only slot that
differs.** Read the E2B rig's `CLAUDE.md` for everything this file does not repeat — the
host table, thermal discipline, threads, building llama.cpp, thinking-first replies and
the conventions. Those are properties of the host and the engine, **measured there on
E2B, not here**; `tpu.env` says so at the top.

## What it serves

`gemma-4-E4B-it-q4_0-exact.gguf` (SHA-256 `5462cc10…`, 4.22 GB): Google's E4B QAT GGUF
metadata byte for byte, with every weight matrix, both embedding tables and
`per_layer_model_proj` rebuilt as Q4_0 from
`google/gemma-4-E4B-it-qat-q4_0-unquantized` on the trained grid step. Built and
measured by `local-llamacpp-1650ti-4b-q4_0`
(`benchmarks/runs/2026-09-30-exact-gguf-e4b-1650ti/`): mean KL divergence from bf16
0.00103 against Google's 0.0353. Hub revision `8b7c081`. `make download` checks the
SHA-256 against `MODEL_SHA256`.

**E4B is not a 4B model.** 4.5B effective, 8.0B total (`@MODELS.md`).

- `per_layer_token_embd` is `[10752, 262144]`, 1.585 GB, 38% of the file, and is read
  lazily out of the mmap. **Never pass `--no-mmap`.** A test asserts it.
- KV at 8192 is 128 MiB full + 40 MiB sliding (measured on the 1650ti sibling). Host
  RAM does not bind at this size.

## It shares the host with the E2B rig

Port **8091**, not 8090, so each rig's status tools see only their own server.
`attest.py` identifies the device, and both rigs are the same device. Running both at
once shares the cores, RAM and thermal budget, so **stop one before measuring the
other**, and check `/v1/models` or the process's `-m` before attributing a number.

`LLAMA_CPP_DIR` (`~/llama.cpp-fc07d78`) is the E2B rig's worktree. Both rigs serve that
one binary; checking out a different commit there moves both.

## Not known yet

Everything on this host: first light, speed, any thread setting chosen for E4B rather
than inherited, and any comparison against the E2B sibling. Expect roughly half E2B's
decode speed (the 1650ti and i7-10750H pairs both measured ~0.5x), but that is not a
measurement here.
