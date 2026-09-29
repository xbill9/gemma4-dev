# CLAUDE.md — local-llamacpp-i71360p-2b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

`llama-server` from `ggml-org/llama.cpp`, driven directly, serving the **exact Q4_0
GGUF** of Gemma 4 E2B-it (`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`) on the **CPU
only** of a Lenovo Yoga 9 14IRP8 (i7-1360P). No GPU, no control plane, no cloud.

**STATUS 2026-09-29: serving on `fc07d781e`.** First light through the MCP tools,
the Google-vs-rebuilt comparison and a thread spot-check are in
`benchmarks/runs/2026-09-29-exact-gguf-i71360p/`. The file now served also stores
`per_layer_model_proj` as Q4_0 (19.8 MB smaller, KL divergence unchanged):
`benchmarks/runs/2026-09-29-exact-gguf-v2-i71360p/`.

## Where the code came from, and what did not come with it

The code (`server.py`, `attest.py`, `sweep.py`, `inspect_gguf.py`, the tests, the
Makefile) is forked from `local-llamacpp-cpu-2b-q4_0` on 2026-09-29. **Nothing
else was.** That rig is pinned to an i7-10750H in a different machine, and its
`tpu.env`, `CLAUDE.md` and `benchmarks/` describe that host; its runs were not
copied here. Slot 3 names the CPU part so the two cannot be confused
(`NAMING.md`, slot 3).

`attest.py` and the CPU-only guards came with the code. There is no GPU twin here
— this host has Intel Iris Xe only — and the guards stay because the rig's claim
should not depend on which machine it is checked out on.

## The model

Google's `gemma-4-E2B-it-qat-q4_0-gguf` metadata, byte for byte, with every Q4_0
tensor, both embedding tables and `per_layer_model_proj` rebuilt from
`google/gemma-4-E2B-it-qat-q4_0-unquantized` on the trained grid step
(`gguf_exact.py` in the v2 run directory). SHA-256 in `tpu.env`.

| tensor | type | bytes | touched per token |
| --- | --- | ---: | --- |
| `per_layer_token_embd.weight` `[8960, 262144]` | Q4_0 | 1,321.2 MB | a few rows — `TENSOR_READ_LAZY` |
| `token_embd.weight` `[1536, 262144]` | Q4_0 | 226.5 MB | all of it (tied output projection) |
| 35 transformer blocks | Q4_0 (norms F32) | 1,049.1 MB | all of it |
| `per_layer_model_proj.weight` `[1536, 8960]` | Q4_0 | 7.7 MB | all of it |

`inspect_gguf.py` re-derives this from the file. All of it is Q4_0 except 1.1 MB
of F32 norms and scale vectors.

- **Never pass `--no-mmap`.** `TENSOR_READ_LAZY` requires mmap. A test asserts it.
- **Reported model size:** llama-bench prints 2.43 GiB for this file, 2.44 GiB for
  the first rebuild and 3.10 GiB for Google's.

## This host

| | |
| :--- | :--- |
| Machine | Lenovo **Yoga 9 14IRP8** (DMI 83B1), bare metal, nvme + ext4 |
| CPU | Intel Core **i7-1360P**, **hybrid**: 4 P-cores with SMT at logical 0-7, 8 E-cores at 8-15 |
| SIMD | `avx2 avx_vnni` — no AVX-512, no AMX |
| RAM | 15 GiB, plus 15.7 GiB swap |
| GPU | none discrete (Intel Iris Xe); no `/dev/nvidia*` |
| OS | Debian sid, kernel 7.2.8, gcc 16.2.0 |

`cpu_status` re-reads this from `/proc` and `/sys`. Trust the kernel over this
table, and re-measure after any hardware move.

## Measuring here

- **Temperature moves absolute numbers more than most levers.** The same file
  generated at 21.70 tok/s in a pass starting at 76 °C and 24.45 in one starting at
  53 °C. Compare files or settings **in one invocation, in both orders**, and log
  `x86_pkg_temp` around each pass (the run directory shows how).
- **Threads: `-t 8`.** In one spot-check, 4, 8 and 12 threads were inside one
  another's spread for generation and 16 was 23% slower. Not a sweep, and no
  affinity is set; the kernel spreads threads over P- and E-cores.
- **The native build is faster than `ghcr.io/ggml-org/llama.cpp:full` at the same
  commit** (about 1.46x in runs an hour apart, not interleaved). Quote this rig's
  binary; the Docker figures in `docker/` are for the three-file comparison only,
  where all three shared one binary.
- `KV_CACHE_TYPE` and `FLASH_ATTENTION` are llama-server's usual values and were
  not varied here.

## Building llama.cpp

`LLAMA_CPP_DIR` is this rig's **own worktree** (`~/llama.cpp-fc07d78`) at
`LLAMA_CPP_COMMIT`. `~/llama.cpp` belongs to `local-llamacpp-cpu-2b-q4_0`; checking
out a different commit there moves that rig's binary. `make build` checks out the
pinned commit and builds `llama-server`, `llama-bench` and `llama-perplexity` with
an explicit `-j`; never pass a bare `-j`.

## Gemma 4 reasons first

llama.cpp routes the thinking block to `reasoning_content` and leaves `content`
empty until it closes. `query_model` defaults to `max_tokens=1024` (a test keeps it
at 512 or more) and reports 📡 when only reasoning came back. For a direct reply,
send `"chat_template_kwargs": {"enable_thinking": false}`.

## Conventions

- Tests are `unittest`, never pytest: `python3 -m unittest discover -s tests -v`.
- Every subprocess call goes through `run_command(cmd: list[str])` using
  `asyncio.create_subprocess_exec`. **Never `shell=True`.** (`_spawn_detached` uses
  `subprocess.Popen` without a shell so the server outlives the MCP process.)
- MCP tools are `async def` returning markdown strings with emoji status
  prefixes (`✅`, `❌`, `📡`).
- `Optional[str]`, not `X | None`.
- Use the system `python3` and install into it. **Never create a virtualenv.**
- `tpu.env` is the source of truth and is committed. Never add `*.env` to
  `.gitignore`.

## Canonical root references

`@MODELS.md`, `@HARDWARE.md`, `@QUANTIZATION.md`, `@NAMING.md`,
`@RIG-ANALYSIS.md`. Read them before deriving their numbers here, and correct them
there.
