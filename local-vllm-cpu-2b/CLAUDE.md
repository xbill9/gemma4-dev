# CLAUDE.md — local-vllm-cpu-2b

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

**vLLM on the local CPU**, serving `google/gemma-4-E2B-it`. No accelerator, no
control plane, no cloud.

**STATUS 2026-09-04: scaffolded, cannot serve on this host, nothing measured.**
`benchmarks/runs/` is empty on purpose.

## Read this first: what this directory was until 2026-09-04

**It was a verbatim copy of `gpu-jax-g4dn-2b`** — 58 tracked files whose README,
`CLAUDE.md`, `tpu.env` and engine all described an AWS G4dn instance with an
NVIDIA T4, plus `jax_engine.py`, `ports/gemma4/`, `boto3`, an `init.sh`, and a
skill directory named **`gpu-jax-g4dn-2b-management`**. The name
`local-vllm-cpu-2b` was a claim nothing in the directory supported: it contained
no vLLM, no CPU targeting, and no `local` anything.

That skill name was the sharpest hazard. `make skill-install` `rm -rf`s its
destination, so two rigs sharing a skill name is destructive rather than merely
confusing (root `CLAUDE.md`, `@NAMING.md`).

All of it is deleted. `tests/test_server.py::TestNoCloudControlPlane` asserts the
**absence** of that vocabulary and is the most load-bearing class in the suite —
**a fork of a cloud rig keeps passing its own tests while describing hardware
that does not exist.** Dead cloud code here is worse than unused: it reads as
live configuration.

The test parses the AST and checks identifiers and string literals with
**docstrings excluded**, because this file and `server.py` legitimately name
`gpu-jax-g4dn-2b` as provenance — and because a substring search for `ami` hits
half the English language. Check code, not prose.

## Why the rig exists

**It is the runtime control for `local-jax-cpu-2b`.** Same host, same checkpoint,
no chip: the only place a vLLM number could be attributed entirely to the engine
rather than to silicon.

**And it is the only vLLM rig this machine could ever run.** The GTX 1650 Ti has
4096 MiB. E2B under vLLM needs **8.15 GB** of w4a16 weights before KV, because vLLM
holds the **2.349 B-parameter per-layer embedding table resident** where
llama.cpp creates it with `TENSOR_READ_LAZY` and gathers rows out of the mmap —
it never reaches the device. That is why `local-llamacpp-1650ti-2b-q4_0` fits in
1612 MiB and vLLM does not fit in 4096. **The PLE is the whole story on this
machine, on the GPU and on the CPU alike.**

## The budget — and three corrections that all ran the same way

MEASURED 2026-09-04. **The first version of this rig got all three wrong, and
every one of them made the rig look less feasible than it is.** They are recorded
here rather than quietly fixed, because the shape of the error is the lesson: an
assumption about a stack was written down as a measurement.

| checkpoint | weights | + KV + overhead | vs 9.25 GB available |
| :--- | ---: | ---: | :--- |
| `gemma-4-E2B-it` (bf16) | 10.25 GB | 12.39 GB | short 3.15 GB |
| **`gemma-4-E2B-it-qat-w4a16-ct`** | **8.32 GB** | **10.46 GB** | **short 1.22 GB** |

Machine total is **16.42 GB**, so both fit the machine. `check_host_capacity`
distinguishes "does not fit right now" from "does not fit this machine" for
exactly this reason — the remedies are unrelated, and conflating them is what
turned a close call into a reported dead end.

**1. There is no fp32 upcast on x86.** `CpuPlatform.supported_dtypes` returns
`[bfloat16, float16, float32]` unconditionally; the comment above it reads
*"x86/aarch64 CPU has supported both bf16 and fp16 natively"*. AVX512-BF16 is a
speed property, not a footprint one. The first version doubled the weights to
20.49 GB and reported a 13.16 GB shortfall — **2× inflated**.

**2. AVX2 is a supported build target, not a hack.**
`cmake/cpu_extension.cmake` carries `CXX_COMPILE_FLAGS_AVX2` beside the AVX512
set and dispatches. The only hard x86 requirement is `gcc/g++ >= 12.3`; this host
has 14.2. What survives: no published CPU wheel, so it is a source build, and
that build replaces the system torch (currently 2.14.0+cu130).

**3. w4a16 buys 19%, not 75%.** The checkpoint's own `ignore` list holds the
vision tower and the embeddings at bf16; only the linears are packed
(`pack-quantized`, 4-bit int, group 32, symmetric). `@MODELS.md` already recorded
8.15 GB resident. **Never size this family from `weights ÷ 4`** — it
under-predicts by ~3×, and that is precisely where this rig's earlier "~2.9 GB"
figure came from.

**What survives all three**, and is the reason this rig is CPU-only: vLLM cannot
fit the GTX 1650 Ti's 4096 MiB, by a wider margin than first stated — 8.15 GB of
w4a16 weights, not 2.9. vLLM holds the 2.349 B-parameter per-layer embedding
table resident where llama.cpp creates it `TENSOR_READ_LAZY` and gathers rows out
of the mmap, so it never reaches the device.

### The failure mode is still why `start_vllm_server` refuses

Exceeding a cloud quota is refused at the API. Exceeding host RAM is **accepted**
and paid for in swap — 15.4 GB of it here — so an over-budget serve is
indistinguishable from a loading one. It never gets worse than "still loading".
That refusal is the one decision this rig takes away from the operator, and it
stands whatever the shortfall is.

## The artifact: this rig cannot borrow its siblings'

**vLLM cannot load a GGUF, on any platform slot.** Verified 2026-09-02 against a
stock vLLM 0.26.0 CUDA install — no `gguf` module under
`model_executor/layers/quantization/`, 31 entries in `QUANTIZATION_METHODS` and
none of them `gguf` (`@QUANTIZATION.md`). So the 3.35 GB q4_0 file already on
this machine, which both 1650 Ti rigs serve, is unusable here.

This rig needs its own **10,246,621,918 B** download of
`google/gemma-4-E2B-it` — MEASURED from the Hub 2026-09-04, ungated. It has not
been fetched, because the capacity arithmetic says it could not be served.

## The reachable CPU baseline is the JAX sibling, and the reason generalises

`local-jax-cpu-2b` fits this host, and `jax 0.11.1` is already installed with a
`CpuDevice`. Its footprints, INHERITED from `gpu-jax-g5g-2b`'s 2026-08-26
measurement and carrying because they are properties of the checkpoint rather
than the device:

| Config | Weights |
| :--- | ---: |
| `ple0` (dense) | 9.257 GB |
| **`ple4` — its default** | **5.752 GB** |

Plus ~1.6 GB of prefill transient — flat below a 4K bucket, not per-token — for
**7.35 GB against 9.48 available.** It fits.

**`PLE_BITS=4` is why, and vLLM has no equivalent.** The per-layer embedding
table is a gather, never a matmul, so quantizing it is a pure memory win and
costs **0.0% decode**. That is the same mechanism that lets llama.cpp keep the
table off a 4 GB GPU entirely. A CPU baseline on this machine is reachable — just
not through vLLM.

## Conventions

- Tests are `unittest`, never pytest: `python3 -m unittest discover -s tests -v`.
- Every subprocess call goes through `run_command(cmd: list[str])` using
  `asyncio.create_subprocess_exec`. **Never `shell=True`.**
- MCP tools are `async def` returning markdown strings with emoji status
  prefixes (`✅`, `❌`, `📡`).
- `Optional[str]`, not `X | None`.
- Use the system `python3` and install into it. **Never create a virtualenv.**
- `tpu.env` is the source of truth and is committed. Never add `*.env` to
  `.gitignore`.
- Read **`MemAvailable`, never `MemFree`**. `MemFree` excludes reclaimable page
  cache and reads catastrophically low on a live desktop; `MemAvailable` is the
  kernel's own estimate of what an allocation can get, and every decision here
  turns on it. A test pins this.
- No `.claude-plugin/`, no `.codex/`, no `skills/` — matching the other `local`
  rigs, which have none and are absent from the root `marketplace.json`.

## Canonical root references

Read these before deriving their numbers here, and correct them **there**:
`@MODELS.md` (checkpoint properties, KV cost, weight footprints), `@HARDWARE.md`
(accelerator properties), `@QUANTIZATION.md` (what the serving stack supports),
`@NAMING.md` (how any of it is spelled), `@RIG-ANALYSIS.md` (the order to consult
them in — and the reason the arithmetic in this file came before a build).
