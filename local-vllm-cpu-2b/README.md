# local-vllm-cpu-2b

Serve **`google/gemma-4-E2B-it`** with **vLLM on the local CPU** — no accelerator,
no cloud, no control plane.

> **STATUS 2026-09-04: scaffolded, nothing served, and the budget is CLOSE
> rather than closed.** `make capacity` has the live arithmetic. On the
> quantized checkpoint it is short by ~1.2 GB of a 16.42 GB machine — a
> close-a-browser problem, not a wall. `benchmarks/runs/` is empty on purpose.

## Why it exists

**It is the runtime control for `local-jax-cpu-2b`** — same host, same
checkpoint, no chip — and the only place a vLLM number could be attributed
entirely to the engine rather than to silicon.

It is also the **only vLLM rig this machine could ever run**. The GTX 1650 Ti has
4096 MiB; E2B under vLLM needs **8.15 GB** of w4a16 weights before KV, because vLLM
holds the 2.349 B-parameter per-layer embedding table resident, where llama.cpp
gathers it lazily out of an mmap and never puts it on the device at all. That one
structural difference is why `local-llamacpp-1650ti-2b-q4_0` fits in 1.6 GB and
vLLM does not fit in 4.

## Quick start

```bash
make capacity     # run this FIRST — live budget, and it names the quantized route
make install      # deps for the MCP server only; does NOT install vLLM
make verify       # is the installed vLLM actually a CPU build?
make serve        # refuses while capacity refuses
```

## The budget, and the three corrections behind it

MEASURED 2026-09-04. **Everything in this section was wrong in the first version
of this rig, and wrong in the direction that made it look impossible.**

| checkpoint | weights | + KV + overhead | vs 9.25 GB available now |
| :--- | ---: | ---: | :--- |
| `gemma-4-E2B-it` (bf16) | 10.25 GB | 12.39 GB | short 3.15 GB |
| **`gemma-4-E2B-it-qat-w4a16-ct`** | **8.32 GB** | **10.46 GB** | **short 1.22 GB** |

The machine has **16.42 GB total**, so both fit it — the shortfall is against
what is free on a live desktop right now, and the remedy is to close something.

**Correction 1 — there is no fp32 upcast.** vLLM's `CpuPlatform.supported_dtypes`
returns `[bfloat16, float16, float32]` for x86 unconditionally; its own comment
reads *"x86/aarch64 CPU has supported both bf16 and fp16 natively"*. AVX512-BF16
governs how fast a bf16 datapath is, not how many bytes the weights occupy. The
first version doubled the weights to 20.49 GB on this assumption and reported a
13.16 GB shortfall.

**Correction 2 — AVX2 is a first-class build target.**
`cmake/cpu_extension.cmake` carries `CXX_COMPILE_FLAGS_AVX2` beside the AVX512
set and dispatches between them. The only hard x86 requirement is
`gcc/g++ >= 12.3`; this host has 14.2. The build is from source only because no
CPU wheel is published (`wheels.vllm.ai/cpu` → 404; the PyPI `vllm` wheel is the
CUDA build).

**Correction 3 — w4a16 saves 19%, not 75%.** The checkpoint's own `ignore` list
keeps the vision tower and the embeddings at bf16, so only the linears are
packed. `@MODELS.md` records the resident figure as 8.15 GB and it is right. A
`weights ÷ 4` estimate under-predicts by ~3×, which is where this rig's earlier
"~2.9 GB" claim came from.

**What survives all three: this still cannot fit the 1650 Ti's 4096 MiB**, and by
a wider margin than first stated — 8.15 GB of w4a16 weights, not 2.9. vLLM holds
the 2.349 B-parameter per-layer embedding table resident where llama.cpp gathers
it lazily from an mmap.

## The reachable CPU baseline is the JAX sibling

`local-jax-cpu-2b` fits this host and `jax 0.11.1` is already installed with a
`CpuDevice`. Its `PLE_BITS=4` lever takes the weights from 9.257 GB to **5.752
GB** by quantizing the per-layer embedding table — a gather, never a matmul, so
it costs **0.0% decode**. Plus ~1.6 GB of prefill transient, that is 7.35 GB
against 9.48 available.

**vLLM exposes no equivalent.** The PLE is the whole story on this machine, on
both the GPU and the CPU.

## Layout

```
server.py    MCP server — capacity, lifecycle, inference. No provisioning tools.
tpu.env      source of truth, committed
tests/       offline unittest suite — python3 -m unittest discover -s tests -v
benchmarks/  empty; the synced root schema and README only
```

Read `CLAUDE.md` before changing anything.
