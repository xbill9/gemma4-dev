# 2026-08-19 — first serve under pure JAX on EC2 G5g (Graviton2 + NVIDIA T4G)

**This is the rig's first measurement and the finding it was built to establish.**
Gemma 4 E2B serves on Turing (SM 7.5) behind an aarch64 host through **pure JAX** — no
PyTorch, no torch_xla, no vLLM — installed entirely from wheels, with **no from-source
build, no CUDA toolkit, no Rust toolchain, and no kernel patch**. Everything below was
executed on real hardware. Where a number was not measured, it says so.

## Result

```
prompt:  "In two sentences, what is a Graviton processor?"
content: 'A Graviton processor is a type of processor developed by Google that is
          optimized for efficiency and performance, particularly in cloud computing
          environments. These processors are designed to deliver better performance
          per watt, making them highly effective for data centers and scalable
          applications.'
finish_reason: stop
```

The claim is wrong — Graviton is AWS, not Google — which is E2B being a 2B model, not a
serving defect. Output is coherent, terminates on EOS, and the chat template applies.

| | |
| --- | --- |
| Throughput | **12.00 tok/s** end-to-end, 5 runs, min 11.99 / max 12.01 |
| Decode gauge (warm) | 12.5 tok/s |
| Prefill (warm) | 131 ms @ 17-token prompt |
| Prefill (cold) | **7,372 ms** — XLA compilation |
| Weight load | 9.26 GB in **158.8 s**; ~80 s on warm-cache restart |
| GPU memory | 13,573 / 15,360 MiB in use |
| Runtime install | ~5 min, wheels only |

**One sample per cell, single stream, `max_num_seqs=1`, no repeats.** Coverage:
**2 cells measured, 0 failed, 5 infeasible** — see "What was not measured".

## Environment

| | |
| --- | --- |
| Instance | `g5g.2xlarge` spot, `us-east-1c`, 8 vCPU / 16 GiB, $0.369/hr |
| GPU | NVIDIA **T4G**, compute capability **7.5**, **15,360 MiB**, driver 580.126.09 |
| AMI | `ami-077792d0bb6a000b8` — Deep Learning ARM64 AMI OSS Nvidia Driver GPU PyTorch 2.7 (Ubuntu 22.04.5) |
| Resolved via | SSM `/aws/service/deeplearning/ami/arm64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id` |
| JAX | **jax 0.11.1**, jaxlib 0.11.1, jax-cuda12-plugin 0.11.1, jax-cuda12-pjrt 0.11.1 |
| CUDA | **from pip**: nvidia-cublas-cu12 12.9.2.10, nvidia-cudnn-cu12 9.24.0.43. No toolkit, no `nvcc`. |
| Python | 3.12 (deadsnakes) — jax 0.11 requires it; the 22.04 base ships 3.10 |
| Other | transformers 5.15.1, fastapi 0.141.1, jinja2 3.1.6 |
| Serving flags | `--kv-cache-dtype auto --quant-mode fp16 --max-model-len 8192` |
| Engine | this repo's `jax_engine.py` + `ports/gemma4`, served by `jax_openai_server.py` under systemd |

**No tensor-parallel flag.** The engine is single-device (`jax.devices()[0]`); this size has
one T4G, so nothing was sharded.

## Throughput, measured

Five end-to-end runs from a laptop over the public endpoint, 17-token prompt, 64 completion
tokens, after warm-up:

| run | wall (s) | completion tokens | tok/s |
| ---: | ---: | ---: | ---: |
| 1 | 5.34 | 64 | 11.99 |
| 2 | 5.33 | 64 | 12.01 |
| 3 | 5.33 | 64 | 12.00 |
| 4 | 5.33 | 64 | 12.00 |
| 5 | 5.33 | 64 | 12.01 |

A 0.02 tok/s spread. End-to-end includes prefill and HTTP round trip; the server's own
`decode_tokens_per_second` gauge reads 12.5 over the same requests.

**Cold is not warm, and it is not close.** The first two requests off a fresh engine:

| | cold | warm | ratio |
| --- | ---: | ---: | ---: |
| Prefill | 7,372.5 ms | 131.0 ms | **56x** |
| Decode | 4.6 tok/s | 12.5 tok/s | **2.7x** |

That is XLA tracing and compiling per shape bucket. A harness that does not warm up
understates this rig by more than a factor of two on decode and a factor of 56 on TTFT.
The engine pads to static sequence buckets specifically to bound how often this recurs.

## Memory

| | Value |
| --- | ---: |
| Capacity (`nvidia-smi`) | 15,360 MiB |
| In use while serving | 13,573 MiB |
| Weights loaded | 9.26 GB reported by the engine |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 0.90 |

**KV capacity was not measured and is not derivable from this run.** This engine allocates a
static per-sequence cache as ordinary JAX arrays inside the preallocated fraction — there is
no engine-managed shared KV pool, so the vLLM-style "N resident KV tokens / max concurrency
Nx" figures have no equivalent here. The ceiling is `max_model_len × max_num_seqs`, which was
8,192 × 1 for this run.

Host RAM: 16 GiB, **no swapfile** — `_needs_swap` is false at 16 GiB. The sub-16 GiB path
(`g5g.xlarge` + 16 GiB swap) was not exercised on this rig.

## Three failures that reported success

All three were found by running the thing, and none was visible to the offline test suite.
Recorded here because each is a false-negative in a *check*, which is the failure mode this
project treats as most expensive.

**1. `verify_gpu_arch` could never pass.** Its probe built a 256×256 fp16 matrix of ones,
multiplied, and compared the sum against 256³ = 16,777,216. **float16's maximum finite value
is 65,504**, and `jnp.sum` on a float16 array accumulates in float16, so the reduction
overflowed to `inf` on any device in existence. It reported `fp16 matmul ok: False` against a
GPU that was provably healthy:

```
y[0,0]: 256.0            elementwise correct: True
fp16 sum: inf            fp32 sum: 16777216.0   expected: 16777216.0
```

Fixed by accumulating in fp32. A dead CPU-fallback branch was fixed in the same pass — it
matched on the literal `"platform"`, which the probe never printed, so a genuine CPU fallback
would have produced no verdict at all.

**2. The systemd unit ran an interpreter that had no jax.** The DLAMI already carries
`/usr/local/bin/python3.12`, which precedes `/usr/bin` on PATH. The bootstrap installed 3.12
from deadsnakes — a *second* interpreter at `/usr/bin/python3.12` — then ran bare `python3.12`
to install jax, which resolved to the DLAMI's copy. `ExecStart=/usr/bin/python3.12` therefore
crash-looped on `ModuleNotFoundError: No module named 'jax'` **after the install had reported
success**, because the post-install verification also resolved through PATH and so tested the
wrong interpreter.

```
PATH python3.12       -> /usr/local/bin/python3.12   jax 0.11.1   OK
/usr/bin/python3.12   -> ModuleNotFoundError: No module named 'jax'
```

`install.sh` now resolves the interpreter after installing and rewrites `ExecStart` to that
absolute path. Pinned by `test_execstart_is_repointed_at_the_installed_interpreter`.

**3. `/health` returned 200 while every generation returned 500.** `jinja2` was absent.
`transformers` renders chat templates through it and does not depend on it, so the process
started clean, reported healthy indefinitely, and could not emit a token:

```
{"detail":"apply_chat_template requires jinja2 to be installed."}
```

It also does not self-heal: `transformers` memoizes the availability check at import, so
installing jinja2 into the running host changed nothing until the unit restarted. Added to
`_SERVING_REQUIREMENTS` and `requirements-serving.txt`.

**This is the one to design against.** A liveness probe that checks only `/health` would have
called this deployment successful. `verify_model_health` caught it solely because it asserts
on the returned completion text.

## What the JAX path skipped

Same instance family, same GPU, same checkpoint, single stream. The vLLM column is this
monorepo's `2026-08-12-first-serve-g5g` run in `gpu-vllm-g5g-2b`.

| | vLLM (2026-08-12) | pure JAX (this run) |
| --- | --- | --- |
| Time to a serving endpoint | ~67 min build + toolkit + Rust | **~5 min, wheels only** |
| Patches required | 1, unlanded, on one volume | **0** |
| Compiler on the box | `cuda-toolkit-13-2` (sbsa) | **none** |
| Attention | `TRITON_ATTN`, forced, patched | XLA |
| GPU memory in use | 13,501 MiB | 13,573 MiB |
| Throughput, single stream | **43.1 tok/s** | **12.0 tok/s** |

> **CORRECTION 2026-08-30.** The 43.1 in the row above is a single sample from the vLLM rig's
> first-serve smoke test, not a benchmark — see that report's own caveat. The comparable
> measured figures are c=1 TPOT 31.44 ms (~31.8 tok/s decode) and c=8 168.33 tok/s, from
> `gpu-vllm-g5g-2b/benchmarks/runs/2026-08-14-rust-frontend-g5g/`. **The 3.5x gap is real and
> survives the correction** (~31.8 against 12.0 is 2.6x at c=1), but the caveat below
> understates the case: the tile clamp applies to every vLLM-on-T4G number, so it was never
> the thing that made this comparison uncontrolled.

**Treat the 3.5x as directional, not as a controlled comparison.** Different instance sizes
(`g5g.4xlarge` vs `g5g.2xlarge` — irrelevant to decode, which is GPU-bandwidth-bound, but not
nothing), different context (16,384 vs 8,192), different concurrency ceiling (`max_num_seqs`
8 vs 1), and the vLLM figure was obtained on a patched kernel with reduced Triton tiles. One
sample per cell on both sides. What the number does support: a reference model implementation
with an HTTP server attached does not approach a serving engine, and the build was never what
you were paying for.

## What the fused W4A16 kernel cannot do here

Not measured — **refused at startup by design**, and recorded because it is a property of this
pairing. The fused W4A16 Pallas kernel is tiled for TPU VMEM (megabytes) and needs
**550 KiB – 1.1 MiB of shared memory per block** at this model's shapes, against Turing's
64 KiB opt-in ceiling. On GPU, Pallas lowers through Triton and those tiles become shared
memory. `check_w4a16_fits_scoped_memory()` computes the requirement and raises before
compiling rather than dying as an `OutOfResources` at the first token.

This rig therefore serves the **dense reference checkpoint** at fp16. The dequantize-then-
matmul fallback is worse than not quantizing: 4x the weight traffic of the fused path *and*
the dense model's memory.

## What was not measured

Recorded so coverage is not overstated by omission. Five cells were deliberately not run:

| Cell | Status | Why |
| --- | --- | --- |
| Concurrency > 1 | infeasible | Engine is `max_num_seqs=1`; no continuous batching exists to measure. |
| Context sweep (2k/4k/8k) | not run | Single first-serve validation; only the 17-token prompt was exercised. |
| Memory bandwidth (STREAM-like) | not run | No microbenchmark executed. The sibling's 277 GB/s streaming-read figure is a property of the chip and lives in `HARDWARE.md`; it was not re-measured here. |
| KV capacity / resident KV tokens | infeasible | No engine-managed KV pool — see "Memory". |
| Two-GPU sizes (`16xlarge`, `metal`) | infeasible | Engine is single-device; the second T4G would idle. |
| `g5g.xlarge` + swapfile path | not run | 16 GiB host needs no swap; the sub-16 GiB path is untested on this rig. |

No thermal or power figures were captured. GPU utilization was read only at idle (0%).

## Corrections to this rig's earlier documentation

- `CLAUDE.md` claimed this rig had "now served" at ~43 tok/s and pointed at
  `benchmarks/runs/2026-08-12-first-serve-g5g/`. **That was the vLLM sibling's run**, carried
  over by the fork. This directory is the rig's first actual measurement.
- The same file recommended launching from `ami-0b44b90b3d02430ee`. That is a vLLM image
  carrying the Triton patch and no JAX; it is wrong for this rig, which needs no prebuilt AMI
  because nothing is compiled.
- `project-setup.sh` carried a hardcoded `SKILL_STEM` naming the vLLM skill and died with
  `cannot locate the bundled skill`, so this rig could never be registered at all.
- `tpu.env` marked throughput values PREDICTED. The predictions were not tested here — this
  run replaces the serving question, not the KV-ceiling estimate, which remains unmeasured.

## Reproduce

```bash
# 1. Launch. g5g.2xlarge spot; expect to walk AZs — 1a and 1b both refused capacity.
#    AWS's own InsufficientInstanceCapacity message recommends AZs by on-demand
#    availability and is not spot-aware.
python3 -c "import asyncio,server; print(asyncio.run(server.create_g5g_instance(
    subnet_id='<subnet>', security_group_id='<sg>',
    iam_instance_profile='g5g-jax-instance-profile')))"

# 2. Wait for the wheel install (~5 min). INSTALL COMPLETE means jax imported AND saw the GPU.
#    get_install_progress

# 3. Prove the GPU before deploying anything.
#    verify_gpu_arch   -> expects "fp16 matmul ok: True"

# 4. Ship the payload; the engine compiles on first request.
#    deploy_jax_server -> get_jax_logs -> verify_model_health

# 5. WARM UP before recording anything. The first request is 56x slower on TTFT.
```

Measured 2026-08-19 on `g5g.2xlarge` spot, `us-east-1c`, instance `i-063d52c913140b787`,
terminated after the run. Total instance cost for the session: ~$0.83.
