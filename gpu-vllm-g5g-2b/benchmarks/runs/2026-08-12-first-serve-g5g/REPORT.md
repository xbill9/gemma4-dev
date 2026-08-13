# 2026-08-12 — first successful serve on EC2 G5g (Graviton2 + NVIDIA T4G)

**This is the rig's first measurement and the finding it was built to establish.**
Gemma 4 E2B serves coherently on Turing (SM 7.5) behind an aarch64 host — but only after
one upstream patch. Everything below was executed on real hardware, not inferred.

## Result

```
content: 'Site Reliability Engineering (SRE) is a discipline that applies software
          engineering principles to infrastructure and operations problems to create
          highly reliable, scalable, and efficient systems.'
finish_reason: stop      usage: 19 prompt / 32 completion / 51 total
```

| | |
| --- | --- |
| Throughput | **42.9 tok/s** @ 64 tokens, **43.1 tok/s** @ 256 — single stream, greedy, `ignore_eos` |
| GPU memory | 13501 / 15360 MiB in use, 100% util, 67 °C |
| KV cache | **2.95 GiB → 329,579 tokens**, max concurrency 20.12x at 16,384 ctx |
| Engine init | 76.4 s (compilation 2.9 s), CUDA graph capture 17 s / 0.15 GiB |

Single-run, single-stream, no repeats and no variance figure. One sample per cell. For
scale only: the Inferentia port records ~44 tok/s for E2B on one NeuronCore, so this is the
same order — but that is a different harness on different silicon and is not a controlled
comparison.

## Environment

| | |
| --- | --- |
| Instance | `g5g.4xlarge` spot, `us-east-1a`, 16 vCPU / 30 GiB |
| GPU | NVIDIA **T4G**, compute capability **7.5**, **15360 MiB**, driver 595.71.05 |
| AMI | `ami-07a66fa2acbcfea88` — Deep Learning ARM64 AMI OSS Nvidia Driver GPU PyTorch 2.12 (Ubuntu 24.04) 20260724 |
| PyTorch | **2.12.0+cu132**, arch list `['sm_75','sm_80','sm_90','sm_100','sm_110','sm_120']` |
| CUDA toolkit | 13.2 (installed separately — see below) |
| vLLM | **v0.27.2rc0** → `0.27.2rc1.dev0+g7f7a32cfe`, built from source for `TORCH_CUDA_ARCH_LIST=7.5` |
| Serving flags | `--dtype float16 --kv-cache-dtype auto --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 8 --tensor-parallel-size 1` |

## Memory, measured

**GDDR6, not HBM** — the only accelerator in this monorepo that isn't. Canonical home for these
is `HARDWARE.md`; repeated here because they were taken in this run.

| | Value |
| --- | ---: |
| Capacity (`nvidia-smi` / torch) | 15,360 MiB / 14,913 MiB |
| Bus, clock | 256-bit, 5,001 MHz |
| Theoretical peak | 320.1 GB/s |
| **Streaming read** | **277.0 GB/s** (87% of peak) |
| Copy (r+w) / in-place scale | 234.3 / 232.3 GB/s (73%) |

Decode is bandwidth-bound, so **277 GB/s is the ceiling that matters**, not 320. Roughly a third
of v5e's 858.99 GB/s and a sixth of v6e's 1,638 — which is the honest frame for 43 tok/s.

**Shared memory is two numbers.** 49,152 B (48 KiB) is the default static per-block limit torch
reports; 65,536 B (64 KiB) is the opt-in ceiling via the dynamic shared-memory attribute, and
what Triton measures against. Both are real; cite the 64 KiB figure only with the qualifier.

## The blocker, and the patch

The packaging gap this rig was created for is **not** what stops it. The real wall:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 98304, Hardware limit: 65536
```

vLLM logs why it gets there:

```
Gemma4 model has heterogeneous head dimensions
{'sliding_attention': 256, 'full_attention': 512}. FA4 not available,
forcing TRITON_ATTN backend.
```

Gemma 4's global-attention layers are **512-wide** (sliding are 256 — matches `MODELS.md`).
Only FA4 or Triton support heterogeneous head dims; FA4 is unavailable, so Triton is forced.
At `head_size=512` the Triton unified-attention kernel wants ~96 KiB of shared memory per
block. **Turing has 64 KiB; Ampere and later have 164 KiB+.** Triton refuses outright.

The fix applied here clamps the KV tile until Q + K/V tiles fit, and drops the software
pipeline to one stage, only on pre-Ampere devices — patched into
`vllm/v1/attention/ops/triton_unified_attention.py`:

```python
if current_platform.get_device_capability()[0] < 8:
    _smem_budget = 60000          # headroom under 65536 for accumulators
    _esz = q.element_size()
    def _fits(tile): return (BLOCK_M + 2 * tile) * head_size * _esz <= _smem_budget
    while TILE_SIZE_PREFILL > 16 and not _fits(TILE_SIZE_PREFILL): TILE_SIZE_PREFILL //= 2
    while TILE_SIZE_DECODE  > 16 and not _fits(TILE_SIZE_DECODE):  TILE_SIZE_DECODE  //= 2
    launch_num_stages = 1
```

With it, CUDA graph capture completes and the server starts. **This patch is not upstream
and is not in this rig's `server.py`** — it currently lives only on the built instance. It
is the single thing standing between a stock vLLM and a working G5g deployment, and it is
the obvious upstream contribution from this work.

The 43 tok/s figure is therefore measured with reduced tiles. An unclamped kernel would
likely be faster and cannot run at all here, so there is no baseline to compare against.

## What the build actually required

- **The PyTorch DLAMI ships no `nvcc`.** Driver + torch only; CUDA toolkit is a separate
  install (`cuda-keyring` → `cuda-toolkit-13-2`) from NVIDIA's **sbsa** repo.
- **vLLM needs `setuptools_rust` and a Rust toolchain** for its `vllm-rs` frontend.
- **No vLLM tag pins torch 2.12** (they go 2.11 → 2.13). `use_existing_torch.py` plus
  `--no-build-isolation` builds against the DLAMI's torch; CMake emits a version warning and
  proceeds.
- **v0.26.0 cannot serve Gemma 4** against current `transformers` —
  `AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute`. The
  `per_layer_config` handling exists only in **v0.27.2rc0 and later**.
- Build time **~67 min** on `g5g.4xlarge` at `MAX_JOBS=12`. **FlashAttention 2 and 3 are
  compiled regardless of `TORCH_CUDA_ARCH_LIST`** and dominate that time; FA2 needs sm80 and
  FA3 needs sm90, so neither can ever load on 7.5. Constraining `VLLM_FA_CMAKE_GPU_ARCHES`
  should cut the build substantially and was not attempted here.

## Corrections to this rig's earlier documentation

Every one of these was asserted before hardware existed, and measurement contradicted it:

| Claimed | Measured |
| --- | --- |
| PyTorch aarch64 wheels lack `sm_75` | **AWS DLAMI torch has it** — 2.7.0+cu128 and 2.12.0+cu132 both |
| `--dtype bfloat16` is a hard failure | bf16 matmul runs; vLLM **auto-casts bf16→fp16** (`Casting torch.bfloat16 to torch.float16`) |
| Attention backend is `XFORMERS` | **`TRITON_ATTN`, forced** by vLLM for heterogeneous heads |
| `VLLM_ATTENTION_BACKEND` selects a backend | **Not a recognized vLLM env var** in v0.27 — silently ignored |
| w4a16 would need Marlin kernels wanting sm80+ | vLLM compiles **sm75 Marlin kernels** (`sm75_kernel_float16_u4b8_float16.cu.o`) |
| T4G has 16 GB | **15360 MiB** |
| `g5g.xlarge` cannot stage 9.5 GiB of weights | **still untested** — not measured here either |

## Reproduce

1. Launch `g5g.4xlarge` spot on the ARM64 GPU PyTorch DLAMI, instance profile with
   `AmazonSSMManagedInstanceCore`.
2. `cuda-keyring` → `apt install cuda-toolkit-13-2`; `rustup`; `pip install setuptools_rust`.
3. `git clone` vLLM, `git checkout v0.27.2rc0`, `python use_existing_torch.py`.
4. Apply the Turing shared-memory clamp above.
5. `TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=12 pip install -v --no-build-isolation -e .`
6. `vllm.entrypoints.openai.api_server --dtype float16 --kv-cache-dtype auto`.
