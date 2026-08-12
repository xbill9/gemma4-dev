# Turing + aarch64: the packaging gap, and the real blocker

**Rewritten 2026-08-12 after running this on hardware.** The original version of this file
predicted the wrong obstacle. Both halves below are now measured, not inferred; the run is in
`benchmarks/runs/2026-08-12-first-serve-g5g/`.

**Bottom line: Gemma 4 E2B serves on a T4G at ~43 tok/s, and getting there needed one patch
to vLLM — but not for the reason this document originally gave.**

## Part 1 — the packaging gap is real, and AWS already solved it

G5g needs **aarch64 and SM 7.5 together**. Most published CUDA artifacts cover one axis:

| Artifact | 7.5 on arm64? | Status |
| --- | :---: | --- |
| `vllm/vllm-openai:v0.27.1` arm64 | **no** (`8.0 8.7 8.9 9.0 10.0 11.0 12.0`) | current; amd64 of the same tag *does* carry 7.5 |
| NGC `nvcr.io/nvidia/pytorch` arm64 ≤ 24.10 | **yes** | frozen — 24.12 onward dropped it |
| `drikster80/vllm-aarch64-openai` | **yes** | abandoned Sept 2024, vLLM 0.6.1, too old for Gemma 4 |
| **AWS ARM64 GPU DLAMI** | **yes** | **actively maintained, PyTorch 2.2 → 2.12** |

Reproduce the image-config reads with:

```bash
docker buildx imagetools inspect vllm/vllm-openai:v0.27.1 --format '{{json .Image}}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); [print(k, [e for e in v['config']['Env'] if 'ARCH_LIST' in e]) for k,v in d.items()]"
```

**The original worst case here was wrong.** This file used to warn that PyTorch's aarch64
wheels might also lack 7.5, which would force a from-source PyTorch build. That is true of
upstream PyPI wheels and of NGC after 24.10 — but **AWS builds Turing into their aarch64
PyTorch**, measured on two separate DLAMIs:

```
torch 2.7.0+cu128   arch_list ['sm_75','sm_90','sm_100','sm_120']
torch 2.12.0+cu132  arch_list ['sm_75','sm_80','sm_90','sm_100','sm_110','sm_120']
```

So PyTorch never needs building, and the NGC pin is unnecessary. Only vLLM's own kernels are
compiled, with `TORCH_CUDA_ARCH_LIST=7.5`. CMake accepts it: `CUDA target architectures: 7.5`.

Two things the DLAMI does *not* give you, both discovered the hard way:

- **No `nvcc`.** The PyTorch DLAMI ships driver and torch only. Install `cuda-toolkit-13-2`
  from NVIDIA's **sbsa** repo.
- **No Rust.** vLLM needs `setuptools_rust` plus a toolchain for its `vllm-rs` frontend.

## Part 2 — the actual blocker: Turing's shared memory vs Gemma 4's 512-wide heads

With the packaging solved, the build succeeded and the server still would not start:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 98304, Hardware limit: 65536
```

vLLM says why it is on that code path:

```
Gemma4 model has heterogeneous head dimensions
{'sliding_attention': 256, 'full_attention': 512}. FA4 not available,
forcing TRITON_ATTN backend.
```

The chain, and every link is required:

1. Gemma 4's attention layers have **different head dims** — sliding 256, **global 512**
   (`MODELS.md` records the same split).
2. Only **FA4** or **TRITON_ATTN** support heterogeneous head dims.
3. FA4 is unavailable, so vLLM **forces Triton**. This is not overridable —
   `VLLM_ATTENTION_BACKEND` is not even a recognized variable in v0.27
   (`Unknown vLLM environment variable detected`), and setting it to
   `FLEX_ATTENTION` changed nothing.
4. Triton's unified-attention kernel at `head_size=512` needs **~96 KiB** of shared memory
   per block.
5. **Turing has 64 KiB per block.** Ampere and later have 164 KiB+.

So this is the intersection of *this model* and *this chip*, not a packaging problem at all.
Any model with 512-wide heads hits it on any pre-Ampere GPU.

### The patch

Clamp the KV tile until Q + K/V tiles fit, and drop the pipeline to one stage — pre-Ampere
only, so it is a no-op everywhere else. In
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

With it: CUDA graphs capture, engine init takes 76 s, and the model serves.

**This patch is not upstream and is not applied by anything in this rig.** It exists only on
the instance that was built. It is the single thing between stock vLLM and a working G5g
deployment, and it is the obvious contribution back to vLLM. Until it lands, any rebuild has
to reapply it.

## Version constraints, measured

- **vLLM must be ≥ v0.27.2rc0.** v0.26.0 fails on Gemma 4 against current `transformers` with
  `AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute`. The
  `per_layer_config` handling landed in v0.27.2rc0; v0.27.1 does not have it.
- **No vLLM tag pins torch 2.12** (2.11 → 2.13). `use_existing_torch.py` plus
  `--no-build-isolation` builds against the DLAMI's torch with only a CMake warning.
- **Build ~67 min** on `g5g.4xlarge`, `MAX_JOBS=12`. **FlashAttention 2 and 3 compile
  regardless of `TORCH_CUDA_ARCH_LIST`** and dominate it — FA2 needs sm80, FA3 needs sm90, so
  neither can ever load on 7.5. Constraining `VLLM_FA_CMAKE_GPU_ARCHES` should cut this
  sharply; untested.

## Turing dtype facts, corrected

- **bf16 is not a hard failure.** PyTorch runs bf16 on Turing by upconverting; a bf16 matmul
  returned cleanly. vLLM logs `Casting torch.bfloat16 to torch.float16` and proceeds.
  `float16` is still correct because it is what actually executes — but the earlier claim
  that bfloat16 "fails outright here" was wrong, and a wrong reason invites someone to test
  torch, see it pass, and delete the guard.
- **fp8 has no datapath**; `--kv-cache-dtype auto` stands.
- **w4a16 is not obviously blocked.** The earlier claim that it "would land on Marlin kernels
  that want SM 8.0+" is contradicted by the build, which compiled
  `sm75_kernel_float16_u4b8_float16.cu.o` — vLLM ships Turing-specific Marlin kernels.
  Untested here, but it is not ruled out.
- FlashInfer's sampler is genuinely unavailable: `unsupported compute capability 7.5;
  falling back`.

## Why CUDA 13 is not a problem

CUDA 13.0 dropped Maxwell, Pascal and Volta. **Turing survived as the new floor.** The DLAMI
runs CUDA 13.2 against a 7.5 device without complaint.

## Sources

- All hardware figures: `benchmarks/runs/2026-08-12-first-serve-g5g/REPORT.md`
- [vLLM `docker/Dockerfile`](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile) — `torch_cuda_arch_list` ARG, no-`+PTX` comment
- [CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-toolkit-release-notes/index.html)
- [ARM64 DLAMI](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-graviton-pytorch.html)
- [Amazon EC2 G5g](https://aws.amazon.com/ec2/instance-types/g5g/)
