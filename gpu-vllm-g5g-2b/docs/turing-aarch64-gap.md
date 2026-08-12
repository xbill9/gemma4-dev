# The SM 7.5 + aarch64 gap

**Verified 2026-08-12 against `vllm/vllm-openai:v0.27.1` as published.**

G5g is the only instance family here that needs **aarch64 and Turing at the same
time**. Every prebuilt CUDA artifact in the ecosystem covers one of those axes or
the other. This rig exists in the gap between them, and that fact — not anything
about Gemma 4 — is what shapes its whole deployment path.

## The measurement

`vllm/vllm-openai:v0.27.1` publishes both platforms in one manifest list. Reading
the image config for each (no pull, no GPU, no instance) gives:

| Manifest | `NVARCH` | `TORCH_CUDA_ARCH_LIST` | SM 7.5? |
| --- | --- | --- | :---: |
| `linux/amd64` | `x86_64` | `7.5 8.0 8.6 8.9 9.0 10.0 12.0` | **yes** |
| `linux/arm64` | `sbsa` | `8.0 8.7 8.9 9.0 10.0 11.0 12.0` | **no** |

Reproduce with:

```bash
docker buildx imagetools inspect vllm/vllm-openai:v0.27.1 --format '{{json .Image}}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); [print(k, [e for e in v['config']['Env'] if 'ARCH_LIST' in e]) for k,v in d.items()]"
```

**The one arch this rig needs is the only one the two images disagree about.**
7.5 is present on amd64 and absent on arm64. The arm64 list is Ampere and up:
8.0 is A100, 8.7 is Jetson Orin, 9.0 is GH200, 10.0/11.0/12.0 are Blackwell —
the parts that actually ship as ARM+NVIDIA systems. T4G is not among them.

## There is no PTX fallback

Normally a missing SM target degrades to JIT-compiling from embedded PTX. Not
here. vLLM's `docker/Dockerfile` says so explicitly:

```
# Do not add +PTX here: vLLM filters torch's top-level PTX flag when it
# converts global gencode flags into per-kernel arch lists. If a specific
# kernel needs PTX, add +PTX to that kernel's CMake arch list instead.
ARG torch_cuda_arch_list='7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0'
```

So an unsupported device fails hard, with `no kernel image is available for
execution on the device`, rather than running slowly.

Note the **Dockerfile default includes 7.5**. The arm64 image does not lack it
because the source cannot build it — the release pipeline overrides the ARG per
platform (`# From versions.json: .torch.cuda_arch_list`). That is what makes
`serving='build'` viable: pass the ARG back.

## The second layer, and the one thing not verified first-hand

vLLM's own kernels are only half the stack. It installs PyTorch from
`download.pytorch.org/whl/cu130`, and PyTorch's **aarch64/SBSA CUDA wheels are
built for 9.0 / 10.0 / 12.0** — also no 7.5. That is consistent with vLLM's arm64
arch list (both target the Grace-Hopper/Blackwell ARM market), but unlike the
table above it comes from PyTorch's build matrix and release discussion rather
than from an artifact this rig inspected directly.

**So the open question is whether `serving='build'` is sufficient or merely
necessary.** Rebuilding vLLM for 7.5 fixes vLLM's kernels. If the torch aarch64
wheel has no 7.5 kernels either, ATen operations outside vLLM's own kernels will
fail the same way, and the rig would additionally need a from-source PyTorch
build — many more hours on a Graviton2.

**Do not resolve this by reasoning about it. Measure it.** `verify_gpu_arch`
exists for exactly this: it runs one container on a live G5g and prints the
device capability, `torch.cuda.get_arch_list()`, and the result of a real fp16
matmul. It answers in minutes what the build path takes hours to discover, and
it is the first thing to run on a new instance:

```
verify_gpu_arch(instance_id="i-...")                     # published arm64 image
verify_gpu_arch(instance_id="i-...", image=VLLM_IMAGE)   # after a local build
```

Record the result in `benchmarks/runs/` and update this file. Until then the
`build` path is the rig's best-supported hypothesis, not a demonstrated one.

## Turing also has no bf16 and no fp8

Separate from the packaging problem, and just as load-bearing. bf16 arrives with
Ampere (SM 8.0); fp8 with Hopper/Ada. Every L4 sibling rig in this monorepo was
written against SM 8.9 and hardcodes both:

```
--dtype bfloat16 ... --kv-cache-dtype fp8      # L4 rigs. Wrong on T4G.
--dtype float16  ... --kv-cache-dtype auto     # this rig
```

`--dtype bfloat16` on T4G is a hard failure, not a slow path. FlashAttention
likewise requires SM 8.0+, so the attention backend is `XFORMERS`.

This is the same class of error `HARDWARE.md` documents for TPU generations —
"quantization conclusions do not carry forward across generations" — showing up
on the NVIDIA side. A flag accepted by a sibling rig is not evidence it does
anything here.

## Why CUDA 13 is not an additional problem

The image is built on CUDA 13.0.2, and CUDA 13.0 dropped Maxwell, Pascal, and
Volta. **Turing survived that cut** — SM 7.5 is the floor of CUDA 13, not below
it. So the toolchain can target this GPU; only the shipped binaries do not.

## Sources

- Image config read directly from `vllm/vllm-openai:v0.27.1` (both manifests), 2026-08-12
- [vLLM `docker/Dockerfile`](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile) — the `torch_cuda_arch_list` ARG and the no-`+PTX` comment
- [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) — "compute capability 7.5 or higher"; aarch64 images must be built from source
- [CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-toolkit-release-notes/index.html) — pre-Turing architectures removed
- [Amazon EC2 G5g](https://aws.amazon.com/ec2/instance-types/g5g/) — instance sizes and T4G
