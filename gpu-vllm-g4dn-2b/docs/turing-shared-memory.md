# Turing's 64 KiB, Gemma 4's 512-wide heads, and why x86_64 does not help

**Status: this rig has run nothing.** Every measurement cited below was taken on
`gpu-vllm-g5g-2b` (T4G, Turing SM 7.5, Graviton2) on 2026-08-12, or is arithmetic. What is
new here is not a measurement — it is that **G4dn separates two problems that were tangled
together on G5g**, and only one of them survives.

## The two problems, and which one x86_64 deletes

`gpu-vllm-g5g-2b` is the hardest rig in this tree because it hits both at once and has to
solve both to serve a single token.

| | Problem 1: is SM 7.5 in the image? | Problem 2: does the Triton tile fit? |
| --- | --- | --- |
| `gpu-vllm-g5g-2b` — aarch64, SM 7.5 | **NO** → ~67-minute from-source build | **NO** → tile clamp |
| `gpu-vllm-g6-2b` — x86_64, SM 8.9 | yes | yes (~99 KiB), *unverified* |
| **`gpu-vllm-g4dn-2b` — x86_64, SM 7.5** | **YES** | **NO** → tile clamp |

**Problem 1 is a property of the published binaries, not of the silicon.** Read from the
image config of `vllm/vllm-openai` — one manifest list, two platforms:

| Manifest | `TORCH_CUDA_ARCH_LIST` | SM 7.5? | SM 8.9? |
| --- | --- | :---: | :---: |
| `linux/amd64` | `7.5 8.0 8.6 8.9 9.0 10.0 12.0` | **yes** | yes |
| `linux/arm64` | `8.0 8.7 8.9 9.0 10.0 11.0 12.0` | no | yes |

Reproduce it with:

```bash
docker buildx imagetools inspect vllm/vllm-openai:v0.28.0 --format '{{json .Image}}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); [print(k, [e for e in v['config']['Env'] if 'ARCH_LIST' in e]) for k,v in d.items()]"
```

G4dn is an Intel host, so it pulls the **amd64** manifest, and that is the manifest carrying
7.5. **Same image, same tag, different answer purely because of the host architecture.** The
Dockerfile sets no `+PTX`, so on arm64 there is no JIT fallback either — which is why the G5g
rig has to rebuild rather than merely tolerate a slow path.

That deletes, for this rig: the from-source build, the `cuda-toolkit` install from NVIDIA's
sbsa repo, the Rust toolchain for `vllm-rs`, the prebuilt AMI to maintain, and the whole
`serving='build'` / `serving='stock'` mode.

**Problem 2 is untouched, because it is a property of the model and the chip.**

## Problem 2, in detail

Gemma 4's attention layers have **different head dims** — sliding **256**, global **512**
(`MODELS.md` records the same split). Only **FA4** and **TRITON_ATTN** support heterogeneous
head dims. FA4 is unavailable, so vLLM **forces Triton** and says so:

```
Gemma4 model has heterogeneous head dimensions
{'sliding_attention': 256, 'full_attention': 512}. FA4 not available,
forcing TRITON_ATTN backend.
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 98304, Hardware limit: 65536
```

**This is not overridable.** MEASURED on the G5g rig: `VLLM_ATTENTION_BACKEND` is not even a
recognized variable in vLLM v0.27 (`Unknown vLLM environment variable detected`), and setting
it to `FLEX_ATTENTION` changed nothing. So the backend is not the knob — the tile size inside
the forced kernel is.

Triton's unified-attention kernel at `head_size=512` wants **98,304 B** per block.

| | Bytes per block | 96 KiB tile fits? |
| --- | ---: | :---: |
| Turing (SM 7.5) — default static limit | **49,152** | no |
| Turing (SM 7.5) — opt-in maximum | **65,536** | **no** — 65,536 < 98,304 |
| Ada (SM 8.9) | ~101,376 | expected yes, narrow |
| Ampere (SM 8.0) and later | 164 KiB+ | yes |

**Cite the 64 KiB figure only with the qualifier.** A kernel must opt into the dynamic
shared-memory attribute to reach 65,536; the *default* static limit is 49,152, and that is
what `torch.cuda.get_device_properties().shared_memory_per_block` reports. A reader who
checks torch and sees 48 KiB will otherwise conclude this document is wrong. `verify_gpu_arch`
prints that number for exactly this reason.

**On G4dn this is arithmetic, not a margin to check.** The G6 sibling has to *find out*
whether Ada's ~99 KiB clears 96 KiB. Here 65,536 < 98,304 settles it before anything is
launched, and the same silicon has already produced the failure on a sibling.

## The remedy, and why it is cheap here

Clamp the KV tile until Q + K/V tiles fit, and drop the pipeline to one stage — pre-Ampere
only, so it is a no-op on every other device. In
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

With it, on the G5g rig: CUDA graphs capture, engine init takes 76 s, and the model serves.

**`_smem_budget` is 60000, not 65536, and that is not a rounding preference.** The tile
arithmetic does not account for the kernel's accumulators, so budgeting the hard limit still
overflows. `TURING_SMEM_BUDGET` in `tpu.env` is the knob if that ever needs tuning.

**THE DELIVERY IS WHAT IS NEW ON G4dn.** The G5g rig can only get this patch in by compiling
vLLM from source, because its image has no SM 7.5 kernels to start from and the engine has to
be rebuilt anyway — the patch is a free rider on a 67-minute build. Here the kernels are
already present and **exactly one pure-Python file is wrong**, so:

```
docker pull vllm/vllm-openai:v0.28.0
  → resolve the module's path INSIDE the image (site-packages carries the image's
    python version, so it is never hardcoded)
  → docker run --entrypoint cat  →  patch_triton_turing.py  →  patched file
  → FROM <stock>; COPY patched-file <resolved path>   →  docker build
  → verify the clamp is present IN THE BUILT IMAGE
  → serve the derived tag
```

Seconds, no compiler, no CUDA toolkit, no Rust, no source checkout.

## Why `patch_triton_turing.py` refuses instead of guessing

**A patch that silently matches nothing is worse than one that fails.** It leaves an unpatched
module behind a patched-looking tag, and the failure then surfaces ten minutes later as
`OutOfResources` at engine start — attributed to the wrong thing, which is precisely what made
the G5g diagnosis expensive.

So the script checks, and exits 2 with the surrounding source attached, when:

- any of `current_platform`, `BLOCK_M`, `head_size`, `TILE_SIZE_PREFILL`, `TILE_SIZE_DECODE`
  is absent — upstream has restructured the file and the clamp would raise `NameError`;
- no `kernel_unified_attention_{2,3}d[` launch site is found — the anchor is gone;
- the pipeline-stage variable cannot be identified uniquely.

**That last check is subtler than it looks.** The launch site reads
`num_stages=launch_num_stages`, and a line-oriented pattern reads that as an assignment to
`num_stages`. Picking it would write `num_stages = 1` into the enclosing scope, binding a
local nothing reads — **half the fix, applied silently, reported as success.** The script uses
`ast` and considers only real assignment targets. `TRITON_STAGES_VAR` overrides it.

Cloud-init runs the patch under `set -e` with an explicit `|| exit 1`, so a refusal **kills
the launch**. `get_install_progress` recognises the message and says so in as many words,
because "the patch refused" and "the install is slow" must not share a rendering.

## Verified against real upstream source, 2026-08-29

The arch-list table above was re-read from the **live `v0.28.0` manifest**, not inherited: the
amd64/arm64 split still holds at the current release.

`triton_unified_attention.py` was pulled at `v0.28.0` (**byte-identical to `main`**) and the
patch run against it. Three things came out of that, and the second one nearly shipped:

- **The clamp is still needed.** `_get_tile_size` has no shared-memory awareness whatsoever —
  it returns 32/16/32 from `head_size` and element size alone. Upstream has not fixed this.
- **The launch-site anchor was WRONG.** `kernel_unified_attention_2d`/`_3d` have been merged
  into a single `kernel_unified_attention`, so the old pattern matched nothing — and chasing
  that turned up the real problem: the tile constants are copied into `tile_size` at the
  `grid` selection, and `launch_num_stages` into `launch_kwargs`, **both before the launch**.
  A clamp at the launch site would rewrite three variables nothing reads afterwards, pass
  every verification this rig has, and leave the kernel asking for 98,304 bytes.
- **The kernel is `unified_attention`, lines 802–1189**, and the derived insertion point is
  line 997 — immediately after the `if use_td:` block that last assigns the tiles, with the
  three reads at 1079, 1082 and 1182 behind it.

**What the clamp does at Gemma 4's shapes** (fp16, `BLOCK_M=16`, budget 60000):

| layer | head | path | tile in → out | bytes in → out |
| --- | ---: | --- | --- | --- |
| `sliding_attention` | 256 | prefill / decode | 32 → 32 | 40,960 → unchanged |
| `full_attention` | 512 | decode | 16 → 16 | 49,152 → unchanged |
| **`full_attention`** | **512** | **prefill** | **32 → 16** | **81,920 → 49,152** |

Narrow by design: only the 512-wide global prefill path moves, and it lands under 65,536.

## What is still open

- **Nothing here has been RUN.** Source-level verification is not a served token: the patch
  applies cleanly to the real file, but no instance has been launched and no kernel compiled.
- **`launch_num_stages` defaults to `None`** upstream (Triton picks), and is set to 2 only on
  a B200-specific `tuned_large_head` path. The clamp forces 1. Whether that is necessary once
  the tiles fit is untested — it is kept because it is what the G5g rig ran with.
- **The clamp is not upstream.** It is the obvious contribution back to vLLM, and until it
  lands every image upgrade re-runs this patch. That is now automatic rather than manual,
  which is the one thing this rig improves on the G5g sibling's handling of it.
- **`HARDWARE.md` says the Turing-capable vLLM attention backend is `XFORMERS`.** That line
  predates the 2026-08-12 measurement showing vLLM forces `TRITON_ATTN` and ignores the
  variable. It should be corrected at the root, not restated here.

## Sources

- All Turing measurements: `gpu-vllm-g5g-2b/docs/turing-aarch64-gap.md` and
  `gpu-vllm-g5g-2b/benchmarks/runs/2026-08-12-first-serve-g5g/REPORT.md`
- Shared-memory limits and the 48/64 KiB distinction: `@HARDWARE.md`, "T4G"
- Head-dim split: `@MODELS.md`
- [vLLM `docker/Dockerfile`](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile) — `torch_cuda_arch_list` ARG, no-`+PTX` comment
- [Amazon EC2 G4dn](https://aws.amazon.com/ec2/instance-types/g4/)
