# gpu-jax-g5g-2b

Serve **`google/gemma-4-E2B-it`** with **pure JAX** on **AWS EC2 G5g** — an AWS Graviton2
(64-bit Arm) host paired with an **NVIDIA T4G Tensor Core** GPU.

"Pure JAX" is literal: no PyTorch, no torch_xla, no vLLM. The engine is this repo's own
Gemma 4 port (`ports/gemma4/`) driven by `jax_engine.py` behind an OpenAI-compatible FastAPI
server (`jax_openai_server.py`), run under **systemd — not docker**.

The rig ships a single-file FastMCP server exposing a devops agent that provisions G5g
capacity with boto3, ships the serving payload over SSM, and does SRE diagnostics against the
endpoint.

| | |
| --- | --- |
| Model | `google/gemma-4-E2B-it` (reference bf16 release — no encoding slot in the name) |
| Runtime | pure JAX + XLA-GPU, OpenAI-compatible API on `:8000` |
| Host | AWS Graviton2, `aarch64` |
| GPU | NVIDIA T4G — Turing, **SM 7.5**, **15360 MiB measured** (not the nominal 16 GB) |
| Default size | `g5g.2xlarge` (1 GPU, 8 vCPU, 16 GiB RAM) |
| Region | `us-east-1` |
| Measured | **13.10 tok/s** decode at the current default |

Authoritative values live in [`tpu.env`](tpu.env). The directory name describes; the env
file decides.

## Why this exists next to `gpu-vllm-g5g-2b`

Identical hardware, different runtime, and **the runtime is the whole point.** The vLLM path
reaches a served token only through a ~67-minute from-source build, a CUDA toolkit the DLAMI
does not ship, a Rust toolchain, and an **unlanded patch to Triton's attention kernel** that
has to be reapplied after every upgrade — because Gemma 4's heterogeneous head dims (sliding
256, global 512) force `TRITON_ATTN`, whose 512-wide tile wants ~96 KiB of shared memory
against Turing's 64 KiB ceiling.

JAX sidesteps all four: pip supplies CUDA so there is **no build**, no toolkit and no Rust,
the plugin's cubins already cover SM 7.5, and attention is ordinary XLA rather than a
hand-tiled Triton kernel — so there is no per-block shared-memory ceiling in the attention
path and no patch to carry.

`docs/turing-aarch64-gap.md` is the vLLM-side write-up and is **measured**; it is kept here
because it is the reason this rig was built.

## What JAX does not sidestep

The same 64 KiB ceiling bites elsewhere. The fused **W4A16 Pallas kernel** is tiled for TPU
VMEM (16 MB per core) and needs **550 KiB – 1.1 MiB per block** at this model's shapes. On
GPU, Pallas lowers through Triton and those tiles become shared memory, so the kernel cannot
run. `check_w4a16_fits_scoped_memory()` computes the requirement and **raises at startup with
the arithmetic attached**, rather than dying as an `OutOfResources` at the first token.

So this rig serves the **dense reference checkpoint** at float16, deliberately not the
`-qat-w4a16-ct` export the TPU JAX rig serves — and the rig name therefore carries no
encoding slot.

## Turing is not L4, and it is not v6e

**Turing has no bf16 and no fp8.** Do not copy a flag from either lineage.

| | TPU v6e-1 | L4 (SM 8.9) | this rig (SM 7.5) |
| --- | --- | --- | --- |
| compute dtype | `bfloat16` | `bfloat16` | **`float16`** |
| KV cache dtype | `bf16`/`fp8` | `fp8` | **`auto` → float16** |
| fused W4A16 Pallas | yes | — | **refused at startup** |

The **device** decides, not `tpu.env`: `jax_e_model.py` reads the live compute capability and
picks `float16` below SM 8.0. bfloat16 does not fail here, it **emulates** through fp32 — so
it gets a warning; fp8 is refused outright.

## Quickstart

```bash
pip install -r requirements.txt          # control plane only
python3 -m unittest discover -s tests -v # 105 tests, fully offline
```

Then, through the MCP tools (`mcp__gpu-jax-g5g-2b__…`):

```
create_g5g_instance → get_install_progress → verify_gpu_arch → deploy_jax_server
                    → get_jax_logs → verify_model_health → query_model → get_metrics
```

There is **no `make deploy`** on purpose: provisioning resolves an arm64 GPU DLAMI at launch
time, and a Makefile would have to hardcode one.

**Always `make skill` before `deploy_jax_server`** — the deploy ships the *skill snapshot*,
not the working tree, and the deploy output now prints the payload root and build id so a
stale deploy is visible in one line.

## Things that will bite you

- **Warm up at the shape you measure.** `max_new_tokens` is a `static_argnames` entry, so
  `(bucket, max_tokens)` is the compiled shape. Cold measured 18.77 s against 4.35 s warm for
  the same request.
- **`MAX_MODEL_LEN` is 4096 and that is the honest number.** 4,105 prompt tokens serve; 5,120
  fails on a prefill transient. It was 8192 until 2026-08-26.
- **The AMI must be arm64 *and* carry the NVIDIA driver.** AWS also ships ARM64 DLAMIs built
  for Graviton CPU inference; they boot fine here and simply have no GPU. Never hardcode an
  AMI id.
- **`g5g.xlarge` and `g5g.2xlarge` both need a swapfile**, which `_user_data` provisions.
- **Two GPUs on `16xlarge`/`metal` do nothing** — the engine is single-device.

## Documentation

| File | What |
| --- | --- |
| `CLAUDE.md` | authoritative working notes for this rig |
| `docs/turing-aarch64-gap.md` | why the vLLM path is hard here (measured) |
| `docs/larger-models-on-t4g.md` | how large a model this rig will serve |
| `docs/bf16-weights-on-turing.md` | the dtype tax: 87% of decode |
| `docs/padding-window-eviction.md` | a fixed silent-correctness bug in the shared port |
| `benchmarks/` | every measurement, schema-validated |
