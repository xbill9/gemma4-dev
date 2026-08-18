# gpu-vllm-g5g-2b

Serve **`google/gemma-4-E2B-it`** with **vLLM** on **AWS EC2 G5g** — an AWS Graviton2
(64-bit Arm) host paired with an **NVIDIA T4G Tensor Core** GPU.

The rig ships a single-file FastMCP server exposing a devops agent that provisions G5g
capacity with boto3, brings up the model server, and does SRE diagnostics against the
endpoint.

| | |
| --- | --- |
| Model | `google/gemma-4-E2B-it` (reference bf16 release — no encoding slot in the name) |
| Runtime | vLLM, OpenAI-compatible API on `:8000` |
| Host | AWS Graviton2, `aarch64` |
| GPU | NVIDIA T4G — Turing, **SM 7.5**, 16 GB |
| Default size | `g5g.2xlarge` (1 GPU, 8 vCPU, 16 GiB RAM) |
| Region | `us-east-1` |

Authoritative values live in [`tpu.env`](tpu.env). The directory name describes; the env
file decides.

## Read this before deploying

**No prebuilt CUDA artifact covers aarch64 *and* SM 7.5 at once.** `vllm/vllm-openai:v0.27.1`
publishes both platforms, and the arch lists differ in exactly the one place that matters:

| Manifest | `TORCH_CUDA_ARCH_LIST` | SM 7.5? |
| --- | --- | :---: |
| `linux/amd64` | `7.5 8.0 8.6 8.9 9.0 10.0 12.0` | **yes** |
| `linux/arm64` | `8.0 8.7 8.9 9.0 10.0 11.0 12.0` | **no** |

The arm64 list targets the ARM+NVIDIA parts that actually ship — A100, Jetson Orin, GH200,
Blackwell. T4G is not among them, and vLLM's Dockerfile adds no `+PTX`, so there is no JIT
fallback. Deploying the published image gives `no kernel image is available for execution on
the device`.

So this rig **builds vLLM from source for SM 7.5 on the instance** (`serving='build'`, the
default), which takes hours on a Graviton2. Full write-up, reproduction command, and the one
layer still unverified: [`docs/turing-aarch64-gap.md`](docs/turing-aarch64-gap.md).

**Start with `verify_gpu_arch`.** It answers in minutes what the build path takes hours to
discover.

## Turing constraints

T4G is Turing, not Ada. It has **no bf16 and no fp8**. The `gpu-vllm-l4-*` sibling rigs and
the legacy `~/gemma4-tips-aws` tree hardcode `--dtype bfloat16` and `--kv-cache-dtype fp8`;
both are wrong here, and the first is a hard failure rather than a slow path. This rig uses
`--dtype float16`, `--kv-cache-dtype auto`, and the `XFORMERS` attention backend.

E2B fits comfortably regardless: `MODELS.md` puts it at 9.5 GiB of weights (8.97 measured)
against 16 GB of GPU memory, leaving roughly 4.5 GiB for KV at 18 KiB/token.

## Instance sizes

| Size | GPUs | GPU mem | vCPU | RAM | |
| --- | --- | --- | --- | --- | --- |
| `g5g.xlarge` | 1 | 16 GB | 4 | 8 GB | **rejected** — cannot stage 9.5 GiB of weights |
| `g5g.2xlarge` | 1 | 16 GB | 8 | 16 GB | default |
| `g5g.4xlarge` | 1 | 16 GB | 16 | 32 GB | |
| `g5g.8xlarge` | 1 | 16 GB | 32 | 64 GB | |
| `g5g.16xlarge` | 2 | 32 GB | 64 | 128 GB | `--tensor-parallel-size 2` |
| `g5g.metal` | 2 | 32 GB | 64 | 128 GB | `--tensor-parallel-size 2` |

## Setup

```bash
./init.sh                        # verify AWS identity, install deps, register the MCP server
```

or, to register into another project:

```bash
make init TARGET=/path/to/project ARGS='--region us-east-1'
```

Then restart Claude Code and check `gpu-vllm-g5g-2b` under `/mcp`.

## Tools

| Tool | |
| --- | --- |
| `verify_gpu_arch` | **run this first** — device capability, torch arch list, real matmul |
| `get_deployment_config` | cloud-init + CLI launch command, changes nothing |
| `create_g5g_instance` | launch one tagged G5g on the latest arm64 DLAMI (spot by default) |
| `get_build_progress` | tail the from-source SM 7.5 build |
| `list_g5g_instances` / `start_` / `stop_` / `terminate_` | lifecycle |
| `get_endpoint` / `verify_model_health` / `query_model` | endpoint and serving checks |
| `get_vllm_logs` | container logs over SSM |
| `check_g5g_quotas` | G-family on-demand and spot vCPU quotas |
| `save_hf_token` | store the HF token in Secrets Manager |
| `get_help` | resolved config and the constraints behind it |

Provisioning needs explicit subnet, security-group, and instance-profile ids — the rig
creates no network or IAM policy of its own. Remote administration goes over SSM Run
Command, so no inbound SSH rule and no key pair are required.

## Development

```bash
make test     # offline unittest suite — no AWS, no network, no GPU
make lint     # ruff + bash -n
make skill    # regenerate the skill snapshots (never hand-edit them)
```

## Status

**Nothing has been provisioned and this rig carries no measurements.** `benchmarks/` holds
the synced schema and README only. The L4 artifacts that came with the directory it was
scaffolded from were deleted rather than left to be misattributed to T4G hardware.
