---
title: "The Cheapest CUDA GPU on AWS Has an Arm CPU — and You Probably Want the Intel One"
published: false
description: "Gemma 4 E2B on the two cheapest whole-GPU CUDA instances AWS sells. Same Turing GPU, different host CPU. One is cheaper per hour, the other is cheaper per token, and the reason is not the CPU."
tags: aws, vllm, cuda, machinelearning
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-vllm-g4dn-2b/devto-cover.jpg
---

This article provides a step by step deployment guide for Gemma 4 E2B onto the two cheapest
whole GPU CUDA instances AWS sells, and compares what they cost to run. A suite of Python MCP
tools is built to simplify management of the vLLM hosted deployment. Everything was measured on
2026-08-30.

## What is this project trying to Do?

The question is simple: if you want a CUDA GPU on AWS as cheaply as possible, which one do you
rent?

Two instance families sit at the bottom of the price list, and they carry the same generation of
NVIDIA Turing silicon. G5g pairs a T4G with a Graviton2 host on aarch64. G4dn pairs a T4 with an
Intel host on x86_64. The GPUs are effectively the same part. The host CPU is the variable.

The answer turns out to depend entirely on whether you are buying hours or tokens.

## Where do I start?

Start with price. Every NVIDIA instance type in us-east-1 was priced from the AWS Pricing API
rather than from documentation:

```
aws pricing get-products --region us-east-1 --service-code AmazonEC2 \
  --filters Type=TERM_MATCH,Field=instanceType,Value=g4dn.xlarge \
            Type=TERM_MATCH,Field=location,Value="US East (N. Virginia)" \
            Type=TERM_MATCH,Field=operatingSystem,Value=Linux \
            Type=TERM_MATCH,Field=tenancy,Value=Shared \
            Type=TERM_MATCH,Field=preInstalledSw,Value=NA \
            Type=TERM_MATCH,Field=capacitystatus,Value=Used
```

Sixty four types came back. Keeping only those that give you a whole GPU rather than a
fractional slice:

| Rank | Instance | Host CPU | $/hr On-Demand | $/hr Spot | GPU | VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| 🥇 | `g5g.xlarge` | Graviton2 arm64 | 0.4200 | 0.1458 | T4G | 15,360 MiB |
| 🥈 | `g4dn.xlarge` | Intel x86_64 | 0.5260 | 0.3559 | T4 | 15,360 MiB |
| 🥉 | `g5g.2xlarge` | Graviton2 | 0.5560 | — | T4G | 15,360 MiB |
| | `g4dn.2xlarge` | Intel | 0.7520 | — | T4 | 15,360 MiB |
| | `g6.xlarge` | AMD x86_64 | 0.8048 | 0.7033 | L4 | 22,888 MiB |

**The cheapest real CUDA GPU on AWS is an Arm box.** It is 20 percent cheaper per hour on demand
and 59 percent cheaper on spot.

Three instances are cheaper still, at 0.2020, 0.2375 and 0.4750, but all three are fractional L4
slices with 2,861 to 5,722 MiB, and none of them can map Gemma 4 E2B's 10.2 GB checkpoint. The
cheapest slice that could, `g6f.4xlarge` at 11,444 MiB, costs 0.95 per hour.

So the headline is the Arm box. It is also the wrong number to buy on, and the rest of this
article is why.

## The GPUs Are the Same Part

Read off the running instances rather than the spec sheets:

| Property | T4 on G4dn | T4G on G5g |
| --- | --- | --- |
| Compute capability | 7.5 | 7.5 |
| VRAM, `nvidia-smi` | 15,360 MiB | 15,360 MiB |
| Memory clock | 5,001 MHz | 5,001 MHz |
| Bus width | 256 bit | 256 bit |
| Theoretical peak bandwidth | 320.1 GB/s | 320.1 GB/s |
| GPU KV cache allocated by vLLM | 329,579 tokens | 329,579 tokens |

Every measurable property matches, down to vLLM independently arriving at a KV cache of exactly
329,579 tokens on both. Whatever separates these deployments, it is not the accelerator.

## The Host CPU Decides Which Kernels You Get

This is the mechanism, and it has nothing to do with how fast either CPU runs.

`vllm/vllm-openai` publishes one manifest list with two platforms, and they are not compiled for
the same GPUs. Read straight out of the registry config blobs:

```
linux/amd64  sha256:2286e8533ca8
  TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 8.9 9.0 10.0 12.0    sm_75 present

linux/arm64  sha256:2a7cde230b59
  TORCH_CUDA_ARCH_LIST=8.0 8.7 8.9 9.0 10.0 11.0 12.0   sm_75 absent
```

Same tag, same day. **Only the amd64 image carries SM 7.5.**

The host architecture selects the manifest. An Intel host pulls kernels that run on its T4. A
Graviton2 host pulls an image with no kernels for its own GPU, and the Dockerfile sets no `+PTX`,
so there is not even a JIT fallback. That rig compiles vLLM from source before it serves a
token.

This is a packaging decision by the vLLM project, not a property of either CPU, and it is the
single largest cost difference between the two families.

## The Arm Box Also Has Half the RAM

G4dn gives you 4 GiB per vCPU. G5g gives you 2:

| Instance | vCPU | Host RAM |
| --- | --- | --- |
| `g4dn.xlarge` | 4 | 16 GiB |
| `g5g.xlarge` | 4 | 8 GiB |

Gemma 4 E2B's checkpoint is 9.54 GiB. On `g5g.xlarge`, with about 7.5 GiB usable, the kernel
declines to map it and vLLM crash loops before a single page is faulted in:

```
RuntimeError: unable to mmap 10246621918 bytes from model.safetensors:
Cannot allocate memory (12)
```

That is a failure of the mapping, not of residency, and a swapfile fixes it. But the cheapest
CUDA instance on AWS needs configuration the next one up does not. `g4dn.xlarge` has 16 GiB and
maps the checkpoint with no swapfile at all.

## At this point you should have

- An AWS account with G family spot quota — 16 vCPU here, and `g4dn.xlarge` needs 4
- A Hugging Face token in Secrets Manager under `vllm/hf-token`
- A subnet, a security group, and an instance profile with `AmazonSSMManagedInstanceCore`
- No inbound SSH rule and no key pair — everything runs over SSM

## Setup the Basic Environment

```
git clone https://github.com/xbill9/gemma4-dev
cd gemma4-dev/gpu-vllm-g4dn-2b
pip install -r requirements.txt
```

The requirements are small:

```
mcp
httpx
boto3
```

The Hugging Face token is fetched at boot and never placed in user data, because instance
metadata is readable by anything on the box. The fetch is wrapped in `set +x`, because the
bootstrap runs under `set -x` and bash traces assignments with their values.

## MCP stdio Transport

The MCP server is a single Python file started over stdio. The standard MCP libraries abstract
the transport, so the tool implementations are identical no matter which client connects.

Registration lives in four places and all four must name the server identically: `.mcp.json`,
the plugin manifest, `.codex/config.toml`, and `enabledMcpjsonServers`. A rename that updates
three of four leaves an approval gate naming a tool that does not exist, and a gate on a tool
name that does not exist fails open and says nothing.

```
./project-setup.sh --server-name gpu-vllm-g4dn-2b
```

## Model Lifecycle Management via MCP

Quota first:

```
check_g4dn_quotas
```

```
| Running On-Demand G and VT instances (vCPU) | 16 |
| All G and VT Spot Instance Requests (vCPU) | 16 |
g4dn.xlarge needs 4 vCPUs.
```

Quota is not capacity. G family spot in us-east-1 has been exhausted in every AZ but one with
quota to spare, and the one AZ with capacity was the most expensive. Price is not a proxy for
availability:

```
aws ec2 get-spot-placement-scores --region us-east-1 --instance-types g4dn.xlarge \
  --target-capacity 1 --single-availability-zone --region-names us-east-1
```

```
use1-az1  1
use1-az2  3
use1-az4  3
use1-az5  3
use1-az6  3
```

`use1-az4` maps to `us-east-1c`, which scored 3 and carried the lowest spot price at 0.3559.

## Deploy The Model

```
create_g4dn_instance
  subnet_id=subnet-0c2872fe4182b9ec1
  security_group_id=sg-01ee54036d37aa770
  iam_instance_profile=<profile>
  instance_type=g4dn.xlarge
  spot=true
```

```
✅ Launching i-050dca2ed568dcc1b (g4dn.xlarge, spot, 1x T4) in us-east-1.
AMI: ami-0216c4aa131462acf
Patch sha: 26b1cead19f4 → vllm-openai:v0.28.0-sm75-patched
```

The AMI is never hardcoded. It resolves from SSM at launch and returned the Deep Learning Base
OSS Nvidia Driver GPU AMI on Ubuntu 26.04, built two days before this run. The base DLAMI is
used rather than the PyTorch one, because the deployment serves from a container carrying its
own CUDA and torch.

## Checking Install Progress

```
get_install_progress i-050dca2ed568dcc1b
```

```
[stage] image-pull-start          +0s
[stage] image-pull-done         +155s
[stage] patch-applied           +176s
[stage] image-build-done        +178s
[stage] patch-verified-in-image +193s
[stage] serving-started         +195s
[stage] INSTALL_COMPLETE        +195s
```

**195 seconds, and nothing is compiled.** The image derivation is 23 seconds of that, because
the kernels are already correct and exactly one pure Python file is replaced.

That file exists because Gemma 4 has two attention geometries — 28 sliding attention layers at
head dimension 256 and 7 full attention layers at 512, verified against the safetensors headers.
Only FA4 and Triton handle heterogeneous head dims, FA4 is unavailable on Turing, so vLLM forces
`TRITON_ATTN`. Its tile at head size 512 wants 98,304 bytes of shared memory per block against
Turing's 65,536 hard limit, and 49,152 static. The patch clamps that one path from 32 tiles to
16, and leaves the other three alone.

Turing has no bfloat16 and no fp8 datapath, so the deployment runs float16. Be precise about
why: bfloat16 does not fail on Turing, it upconverts, and vLLM logs the cast and proceeds.
float16 is correct because it is what executes.

## Verify the GPU Architecture

Two checks, and passing the first says nothing about the second. The arch gap and the shared
memory ceiling are independent problems, and the Intel host only deletes the first:

```
verify_gpu_arch i-050dca2ed568dcc1b
```

```
Tesla T4, 7.5, 15360 MiB
capability: (7, 5)
torch arch list: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
shared mem per block (static): 49152
fp16 matmul ok: True
```

The probe uses float16, not bfloat16. A bfloat16 probe would pass by upconversion and tell you
nothing about what executes.

## Verify the Turing Patch

```
verify_triton_patch i-050dca2ed568dcc1b
```

```
--- image ---             PATCHED IMAGE PRESENT
--- module in image ---   CLAMP PRESENT
--- running container --- vllm-openai:v0.28.0-sm75-patched
```

The third line matters as much as the first two. An image can be correctly patched while the
container runs the stock tag, and everything else still reports healthy.

## Checking System status

```
GPU KV cache size: 329,579 tokens, Maximum concurrency for 16,384 tokens per request: 20.12x
```

| Phase | Time |
| --- | --- |
| Weights download | 30.03 s |
| Checkpoint load, 9.54 GiB | 23.33 s |
| Model loading total | 55.75 s, 9.8 GiB |
| Engine init | 150.90 s, 82.39 s of it compilation |
| CUDA graph capture | 13 s, 0.16 GiB |

## Cross Check The Deployed Model

The health check uses `/v1/chat/completions`. Raw `/v1/completions` skips the chat template and
returns an empty body on `-it` models, so an empty response there is not evidence either way.

It also carries a degeneracy check, and that is not a quality metric. A broken deploy on this
lineage once answered `': ok: ok: ok…'` — sixteen tokens, non empty, completely wrong. Testing
for a non empty response would have passed it.

What the deployed model actually returned:

> This is a **description of a data collection process** for performance metrics. Here's a
> breakdown of what the text tells us: * **What is being measured:** Throughput and latency.

## Benchmark the Local Model

The client runs on the box against localhost, so no network sits between it and the engine under
test. 512 input tokens, 128 output, `ignore_eos`, request count scaling at four times
concurrency, prompts sized with vLLM's own `/tokenize` so input length is a model token count
and not a word count guess.

```
python3 benchmarking_suite.py --url http://127.0.0.1:8000 \
  --contexts 512 --concurrencies 1,4,8,16,32 \
  --output-tokens 128 --prompts-per-concurrency 4
```

| Concurrency | Output tok/s | Per-stream tok/s | TTFT p50 ms | TPOT p50 ms |
| --- | --- | --- | --- | --- |
| 1 | 42.36 | 42.77 | 53 | 23.38 |
| 4 | 140.80 | 35.88 | 95 | 27.87 |
| 8 | 🥇 242.47 | 31.08 | 130 | 32.18 |
| 16 | 243.93 | 31.10 | 4,304 | 32.15 |
| 32 | 242.67 | 30.87 | 12,749 | 32.39 |

Five cells, all measured, every request successful, every request exactly 128 output tokens.

The engine saturates at concurrency 8, which is `--max-num-seqs 8`. Above that, throughput is
flat to within 0.6 percent while median TTFT rises from 130 milliseconds to 12.7 seconds.
**Concurrency past your `max-num-seqs` buys latency, not throughput.**

## Compare to Other Deployments

The Arm deployment, measured on `g5g.4xlarge`:

| Concurrency | G4dn Intel T4 | G5g Graviton2 T4G |
| --- | --- | --- |
| 1 | 🥇 42.36 | 28.65 |
| 4 | 🥇 140.80 | 97.48 |
| 8 | 🥇 242.47 | 168.33 |
| 16 | 🥇 243.93 | 169.96 |
| 32 | 🥇 242.67 | 170.99 |

Both saturate at concurrency 8. The Intel box converts its hour into 44 percent more output, and
it does that on a quarter of the host — 4 vCPU against 16.

## Cost Analysis

Cheapest per hour and cheapest per token are different boxes. That is the finding:

| Deployment | $/hr On-Demand | tok/s at c=8 | $/M Output Tokens |
| --- | --- | --- | --- |
| 🥇 `g4dn.xlarge` Intel T4 | 0.5260 | 242.47 | **0.603** |
| 🥈 `g5g.4xlarge` Graviton2 T4G | 0.8280 | 168.33 | 1.366 |

The Arm family owns the cheapest hourly rate on AWS for a whole CUDA GPU. The Intel family
delivers the cheaper token, by 2.3 times.

Within the Intel box, the operating point matters more than the family choice does:

| Operating point | tok/s | Spot 0.3559/hr | On-Demand 0.526/hr |
| --- | --- | --- | --- |
| Saturation, c=8 | 242.47 | 🥇 0.408 per M | 0.603 per M |
| Single stream, c=1 | 42.36 | 2.334 per M | 3.449 per M |

Compute only, excluding EBS and data transfer. **Serving one stream at a time costs 5.7 times
more per token on identical hardware.** Get your concurrency to `max-num-seqs` before you shop
for a cheaper instance — batching is a bigger lever than the instance family is.

## And Price/Performance?

The two hourly rates are not comparable as operating costs, because they buy different amounts
of work before you serve anything.

On the Intel box a launch costs an image pull and a 23 second derivation, reaching a serving
endpoint in 195 seconds. On the Arm box the published image has no kernels for the GPU in the
instance, so a launch costs a from source build first, and the `xlarge` also needs a swapfile
before the checkpoint will map.

That compounds on spot, which is where the Arm discount is largest. Spot instances get
reclaimed. A reclaimed Intel instance costs an image pull and a model download to replace. A
reclaimed Arm instance costs a build. **The architecture with the cheaper hour is the one that
pays most to come back**, and on spot those are the same decision.

If you want the cheapest CUDA hour on AWS, rent the Arm box. If you want the cheapest CUDA
token, rent the Intel one.

## Tear Down

Termination is cheap on the Intel side precisely because nothing was compiled:

```
terminate_g4dn_instance i-050dca2ed568dcc1b
```

Then confirm nothing billable is left — instances, spot requests, orphaned volumes:

```
aws ec2 describe-instances --region us-east-1 \
  --filters "Name=instance-state-name,Values=pending,running,shutting-down,stopping,stopped" \
  --query 'length(Reservations[].Instances[])' --output text
aws ec2 describe-volumes --region us-east-1 \
  --filters Name=status,Values=available --query 'length(Volumes)' --output text
```

```
0
0
```

## Summary

The cheapest whole GPU CUDA instance on AWS is `g5g.xlarge` at 0.4200 per hour on demand and
0.1458 on spot, with a Graviton2 host. `g4dn.xlarge` with an Intel host costs 20 percent more
per hour on demand and 144 percent more on spot.

The GPUs are the same part — identical compute capability, VRAM, memory clock, bus width and
bandwidth, and vLLM independently allocated a KV cache of exactly 329,579 tokens on both.

What the host CPU changes is not speed but which container image runs at all. vLLM ships SM 7.5
kernels in its amd64 manifest and not in its arm64 manifest, so the Intel host runs the
published image while the Graviton2 host must build from source. The Arm instance also carries
half the RAM per vCPU and cannot map the 9.54 GiB checkpoint at the `xlarge` size without swap.

Measured, the Intel deployment saturates at 242.47 output tokens per second against 168.33, and
costs 0.603 per million output tokens against 1.366. The instance with the higher hourly rate
produces the cheaper token by 2.3 times, and on spot it reaches 0.408 per million.

Scope: one instance per family, one region, one model size, one sweep with no repeats. The Arm
figures come from a `g5g.4xlarge` running vLLM 0.27.2rc1 built from source under `vllm bench
serve`; the Intel figures from a `g4dn.xlarge` running vLLM 0.28.0 from the published image
under this rig's harness. These are properties of these two deployments on their measurement
dates.

The strategy for using MCP for Gemma 4 deployment on AWS EC2 G4dn was validated with an
incremental step by step approach.
