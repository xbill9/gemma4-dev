# The Cheapest CUDA GPU on AWS Has an Arm CPU. You Probably Want the Intel One.

*Subtitle: Two AWS instance families put the same Turing GPU behind different host CPUs. One is cheaper per hour and the other is cheaper per token, and the reason is not the CPU.*

This article provides a step by step deployment of Gemma 4 E2B onto the two cheapest whole GPU
CUDA instances AWS sells, and compares what they cost to run. Both carry the same generation of
NVIDIA Turing silicon. One has a Graviton2 host and one has an Intel host. A suite of Python MCP
tools manages the vLLM deployment, and every number here was measured on 2026-08-30.

The code is here:

github.com/xbill9/gemma4-dev

## Two Instances, One GPU Generation

Pricing every NVIDIA instance type in us-east-1 from the AWS Pricing API returns 64 types.
Sorted by on demand price, and keeping only those that give you a whole GPU rather than a
fractional slice, the top of the list is short:

| rank | instance | host CPU | $/hr on-demand | $/hr spot | GPU | VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | g5g.xlarge | Graviton2 (arm64) | **0.4200** | **0.1458** | T4G | 15,360 MiB |
| 2 | g4dn.xlarge | Intel (x86_64) | 0.5260 | 0.3559 | T4 | 15,360 MiB |
| 3 | g5g.2xlarge | Graviton2 | 0.5560 | — | T4G | 15,360 MiB |
| 4 | g4dn.2xlarge | Intel | 0.7520 | — | T4 | 15,360 MiB |
| 5 | g6.xlarge | AMD | 0.8048 | 0.7033 | L4 | 22,888 MiB |

The three instances cheaper than g5g.xlarge are all fractional L4 slices at 2,861 to 5,722 MiB,
and none of them can map Gemma 4 E2B's 10.2 GB checkpoint. The cheapest slice that could,
g6f.4xlarge at 11,444 MiB, costs 0.95 per hour.

So the cheapest real CUDA GPU on AWS is an Arm box, and it is 20 percent cheaper per hour on
demand and 59 percent cheaper on spot. That is the headline number, and it is the wrong one to
buy on.

## The GPUs Are the Same Part

Read off the running instances rather than the spec sheets:

| property | T4 on g4dn | T4G on g5g |
| --- | --- | --- |
| compute capability | 7.5 | 7.5 |
| VRAM, nvidia-smi | 15,360 MiB | 15,360 MiB |
| memory clock | 5,001 MHz | 5,001 MHz |
| bus width | 256 bit | 256 bit |
| theoretical peak bandwidth | 320.1 GB/s | 320.1 GB/s |
| GPU KV cache allocated by vLLM | 329,579 tokens | 329,579 tokens |

Every measurable property matches, down to vLLM independently arriving at a KV cache of exactly
329,579 tokens on both. Whatever separates these two deployments, it is not the accelerator.

## The Host CPU Decides Which Kernels You Get

Here is the mechanism, and it has nothing to do with how fast either CPU runs.

The vllm/vllm-openai image publishes one manifest list with two platforms, and they are not
compiled for the same GPUs. Reading TORCH_CUDA_ARCH_LIST out of each platform's config blob
straight from the registry:

```
linux/amd64  sha256:2286e8533ca8
  TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 8.9 9.0 10.0 12.0    sm_75 present

linux/arm64  sha256:2a7cde230b59
  TORCH_CUDA_ARCH_LIST=8.0 8.7 8.9 9.0 10.0 11.0 12.0   sm_75 absent
```

Same tag, same day. Only the amd64 image carries SM 7.5.

The host architecture selects the manifest. An Intel host pulls kernels that run on its T4. A
Graviton2 host pulls an image with no kernels for its own GPU, and the Dockerfile sets no +PTX,
so there is not even a JIT fallback. That rig has to compile vLLM from source before it can
serve one token.

This is a packaging decision by the vLLM project, not a property of either CPU. It is also the
single largest cost difference between the two families.

## The Arm Box Also Has Half the RAM

The second trap is host memory. G4dn gives you 4 GiB per vCPU and G5g gives you 2:

| instance | vCPU | host RAM |
| --- | --- | --- |
| g4dn.xlarge | 4 | 16 GiB |
| g5g.xlarge | 4 | 8 GiB |

Gemma 4 E2B's checkpoint is 9.54 GiB. On g5g.xlarge, with about 7.5 GiB usable, the kernel
declines to map it and vLLM crash loops before a single page is faulted in:

```
RuntimeError: unable to mmap 10246621918 bytes from model.safetensors:
Cannot allocate memory (12)
```

That is a failure of the mapping, not of residency, and it is fixed with a swapfile. But it
means the cheapest CUDA instance on AWS needs configuration the next one up does not, and that
configuration costs boot time on every launch.

g4dn.xlarge has 16 GiB and maps the checkpoint with no swapfile at all.

## Deploying on the Intel Box

The rig ships a single file MCP server registered as gpu-vllm-g4dn-2b, started over stdio. Every
tool is prefixed with the rig name, so calls stay unambiguous with several rigs loaded.

At this point you should have G family spot quota, a Hugging Face token in Secrets Manager, and
a subnet, security group and instance profile. The profile needs only
AmazonSSMManagedInstanceCore plus Secrets Manager read. All administration runs over SSM Run
Command, so there is no inbound SSH rule and no private key.

Quota is not capacity. G family spot in us-east-1 has been exhausted in every availability zone
but one with quota to spare, and the zone with capacity was the most expensive. Check placement
scores before launching in a loop:

```
$ aws ec2 get-spot-placement-scores --region us-east-1 --instance-types g4dn.xlarge \
    --target-capacity 1 --single-availability-zone --region-names us-east-1

use1-az1  1
use1-az2  3
use1-az4  3
use1-az5  3
use1-az6  3
```

use1-az4 maps to us-east-1c, which scored 3 and carried the lowest spot price. Launch there:

```
create_g4dn_instance
  subnet_id=subnet-0c2872fe4182b9ec1
  security_group_id=sg-01ee54036d37aa770
  iam_instance_profile=<profile>
  instance_type=g4dn.xlarge
  spot=true

Launching i-050dca2ed568dcc1b (g4dn.xlarge, spot, 1x T4) in us-east-1.
AMI: ami-0216c4aa131462acf
```

The AMI is never hardcoded. It resolves from SSM Parameter Store at launch, and returned the
Deep Learning Base OSS Nvidia Driver GPU AMI on Ubuntu 26.04, built two days before this run.
The base DLAMI is used rather than the PyTorch one, because the deployment serves from a
container carrying its own CUDA and torch.

## 195 Seconds to a Serving Endpoint

```
[stage] image-pull-start          +0s
[stage] image-pull-done         +155s
[stage] patch-applied           +176s
[stage] image-build-done        +178s
[stage] patch-verified-in-image +193s
[stage] serving-started         +195s
[stage] INSTALL_COMPLETE        +195s
```

Nothing is compiled. The image derivation is 23 seconds of that timeline, because the kernels
are already correct and exactly one pure Python file is replaced.

That one file exists because Gemma 4 has two attention geometries: 28 sliding attention layers
at head dimension 256 and 7 full attention layers at 512, verified against the safetensors
headers. Only FlashAttention-4 and Triton handle heterogeneous head dimensions, FA4 is
unavailable on Turing, so vLLM forces the Triton backend. Its tile at head size 512 wants 98,304
bytes of shared memory per block against Turing's 65,536 hard limit, and 49,152 static. The
patch clamps that one path from 32 tiles to 16, taking it to 49,152 bytes, and leaves the other
three paths alone.

Turing has no bfloat16 and no fp8 datapath, so the deployment runs float16. Worth being precise
about why: bfloat16 does not fail on Turing, it upconverts, and vLLM logs the cast and proceeds.
float16 is correct because it is what executes.

Engine startup after that: 30.03 seconds to download weights, 23.33 to load the 9.54 GiB
checkpoint, and 150.90 seconds of engine initialisation of which 82.39 is compilation.

## Verifying Before Believing

The architecture gap and the shared memory ceiling are independent problems, so both get checked.

```
$ verify_gpu_arch i-050dca2ed568dcc1b

Tesla T4, 7.5, 15360 MiB
torch arch list: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
shared mem per block (static): 49152
fp16 matmul ok: True
```

```
$ verify_triton_patch i-050dca2ed568dcc1b

--- image ---             PATCHED IMAGE PRESENT
--- module in image ---   CLAMP PRESENT
--- running container --- vllm-openai:v0.28.0-sm75-patched
```

The third line matters as much as the first two. An image can be correctly patched while the
container runs the stock tag, and everything else still reports healthy.

The health check uses chat completions rather than raw completions, which returns an empty body
on instruction tuned models. It also carries a degeneracy check, which is not a quality metric:
a broken deploy on this lineage once answered with a repeating colon-ok pattern, sixteen tokens,
non empty and completely wrong. Testing for a non empty response would have passed it.

## What It Serves

The benchmark client runs on the instance against localhost, so no network latency enters the
measurement. 512 input tokens, 128 output tokens, end of sequence ignored, request count scaling
at four times concurrency, prompts sized with vLLM's own tokenize endpoint.

| concurrency | output tok/s | per-stream tok/s | TTFT p50 ms | TPOT p50 ms |
| --- | --- | --- | --- | --- |
| 1 | 42.36 | 42.77 | 53 | 23.38 |
| 4 | 140.80 | 35.88 | 95 | 27.87 |
| 8 | 242.47 | 31.08 | 130 | 32.18 |
| 16 | 243.93 | 31.10 | 4,304 | 32.15 |
| 32 | 242.67 | 30.87 | 12,749 | 32.39 |

Five cells, all measured, every request successful and every request exactly 128 output tokens.

The engine saturates at concurrency 8, which is the configured max_num_seqs. Above that,
throughput is flat to within 0.6 percent while median time to first token rises from 130
milliseconds to 12.7 seconds. Concurrency past max_num_seqs buys latency, not throughput.

The Arm deployment, measured on g5g.4xlarge, reaches 28.65 tok/s at concurrency 1, 97.48 at
concurrency 4, and 168.33 at concurrency 8 where it also saturates.

## Cheapest Per Hour Is Not Cheapest Per Token

This is the whole point of the article.

| deployment | $/hr on-demand | tok/s at c=8 | $/M output tokens |
| --- | --- | --- | --- |
| g4dn.xlarge, Intel, T4 | 0.5260 | 242.47 | **0.603** |
| g5g.4xlarge, Graviton2, T4G | 0.8280 | 168.33 | 1.366 |

The Arm family owns the cheapest hourly rate on AWS for a whole CUDA GPU. The Intel family
delivers the cheaper token, by 2.3 times on these two deployments, because it converts its hour
into 44 percent more output.

Within the Intel box, the operating point matters more than the instance choice does:

| operating point | tok/s | spot 0.3559/hr | on-demand 0.526/hr |
| --- | --- | --- | --- |
| saturation, c=8 | 242.47 | **0.408 per M** | 0.603 per M |
| single stream, c=1 | 42.36 | 2.334 per M | 3.449 per M |

Compute only, excluding EBS and data transfer. Serving one stream at a time costs 5.7 times more
per token on identical hardware. Get your concurrency to max_num_seqs before you shop for a
cheaper instance, because batching is a larger lever than the instance family is.

## What the Hourly Rate Does Not Price

The two rates above are not comparable as operating costs, because they buy different amounts of
work before you serve anything.

On the Intel box a launch costs an image pull and a 23 second image derivation, reaching a
serving endpoint in 195 seconds. On the Arm box the published image has no kernels for the GPU
in the instance, so a launch costs a from source vLLM build first, and the instance also needs a
swapfile configured before the checkpoint will map.

That difference compounds on spot, which is where the Arm hourly discount is largest. Spot
instances get reclaimed. A reclaimed Intel instance costs an image pull and a model download to
replace. A reclaimed Arm instance costs a build. The architecture with the cheaper hour is the
one that pays the most to come back, and on spot those are the same decision.

If you want the cheapest CUDA hour on AWS, rent the Arm box. If you want the cheapest CUDA
token, rent the Intel one.

## Tearing Down

Termination is cheap on the Intel side precisely because nothing was compiled. A relaunch costs
an image pull, a 23 second derivation and the model download.

After terminating, confirm nothing billable remains: non terminated instances, open or active
spot requests, and orphaned volumes. All three returned zero.

## Summary

The goal of this article was to find out which of the two cheapest whole GPU CUDA instances on
AWS is actually the better value for serving Gemma 4 E2B. The key to the solution was that both
carry the same Turing GPU, so the host CPU architecture is the only real variable, and what it
changes is not speed but which container image you are able to run.

The results were:

The cheapest whole GPU CUDA instance on AWS is g5g.xlarge at 0.4200 per hour on demand and
0.1458 on spot, with a Graviton2 host. g4dn.xlarge with an Intel host is 20 percent more per
hour on demand and 144 percent more on spot.

The GPUs are the same part. Identical compute capability, VRAM, memory clock, bus width and
theoretical bandwidth, and vLLM independently allocated a KV cache of exactly 329,579 tokens on
both.

The vLLM container ships SM 7.5 kernels in its amd64 manifest and not in its arm64 manifest, so
the Intel host runs the published image while the Graviton2 host must build from source. The
host CPU decides what you can run, not how fast it runs.

The Arm instance also carries half the host RAM per vCPU, and at the xlarge size cannot map the
9.54 GiB checkpoint without a swapfile.

The Intel deployment reaches a serving endpoint 195 seconds after launch and saturates at 242.47
output tokens per second at concurrency 8. The Arm deployment saturates at 168.33.

Cost per million output tokens is 0.603 on the Intel box against 1.366 on the Arm box at on
demand rates, so the instance with the higher hourly rate produces the cheaper token by 2.3
times. On spot the Intel figure is 0.408 per million.

Batching is a bigger lever than the instance family. Single stream costs 2.334 per million
against 0.408 at saturation on the same hardware, a factor of 5.7.

One instance per family, one region, one model size, one sweep with no repeats. The Arm figures
come from a g5g.4xlarge run using vLLM 0.27.2rc1 built from source and the vllm bench serve
tool, while the Intel figures come from a g4dn.xlarge running vLLM 0.28.0 from the published
image under this rig's own harness. These are properties of these two deployments on their
measurement dates rather than a survey of what either family does in general.

Any opinions in this article are those of the individual author and may not reflect the opinions
of AWS.
