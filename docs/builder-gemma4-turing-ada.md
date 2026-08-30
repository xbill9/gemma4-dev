Title: g5g vs g6 for LLM Serving: the Same Code, and 3.7x the Throughput

Description: Serving Gemma 4 E2B in pure JAX on AWS g5g.2xlarge and g6.2xlarge with a byte-identical payload. The older instance family loses 87 percent of decode to dtype conversion, and nothing in any log, metric or health check reports it.

Cover: builder-cover.jpg (1200x675, no text, uploaded in the editor)

---
<!-- Everything above this rule goes in the Title, Description and cover fields.
     Paste only what follows into the body. -->

This article compares two AWS GPU instance families for serving a small language model, using a payload that is byte-identical on both. The older family loses 87 percent of decode to dtype conversion, and nothing in any log, metric or health check says so.

The code is here: https://github.com/xbill9/gemma4-dev

## The Two Instances

The g5g.2xlarge pairs a Graviton2 (aarch64) host with an NVIDIA T4G, a Turing part at SM 7.5. The g6.2xlarge is x86_64 with an NVIDIA L4, an Ada part at SM 8.9. Both were run on spot capacity.

| | g5g.2xlarge | g6.2xlarge |
|---|---|---|
| GPU | NVIDIA T4G, Turing, SM 7.5 | NVIDIA L4, Ada, SM 8.9 |
| GPU memory | 15,360 MiB | 23,034 MiB |
| Host | Graviton2, aarch64 | x86_64, us-east-1d |
| Purchase model | spot | spot |
| Run cited | 2026-08-28-full-run-cached-g5g | 2026-08-28-first-serve-g6 |

The workload is google/gemma-4-E2B-it, the dense reference checkpoint, served through a hand-written pure-JAX port. No PyTorch, no vLLM, no torch_xla.

The payload is byte-identical on both instances: build id 51bc52c9e2e9, config ple4 plus int8_lm_head, and tpu_jax_weight_bytes reading 6,155,450,950 on each. Only the chip and its host differ.

## At This Point You Should Have

- Spot capacity for g5g.2xlarge and g6.2xlarge
- A Hugging Face token with access to google/gemma-4-E2B-it
- A Deep Learning base AMI matching the host architecture, since the aarch64 and x86_64 images are not interchangeable
- No CUDA toolkit and no Rust toolchain, because jax[cuda13] supplies CUDA as wheels

## Install Is a pip Install

There is no build step on either instance. The jax[cuda13] extra ships wheels carrying CUDA, including aarch64 wheels for the Graviton2 host.

```
Install: 117 s, with the cache restore included
```

XLA's persistent compilation cache is pushed to S3 and restored on boot. On the g5g rig it restores 805 files and 12 MB in 6 seconds onto a fresh instance, from a box that had already been terminated.

## Warm Up Before Measuring Anything

The max_new_tokens argument is a static_argnames entry, so the pair (bucket, max_tokens) is the compiled shape. The first request off a fresh engine pays XLA compilation.

On the g5g that first request took 18.06 seconds against 4.50 seconds warm, a 4.0x whole-request ratio, measured in 2026-08-21-cuda13-py314-g5g. A harness that skips warm-up misreports the instance by a factor of four.

The 56x figure quoted from the earlier first-serve baseline is time-to-first-token specifically. That is a different measurement and is not interchangeable with the whole-request ratio.

## The Throughput Sweep

64 output tokens, concurrency 1, three repeats per cell, median reported. The gauge is the engine's steady-state decode counter. End-to-end is wall time over the whole request including prefill.

| Input tokens | g5g gauge | g5g end-to-end | g6 gauge | g6 end-to-end |
|---:|---:|---:|---:|---:|
| 41 | 12.9 tok/s | 12.43 tok/s | 48.5 tok/s | 46.23 tok/s |
| 521 | 13.0 tok/s | 11.28 tok/s | 48.4 tok/s | 42.87 tok/s |
| 2,057 | 12.9 tok/s | 8.22 tok/s | 48.3 tok/s | 34.57 tok/s |
| 3,593 | not measured | not measured | 48.3 tok/s | 27.55 tok/s |

That is 3.7x on decode, for the same code and the same weights.

## Read the Gauge, Not End-To-End

Decode moves 0.8 percent across a 50x context range on the g5g and 0.4 percent on the g6. End-to-end falls hard on both, from 12.43 to 8.22 and from 46.23 to 27.55.

That fall is prefill being linear in the padded bucket, not decode degrading. They are two different claims, and conflating them makes a benchmark misleading.

A cost proportional to the weights rather than the context produces exactly this shape, which is why the KV cache is not what sets decode speed on either instance.

Usable context on the g5g is MAX_MODEL_LEN=4096, and that is the honest number. 4,105 prompt tokens serve; 5,120 fails on a prefill transient.

## Where the g5g Decode Actually Went

Profiling with xprof, 20 decode steps with the service stopped:

| | g5g, T4G, SM 7.5 | g6, L4, SM 8.9 |
|---|---:|---:|
| dtype conversion | 54.1% | 0.0% |
| fp32 gemvx | 32.8% | absent |
| Tensor Core | 0.0% | 0.0% |
| Total kernel time | 1,466.0 ms | 362.8 ms |
| Decode, gauge | 12.9 tok/s | 48.4 tok/s |
| Peak HBM bandwidth | 298.083 GiB/s | 279.441 GiB/s |
| Share of bandwidth roofline | 26% | approximately 100% |

87 percent of decode on the g5g is not math. It is dtype conversion plus an fp32 gemvx path. The instance runs at 26 percent of its own memory-bandwidth roofline, while the g6 runs at roughly all of it.

## Why This Is Invisible

A wrong compute dtype does not raise an error. It emulates. Requesting bfloat16 on a pre-Ampere GPU does not fail, because XLA routes it through fp32, and decode quietly disappears into conversion.

Turing has neither bf16 nor fp8. Its only real 16-bit datapath is float16. So the port reads the live compute capability off the device rather than trusting a configuration file:

```python
COMPUTE_DTYPE = float16 if IS_PRE_AMPERE else bfloat16
```

The server states its decision on the first line it emits, so a misconfigured instance is one grep away rather than a mystery in the throughput:

```
INFO ports.gemma4.jax_e_model: jax_e_model device policy: platform=gpu
compute_capability=8.9 compute_dtype=bfloat16 pallas_interpret=False
```

## The Checkpoint Was Not the Problem

The obvious hypothesis was bf16 weights on a chip with no bf16 datapath, so the checkpoint was converted to float16 on the host and the profile was re-run. Parameter dtypes then read float16 for 541 arrays, uint8 for one and int8 for one, and conversion stayed at 54.0 percent.

Storage dtype was never the problem. The fp32 gemvx line is the tell: XLA was round-tripping through fp32 regardless of what the file on disk said. Only a card whose compute dtype matches its storage dtype removes that cost, which is what the g6 demonstrates.

The measurement reproduces. The same profile on a different instance, a different AMI and a restored cache landed at 1466.0 ms against 1467.1 ms.

## A Health Check Will Not Catch This

The g5g serves correctly the entire time. It returns HTTP 200, valid completions, and a healthy health endpoint. It is simply doing four times more work than it needs to.

The related trap in this engine is a padding-eviction bug in the KV ring cache, whose failure mode is a token loop returning a clean HTTP 200 with a success status and repeated single-token output. Nothing in the logs or the metrics is red. Only a degeneracy check on the response body catches it, which the server now runs on every request.

On this stack, HTTP 200 is not evidence of anything.

## Teardown

Both instances are spot and are terminated after collection. The XLA cache is pushed to S3 first, which is what makes the six-second restore on the next fresh instance possible.

## Summary

The goal of this article was to compare two AWS GPU instance families for serving a small language model with a payload held byte-identical across both. The key to the solution was profiling decode rather than trusting throughput alone. The measured results were:

- 3.7x decode throughput on g6.2xlarge over g5g.2xlarge, 48.4 against 12.9 tok/s, for the same code and the same weights.
- 87 percent of decode on the g5g is dtype conversion and an fp32 path, against 0.0 percent on the g6.
- The g5g runs at 26 percent of its memory-bandwidth roofline; the g6 at roughly 100 percent.
- Nothing surfaces this. The g5g serves valid completions with a healthy endpoint throughout.
- Tensor Core utilization is 0.0 percent on both instances and is not yet explained.

Scope: two spot instances, one in us-east-1d, each measured once with three repeats per sweep cell and medians reported. The two differ in host architecture, aarch64 against x86_64, and in base image as well as in GPU, so this is not a single-variable experiment. The payload is byte-identical across them, being build 51bc52c9e2e9, the same configuration and the same 6,155,450,950 bytes of weights, which is the basis for attributing the difference to the chip. The g5g profile was reproduced on a second instance at 1466.0 ms against 1467.1 ms, while the g6 profile was measured once. Price and price per token were not measured and are not claimed here.

Any opinions in this article are those of the individual author and may not reflect the opinions of AWS.
