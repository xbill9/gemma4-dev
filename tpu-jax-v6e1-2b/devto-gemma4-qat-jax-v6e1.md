---
title: "Deploying a QAT Checkpoint Your Serving Stack Can't Load: Gemma 4 E2B in Pure JAX on One TPU"
published: false
description: "vLLM on TPU cannot load the Gemma 4 E2B QAT exports. So I served them from a hand-rolled JAX engine on a single v6e chip — and found that the thing that actually governs latency is not the model at all."
tags: tpu, jax, llm, quantization
---

*Cloud TPU v6e-1 (`ct6e-standard-1t`, one v6e chip, 32 GB HBM), Compute Engine flex-start, europe-west4-a. All timings below measured 2026-08-19 unless stated otherwise.*

There is a particular kind of dead end where every component is healthy, every version is current, and the thing still does not work. You have a quantized checkpoint the vendor publishes. You have a serving stack the vendor supports. They do not load each other.

That is where this started. What I did not expect is that the interesting part would turn out to have nothing to do with quantization.

### The dead end

Gemma 4 E2B ships QAT variants. The obvious move is to serve one under vLLM on TPU, which is a supported, documented path. Both variants refuse, in different ways, on `vllm/vllm-tpu:nightly`:

```plaintext
-qat-w4a16-ct
  compressed-tensors scheme for layer 'per_layer_model_projection'
  is not yet supported in the JAX path

-qat-q4_0-unquantized
  weights not initialized from checkpoint:
  layers.15-34 self_attn.k_norm.weight
```

Two different failures with one shape: the checkpoint and the loader disagree about what is in the file. The first is a missing quantization scheme. The second is an export that genuinely omits `k_norm` for the upper KV-sharing layers, so the loader is right to complain.

**Neither is a configuration problem, and that matters** — because the reflex is to try another flag. There is no flag. You either wait for the scheme to land upstream, serve the unquantized model instead, or write the load path yourself.

I wrote the load path. This article is what that cost and what it taught me.

### QAT is not compression, and the suffix is not decoration

Worth being precise, because it changes what you expect.

Quantization-aware training simulates quantization *during* training rather than compressing a finished model afterwards. The int4 weights are not a post-hoc approximation of a bf16 model — the model was trained knowing it would be stored this way. That is why a QAT int4 checkpoint holds quality that a naive round-to-int4 of the same model would not.

The suffixes are not interchangeable, and I have watched people assume they are:

| Suffix | What it is | Where it belongs |
| :--- | :--- | :--- |
| `-qat-w4a16-ct` | 4-bit weights, 16-bit activations, compressed-tensors | the engine here decodes this |
| `-qat-q4_0-unquantized` | QAT-trained weights shipped **at half precision** | faster on a 32 GB chip, larger on disk |
| `-gguf` | llama.cpp | not TPU |
| `-mobile-ct` | on-device runtimes | not TPU |

The one to stare at is `-q4_0-unquantized`. It is a QAT checkpoint that has been *dequantized* for distribution: 10.21 GB of bf16 rather than 3.52 GB of int4, carrying QAT-trained values. It is the faster of the two on this hardware — 10.1 vs 8.1 tok/s in this repo's earlier v6e-1 measurements — because nothing has to unpack int4 on the fly.

Which one you want is a memory question, and the answer inverts with the chip.

### What fits 32 GB

Component sizes measured with `jax.devices()[0].memory_stats()`:

| Component | `-w4a16-ct` | `-q4_0-unquantized` |
| :--- | ---: | ---: |
| Model weights | 3.52 GB | 10.21 GB |
| MXU activations & XLA buffers | 1.50 GB | 1.50 GB |
| libtpu & XLA runtime reserve | 2.00 GB | 2.00 GB |
| **Fixed subtotal** | **7.02 GB** | **13.71 GB** |
| Left for KV (of ~31.2 GiB usable) | ~24.2 GB | ~17.5 GB |

A 128K-token fp8 KV cache measures 2.40 GB. So on a v6e chip both fit comfortably and you should take the faster one; `-q4_0-unquantized` still leaves about seven 128K contexts of headroom.

On a 16 GB chip — v5e — that same table reads the other way: the bf16 variant leaves ~3.0 GB, barely one 128K context and nothing for batch, and the int4 variant becomes the only sane choice. **The recommendation is not a property of the checkpoint. It is a property of the chip**, and I have seen the v5e answer copied onto v6e hardware where it costs 20% of throughput for no reason.

Live check on the deployed server, int4 variant:

```plaintext
tpu_jax_hbm_used_bytes{device="TPU_0(process=0,(0,0,0,0))"} 6967732736
```

6.97 GB against a predicted 7.02 GB fixed subtotal. The budget is real.

### Getting the chip

Provisioning is Compute Engine, not the Cloud TPU API — that API is no longer under active development and TPU7x onward is Compute Engine or GKE only. I have written that migration up separately; the short version for this rig:

```shell
gcloud compute instances create tpu-jax-v6e1-2b \
  --zone=europe-west4-a \
  --machine-type=ct6e-standard-1t \
  --provisioning-model=FLEX_START \
  --request-valid-for-duration=2h \
  --max-run-duration=4h \
  --instance-termination-action=DELETE \
  --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
  --image-project=ubuntu-os-accelerator-images \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB \
  --scopes=cloud-platform \
  --metadata-from-file=startup-script=startup.sh
```

Flex-start went `PENDING → STAGING → RUNNING` in 105 seconds from request to booted VM, at $1.35/chip-hour — cheaper than spot on this generation, which surprises people who assume spot is always the budget option. One data point in one zone on one day, though; flex-start queues for up to two hours by design and I have no claim that this is typical. Before queueing I fired a throwaway spot create at the same zone: spot does not queue, so it fails fast and names the reason, which turns "is my request stuck on quota or on capacity" into a five-second question rather than a two-hour one.

**`--boot-disk-size=200GB` is load-bearing** and I want to put a number on it. After the environment and the int4 checkpoint landed, the disk sat at 14 GB used. The image default is 10 GB. You do not get a disk-space error at create time; you get a clean boot, then a failure part-way through a model download, which is a long way from the flag that caused it.

### The bare-JAX VM

No Docker, no Hugging Face token at boot, no 200 GB image pull. The startup script installs a current CPython and `jax[tpu]`, and there are four things in it that are not obvious:

**Ubuntu 22.04's system Python is 3.10, and that silently pins JAX to an old release.** Left alone you get JAX 0.6.x. The script installs 3.13 from deadsnakes. No virtualenv — the dedicated interpreter is the isolation.

**`libtpu` does not come from PyPI.** It resolves from the JAX releases index, so the install needs `-f https://storage.googleapis.com/jax-releases/libtpu_releases.html`. The script also upgrades `libtpu` independently afterwards, deliberately overriding the conservative pin that `jax[tpu]` carries.

**The readiness check asserts on `jax.devices()`, not on the import.** Importing `jax` succeeds on a host with no TPU backend at all. An import-only check reports a healthy environment on a machine with no accelerator, which is the worst possible outcome — it fails later, somewhere else, wearing a different mask.

**The `set -x` trace leak.** The script traps errors and prints a `FAILED` marker. With tracing on, the shell echoes the trap *definition* before running anything — and that echoed line contains the literal `FAILED` string. A log scanner then reports failure on a perfectly healthy boot. Install the trap before enabling `set -x`, and skip lines beginning with `+` when scanning.

Result:

```console
$ verify_jax_tpu
jax 0.11.1 | libtpu 0.0.46
devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
```

One more thing that bit me on the way: **the serving dependencies are not in the JAX extras.** The bare VM has `jax`, `transformers`, `safetensors`, `huggingface_hub`. It does not have `fastapi`, `uvicorn` or `pydantic`, because those are the *server's* dependencies and the dev VM has no opinion about servers. Obvious in hindsight; a confusing `ModuleNotFoundError` at the time.

The Hugging Face token for the gated repo comes from Secret Manager at run time, via the instance's `cloud-platform` scope, so it is never written into instance metadata or onto disk.

### It serves

```json
{"status":"ok","backend":"jax","device":"TPU_0(process=0,(0,0,0,0))",
 "model":"google/gemma-4-E2B-it-qat-w4a16-ct",
 "precision":{"weights":"w4_int4","activations":"bfloat16","kv_cache":"bf16"}}
```

Real QAT int4 weights, decoded in a hand-rolled JAX engine, OpenAI-compatible on port 8000. That is the goal met.

And then the latency made no sense.

### The part I got wrong

First request: 20296 ms prefill. Second: 20130 ms. Third: 6.7 ms.

My first read was that a new *prompt length* triggers recompilation — the engine uses statically shaped, bucket-padded caches, so that is a reasonable guess. It is also wrong, and it took a controlled test to see it, because in those three requests I had been changing two things at once.

Fixing `max_tokens` and varying only the prompt:

```plaintext
prompt_tokens=  14   prefill_ms= 19474.4     <- first request
prompt_tokens=  14   prefill_ms=     6.6     <- same prompt again
prompt_tokens=  28   prefill_ms=     7.0     <- DIFFERENT length, still fast
prompt_tokens=  14   prefill_ms=     6.6
```

A 28-token prompt hit the warm path on its first appearance. Prompt length was not the key.

The variable I had been changing without noticing was `max_tokens`. Holding the prompt constant and varying only that:

```plaintext
max_tokens=8   prompt_tokens=14   prefill_ms=     7.1     <- seen before
max_tokens=9   prompt_tokens=14   prefill_ms= 19902.3     <- NEW value, full compile
max_tokens=9   prompt_tokens=14   prefill_ms=     6.6     <- now warm
max_tokens=9   prompt_tokens=25   prefill_ms=     7.0     <- prompt length irrelevant
max_tokens=8   prompt_tokens=14   prefill_ms=     6.5     <- old value still cached
```

**Every distinct `max_tokens` value costs one ~20-second compilation.** Prompt length is free. Compiled variants cache and do not evict one another.

The engine says so outright, once you know to look:

```python
static_argnames=("model", "max_new_tokens", "quant_mode", "cache_dtype", "window_kv")
```

`max_new_tokens` is a **static** argument to `jax.jit`. Static means it participates in the cache key, so every distinct value is a distinct compiled program. This is not a bug — it is what lets the generation loop unroll to a fixed length — but its cost is paid at a place nobody is looking, on the first request that happens to ask for 129 tokens instead of 128.

### Prompt length is free until it isn't

The prompts are free because they are padded to static buckets. Sweeping the length at fixed `max_tokens`:

```plaintext
prompt_tokens=   61   prefill_ms=     7.0
prompt_tokens=  111   prefill_ms= 19618.3
prompt_tokens=  211   prefill_ms= 22508.3
prompt_tokens=  261   prefill_ms=     9.8
prompt_tokens=  511   prefill_ms= 26237.7
prompt_tokens= 1211   prefill_ms= 28898.3
prompt_tokens= 3011   prefill_ms= 30932.0
```

Read that carefully: 61 fast, 111 slow, 211 slow, **261 fast**, 511 slow. A *longer* prompt was cheaper than a shorter one, which looks like nonsense until you see the buckets — 261 and 511 share a bucket, and the 511 request had already compiled it.

The engine again confirms it directly:

```python
static_sequence_buckets = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
```

Every measurement fits: 14/25/61 → 64, 111 → 128, 211 → 256, 261/511 → 512, 1211 → 2048, 3011 → 4096. Black-box measurement and implementation agree, which is the only version of this I would publish — I have been wrong once already in this article.

Compilation time also grows with bucket size, 19.6 s at 128 up to 30.9 s at 4096, which stands to reason. And once compiled, longer prompts do cost real prefill work: a warm 3011-token request runs 93.8 ms against 8.6 ms at 211 tokens. That part is honest arithmetic rather than a caching artifact.

### What this means if you actually deploy it

The compile matrix is **buckets × distinct `max_tokens` values**. Eight buckets, and as many `max_tokens` values as your clients invent.

That second term is the problem. An OpenAI-compatible client sends whatever `max_tokens` the application feels like — 100, 150, 256, 1000, whatever a config file says this week. Each novel value stalls one request for twenty seconds. Under concurrency that is not one slow request; it is every request queued behind a compilation.

Three mitigations, in the order I would reach for them:

1. **Round `max_tokens` server-side** to a small set — 32/64/128/256/512 — and generate to the rounded length. You cap the matrix at 8 × 5, and clients never see it.
2. **Warm the grid at startup**, before the server accepts traffic. Forty compilations at ~25 s is about seventeen minutes of boot, which is only acceptable if you are baking a disk image; it is why staging a GCE image with a populated compilation cache is worth the trouble.
3. **Persist the XLA compilation cache** to disk and reuse it across restarts. This server already sets `jax_compilation_cache_dir` and `jax_persistent_cache_min_compile_time_secs=0`, so restarts inherit the work — but a fresh flex-start VM starts cold every time, which is exactly the deployment shape here.

None of that is quantization. **I set out to solve a checkpoint-loading problem and the thing that actually governs tail latency turned out to be a jit cache key.**

### What I did not measure

Warm decode came out at 129.8 tok/s on a short single-stream request. **Do not compare that to the 8.1 tok/s figure earlier in this article** — different harness, different prompt, different output length, single request, no concurrency. It is not the same measurement wearing a bigger number.

More importantly, I have not cross-checked it against an absolute physical bound. The rule I try to hold myself to is that a benchmark should be checked against bytes-moved-per-second on calibrated bandwidth, not against another configuration of the same code, because an assumption shared by both sides of an A/B is invisible to the A/B. Until I do that, 129.8 tok/s means "the server works", not "the server is fast", and it is not going in any results table.

Two other honest gaps. This engine has known open issues around bucket padding interacting with the sliding-window attention, and the test suite — which is mostly parity assertions between two of the project's own code paths — passes straight through them. A shared wrong assumption is invisible to a parity test by construction. And I ran all of this on one chip with one stream; nothing here says anything about concurrency, which is where a serving path usually breaks.

### Troubleshooting quick reference

| Symptom | Cause | Check |
| :--- | :--- | :--- |
| `compressed-tensors scheme ... not supported` | vLLM cannot load this QAT export | not a flag; use the JAX path |
| `weights not initialized: k_norm` | the `-q4_0` export omits k_norm on KV-sharing layers | not a flag; use the JAX path |
| Import of `jax` succeeds, no TPU | import does not imply a backend | assert on `jax.devices()` |
| JAX resolves to 0.6.x | system Python is 3.10 | install 3.13; don't use the system interpreter |
| `libtpu` not found on PyPI | it ships from the JAX releases index | `-f .../libtpu_releases.html` |
| Healthy boot reports FAILED | `set -x` echoed the trap definition | install the trap before `set -x`; skip `+` lines |
| `ModuleNotFoundError: fastapi` | server deps are not JAX deps | install them separately |
| Model download dies mid-way | 10 GB default boot disk | `--boot-disk-size=200GB` |
| Random 20 s stalls in production | novel `max_tokens` → new jit variant | round `max_tokens`; warm the grid |
| A longer prompt is faster than a shorter one | static sequence buckets | it is the bucket, not the length |
| Instance RUNNING, nothing serving | RUNNING means the VM booted | read the serial log; curl the port |

### The short version

If your serving stack cannot load a checkpoint the vendor publishes, that is sometimes a genuine gap rather than a flag you have not found, and writing the load path is a real option — the QAT format is documented and the weights are ordinary safetensors.

But budget your attention correctly. The load path took the effort I expected. The behaviour that would have hurt in production was a twenty-second compile on the first request to use an unfamiliar `max_tokens`, sitting behind a jit decorator, invisible to every health check, and impossible to see at all if you change two variables at once — which I did, and then believed the wrong explanation until a controlled test contradicted it.

Measure one thing at a time, then go and read the source to find out whether the machine agrees with you.

* * *
