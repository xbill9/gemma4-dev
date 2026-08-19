---
title: "Serving Gemma 4 on a Turing GPU with no build at all: pure JAX instead of vLLM"
published: false
description: "The same G5g hardware, the same checkpoint, no from-source build, no CUDA toolkit, no Rust, and no kernel patch. jax[cuda12] ships aarch64 wheels that already carry sm_75. It costs 3.5x throughput."
tags: aws, jax, cuda, machinelearning
# canonical_url: https://your-blog.example/gemma4-g5g-pure-jax   # set if republished from your own site
# cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-jax-g5g-2b/devto-cover.jpg  # add devto-cover.jpg + push first
# series: "Gemma-4 on odd accelerators"
---

*A field report on serving Google's Gemma 4 E2B on AWS EC2 **G5g** — Graviton2 (aarch64)
plus an NVIDIA **T4G** (Turing, SM 7.5) — with **pure JAX** instead of vLLM. The vLLM path
on this box takes a 67-minute from-source build, a CUDA toolkit, a Rust toolchain, and an
unlanded patch to a Triton kernel. The JAX path takes a `pip install` and five minutes.
Then it serves at a third of the speed, and three separate things lie to you about whether
it is working.*

| | |
|---|---|
| Model | `google/gemma-4-E2B-it` (reference bf16 release, cast to fp16 at load) |
| Hardware | `g5g.2xlarge` spot — Graviton2 + 1x NVIDIA T4G (SM **7.5**, 15,360 MiB) |
| Base image | Deep Learning ARM64 AMI, OSS Nvidia Driver GPU PyTorch 2.7 (Ubuntu 22.04) |
| Software | jax 0.11.1 · jaxlib 0.11.1 · CUDA 12.9 **from pip** · no compiler on the box |
| Result | **12.0 tok/s** single stream · ~5-minute install · zero patches |

---

I have a rig serving Gemma 4 E2B on G5g under vLLM. Getting there took a from-source build
with `TORCH_CUDA_ARCH_LIST=7.5`, a CUDA toolkit the image does not ship, a Rust toolchain it
does not ship either, and a patch to `triton_unified_attention.py` that is not upstream and
lives on exactly one EBS volume. It serves at 43.1 tok/s. I wrote all of that up.

The obvious question afterwards was whether any of it was necessary. vLLM's problem on this
hardware is not really vLLM — it is that the fast path through it, for this model, is a
hand-tiled Triton kernel that wants more shared memory than a Turing SM has. **A framework
that does not hand-tile its attention kernel does not have that problem.** JAX lowers through
XLA, XLA picks its own tiles, and nobody has to know that 64 KiB is a number that matters.

So I built the same rig again on the same instance family with pure JAX — no PyTorch, no
torch_xla, no vLLM — and ran it. It works. **It is also 3.5x slower, and every failure on the
way there was silent.**

## jaxlib publishes what vLLM does not

The whole vLLM story on G5g starts with an arch list. `vllm/vllm-openai:v0.27.1` publishes
both platforms under one tag, and the arm64 image is compiled for `8.0 8.7 8.9 9.0 10.0 11.0
12.0` while the amd64 image of the same tag carries 7.5. There is no `+PTX`, so there is
nothing to JIT from. The one architecture G5g needs is the only entry the two images disagree
on.

JAX ships its CUDA support differently, and the difference is the entire article. `jax[cuda12]`
resolves to three wheels — `jaxlib`, `jax-cuda12-plugin`, and `jax-cuda12-pjrt` — and all
three publish `manylinux_2_27_aarch64`. You can read the arch table straight out of the
PJRT plugin binary before you launch anything:

```bash
strings xla_cuda_plugin.so | grep -oE 'sm_[0-9]+' | sort -u
```

`sm_75` is there. The floor is SM 6.0 — the plugin's own error string is
`device not supported (request SM60+)`. Turing is comfortably inside it, and nothing about
this is special-cased for Arm: the aarch64 wheel carries the same arch table as the x86 one.

That is the asymmetry. vLLM compiles its own kernels and therefore has to decide, at build
time, which architectures to ship. JAX's device support lives in one PJRT plugin that gets
built once for a wide arch range, so a hardware combination nobody markets still gets covered
by accident.

## pip supplies CUDA; the image only supplies the driver

The second thing that makes this cheap: `jax[cuda12]` does not want a CUDA installation. It
brings its own as ordinary Python wheels, and every one of them publishes aarch64:

```
nvidia-cublas-cu12    12.9.2.10
nvidia-cudnn-cu12     9.24.0.43
```

On the vLLM path I installed `cuda-toolkit-13-2` from NVIDIA's sbsa repo because the DLAMI
ships a driver and torch and no `nvcc`. Here the missing `nvcc` never comes up, because
nothing is compiled. The AMI has one job — supply a kernel driver new enough for the CUDA
runtime in the wheels — and the ARM64 GPU DLAMI does that with driver `580.126.09`.

The whole install, start to finish:

```bash
add-apt-repository -y ppa:deadsnakes/ppa
apt-get install -y python3.12 python3.12-venv python3.12-dev
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
python3.12 -m pip install --upgrade 'jax[cuda12]'
python3.12 -m pip install --upgrade fastapi uvicorn pydantic transformers \
                                    safetensors huggingface_hub numpy jinja2
```

**Just over five minutes.** Against roughly 67 minutes of `MAX_JOBS=12` compilation on the
vLLM side, most
of it spent on FlashAttention 2 and 3 kernels that need sm80 and sm90 and can therefore never
load on this GPU.

Python 3.12 is in there because jax 0.11 requires it and the Ubuntu 22.04 DLAMI base ships
3.10. Hold that thought — it is the cause of the worst failure in this article.

## What the JAX path does not skip

Here is where the neat story stops. Turing's shared-memory ceiling did not go away; it moved.

Gemma 4 also ships a QAT W4A16 export, and my TPU rig serves it through a fused Pallas matmul
kernel — unpack int4, apply per-group scales, and multiply, all inside one kernel so the
weights are read once. On TPU that kernel tiles into VMEM, which is measured in **megabytes**.

On GPU, Pallas lowers through Triton, and those tiles become shared memory, which on Turing is
measured in **kilobytes**. At this model's shapes the kernel wants **550 KiB to 1.1 MiB per
block** against a 64 KiB ceiling. It is the same wall vLLM hits with its 512-wide attention
heads, roughly an order of magnitude further past it.

So the fused path is unavailable, and the alternative — dequantize the weights to fp16 and
run a normal matmul — is worse than not quantizing at all: you pay 4x the weight traffic of
the fused path *and* you end up holding dense weights anyway. **This rig therefore serves the
dense reference checkpoint.** 9.26 GB of parameters into 15,360 MiB of device memory, which
fits, and quantization simply is not on the table on this chip.

I made the engine say so at startup rather than at the first token:

```python
raise ScopedMemoryError(
    f"fused W4A16 kernel needs ~{need/1024:.1f} KiB of shared memory per block "
    f"for seq={seq} K={k} N={out_f}, but this device allows {limit/1024:.0f} KiB"
    ". This is the pre-Ampere shared-memory ceiling, not a bug: the kernel was "
    "tiled for TPU VMEM, which is measured in megabytes."
)
```

That is a direct lesson from the vLLM build. There, the equivalent failure arrived as a Triton
`OutOfResources` *after* a 67-minute build and a full model load, and it took a while to
understand because nothing in the message mentions the model. A pre-flight check that prints
the arithmetic turns the same condition into a startup refusal you can read.

## Turing has no bf16, and that is not an error

Every published example for serving Gemma on a GPU says `bfloat16`. On L4, A10G, anything
Ampere or later, that is right. On Turing it is a trap, and specifically it is not the kind of
trap that stops you.

**bf16 does not fail on Turing. It emulates.** XLA converts through fp32, the numbers come out
correct, and every matmul quietly pays for it. There is no error, no warning you would notice,
and no obvious slowdown unless you have something to compare against. It is a performance bug
wearing a correctness bug's clothing, which is the worst combination to inherit from a config
file someone copied off an L4 rig.

So the port picks the dtype from the device rather than from config:

```python
IS_PRE_AMPERE = COMPUTE_CAPABILITY is not None and COMPUTE_CAPABILITY < (8, 0)
...
return jnp.float16 if IS_PRE_AMPERE else jnp.bfloat16
```

Ampere (8.0) is the line that matters for this whole port: bf16, fp8 groundwork, and >64 KiB
shared memory per block all arrive there. Turing is below it on all three.

fp8 gets different treatment. There is no fp8 datapath at all on Turing, so the KV-cache
resolver raises rather than silently downgrading — an fp8 KV cache is standard advice on L4
and buys exactly nothing here. `int8` is the honest way to halve KV bytes on this chip,
because it carries a per-row scale.

One more thing that does not carry across: `jax_default_matmul_precision="bfloat16"` is right
on a TPU MXU and actively wrong here, because it tells XLA it may demote fp32 matmul inputs to
a format the chip has no unit for. My serving script set it unconditionally, inherited from the
TPU rig. It is now set only when the platform is TPU.

## Three availability zones to get one instance

Before any of the software mattered, capacity did. G5g is a 2020 instance family that never
got a successor, and the spot pools show it:

| AZ | Spot price | Result |
|---|---|---|
| us-east-1a | $0.4134 | `InsufficientInstanceCapacity` |
| us-east-1b | $0.3432 | `InsufficientInstanceCapacity` |
| us-east-1c | $0.3690 | launched |

Worth knowing: the error AWS returned in `us-east-1b` recommended trying `us-east-1a`, which
had refused the identical request ninety seconds earlier. That guidance is about **on-demand**
capacity — it is not spot-aware, and following it will send you straight back to a pool you
have already been rejected from. Walk the AZ list yourself.

## The install said OK and nothing worked

Now the part I actually want to write down. The JAX path removed the build, the toolkit, the
Rust dependency and the kernel patch. What it did not remove was the property that made the
vLLM rig hard: **on hardware this far off the mainstream, things do not fail where you are
looking.** I hit three of those in one afternoon, and all three reported success.

### Two interpreters named python3.12

The service came up and crash-looped:

```
File "/opt/jax-g5g/app/jax_openai_server.py", line 24, in <module>
    import jax
ModuleNotFoundError: No module named 'jax'
```

Five minutes after an install step that had ended by running `import jax`, printing
`jax 0.11.1 devices: [CudaDevice(id=0)]`, and touching its done-marker.

Both were true. The DLAMI already carries `/usr/local/bin/python3.12`, and `/usr/local/bin`
precedes `/usr/bin` on PATH. My bootstrap installed 3.12 from deadsnakes — landing a *second*
interpreter at `/usr/bin/python3.12` — then ran bare `python3.12` to install jax, which
resolved to the DLAMI's copy in `/usr/local`. The systemd unit, meanwhile, had
`ExecStart=/usr/bin/python3.12`, absolute and correct-looking, pointing at the interpreter that
got nothing:

```
--- PATH python3.12 sees jax? ---
OK /usr/local/bin/python3.12 0.11.1
--- /usr/bin/python3.12 sees jax? ---
ModuleNotFoundError: No module named 'jax'
```

The verification step could not catch this, because it *also* ran bare `python3.12` and
therefore tested the interpreter that was never going to serve. I had a test asserting
`ExecStart` was an absolute path — which it was, and which is why the test passed while the
service died.

The fix is to resolve the interpreter after installing and rewrite the unit:

```bash
PY_BIN="$(command -v python3.12)"
sed -i "s|^ExecStart=[^ ]*|ExecStart=$PY_BIN|" /etc/systemd/system/jax-g5g.service
systemctl daemon-reload
```

Still absolute, so systemd is happy and `ExecStart` never depends on the service's PATH — but
now it is the *same* absolute path the packages went to. If you install into an image that
already has your Python version, assume you now have two.

### A GPU probe that could never pass

My rig has a diagnostic whose entire job is to answer the question the project turns on: does
JAX really reach this GPU? It runs a matmul, because a config flag being accepted proves
nothing and a kernel either launches or it does not.

It reported:

```
NVIDIA T4G, 7.5, 15360 MiB
jax: 0.11.1
device: cuda:0 gpu
capability: 7.5
fp16 matmul ok: False
```

On a GPU that was completely healthy. The check was:

```python
x = jnp.ones((256, 256), jnp.float16)
print('fp16 matmul ok:', float((x @ x).sum()) == 256.0 * 256.0 * 256.0)
```

Every element of `x @ x` is exactly 256.0. The sum of 65,536 of them is 16,777,216. **The
largest finite value in float16 is 65,504.** `jnp.sum` on a float16 array accumulates in
float16, so the reduction overflows to `inf`, and `inf != 16777216.0`, on every device that
has ever existed. The matmul was never the problem; the assertion was arithmetically incapable
of passing.

```
y[0,0]: 256.0    elementwise correct: True
fp16 sum : inf
fp32 sum : 16777216.0    expected: 16777216.0
```

One `dtype=jnp.float32` on the reduction fixes it. What is worth sitting with is the shape of
the mistake: I wrote a test *specifically* to avoid trusting a flag, and then put an fp16
overflow inside it. The verification had the same class of bug as the thing it was verifying,
and on unusual hardware the natural reading of `matmul ok: False` is "the hardware does not
work" — which would have sent me chasing a nonexistent kernel problem for an afternoon.

### /health was green and every generation was a 500

Then the model loaded, and the server came up, and `/health` returned 200, and every single
completion returned a 500. The journal showed the status code and no traceback, because the
handler swallowed it. The body had it:

```json
{"detail":"apply_chat_template requires jinja2 to be installed."}
```

`transformers` renders chat templates through Jinja, does not depend on it, and every serving
path on an instruction-tuned model goes through `apply_chat_template`. So the process starts
fine, reports healthy forever, and cannot generate a single token.

It also does not recover on its own: `transformers` memoizes the availability check at import,
so installing jinja2 into the running box changes nothing until you restart the service. I
installed it, retried, got the identical error, and briefly believed I had installed it into
the wrong interpreter again.

**This is the failure mode to design against.** A liveness probe that only checks `/health`
would have called this deployment successful. My health tool asserts on the returned
completion text instead — that is the only reason I caught it — and the same rig's notes warn
against the inverse mistake, because on the vLLM sibling a raw `/v1/completions` call returns
`': ok: ok: ok: ok'`: a non-empty body full of garbage, which passes a "did it return
anything" check just as cheerfully.

Three failures, three green lights: an installer that verified the wrong interpreter, a GPU
probe that could not pass on any hardware, and a health endpoint that was structurally unable
to observe the thing that was broken.

## Cold is not warm, and the difference is 2.7x

With it serving, the first numbers off the box were:

```
decode: 4.6 tok/s        prefill: 7,372 ms
```

That is XLA compiling. Five warm requests later:

```
decode: 12.5 tok/s       prefill: 131 ms
```

**A 56x difference in prefill, and 2.7x in decode, between the first request and the fourth.**
JAX traces and compiles per shape, so every new sequence-length bucket pays again — which is
why the engine pads to static buckets, and why a benchmark harness that does not warm up will
understate this rig by more than a factor of two.

Measured end-to-end from my laptop, five runs, 17-token prompt and 64 completion tokens:

| run | wall (s) | completion tokens | tok/s |
|---|---|---|---|
| 1 | 5.34 | 64 | 11.99 |
| 2 | 5.33 | 64 | 12.01 |
| 3 | 5.33 | 64 | 12.00 |
| 4 | 5.33 | 64 | 12.00 |
| 5 | 5.33 | 64 | 12.01 |

A 0.02 tok/s spread across five runs. Whatever else is true, it is not noisy.

## What it costs: 12 tok/s against 43

Here is the comparison the whole exercise was for. Same instance family, same GPU, same
checkpoint, single stream:

| | vLLM | pure JAX |
|---|---|---|
| Time to a serving endpoint | ~67 min build + toolkit + Rust | **~5 min, pip only** |
| Patches required | 1, unlanded, on one volume | **0** |
| Compiler on the box | `cuda-toolkit-13-2` (sbsa) | **none** |
| Attention | `TRITON_ATTN`, forced, patched | XLA, unremarkable |
| Weight load | — | 9.26 GB in 158.8 s |
| Restart (warm compile cache) | — | ~80 s |
| GPU memory in use | 13,501 MiB | 13,573 MiB |
| Throughput, single stream | **43.1 tok/s** | **12.0 tok/s** |

Roughly 3.5x. Be careful how much weight you put on that ratio: the vLLM figure was obtained
with reduced Triton tiles on a patched kernel, this JAX engine is a reference implementation
running one sequence at a time with no continuous batching, and each is a single sample. It is
directional, not a benchmark result.

But the direction is not subtle, and it is not mysterious either. vLLM is a serving engine —
paged KV, continuous batching, CUDA graphs, kernels written for the job. This is a model
implementation with an HTTP server bolted to it. **The 67-minute build is not what you are
paying for; you are paying for the several thousand engineering hours inside the thing being
built.** Skipping the build skips those too.

## So which one should you run?

Run vLLM if the endpoint is the product. 3.5x is 3.5x, it batches, and the build is a one-time
cost you can bake into an AMI — which is what I did on the other rig, turning a multi-hour
provision into a four-minute launch.

Run pure JAX if the *model* is the product: you are modifying attention, testing a quantization
scheme, checking numerics against a reference, or porting to a chip nobody supports. On this
hardware that last one is not hypothetical. The JAX rig went from nothing to serving in about
fifteen minutes of wall-clock, on a stock AMI, with no artifact to rebuild when jax updates.
The vLLM rig carries a patch that has to be reapplied by hand on every upgrade, forever, until
it lands upstream.

And there is a case the throughput number hides entirely: JAX is where a fix is *possible*.
The Turing shared-memory ceiling stopped vLLM because the tile sizes are baked into a Triton
kernel; the fix was to go edit that kernel. In JAX I hit the same ceiling in the W4A16 Pallas
kernel and could simply not take that path — the dense fp16 route is right there, same code,
one flag. Being 3.5x slower on a chip you can actually reason about beats being fast on one
where the next model shape puts you back in someone else's kernel.

## Troubleshooting quick reference

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'jax'` under systemd | Two interpreters. `command -v python3.12` != your `ExecStart`. |
| Install verified fine, service dies anyway | Verify step and `ExecStart` resolve different interpreters. |
| `fp16 matmul ok: False` on a healthy GPU | fp16 reduction overflow. 256³ > 65,504. Sum in fp32. |
| `/health` 200, every completion 500 | Missing `jinja2`. Needs a restart after installing. |
| `apply_chat_template requires jinja2` after installing it | `transformers` memoizes the check at import. |
| `ScopedMemoryError`, 550 KiB–1.1 MiB needed | Fused W4A16 Pallas kernel on pre-Ampere. Serve dense. |
| Correct output, unexplained slowness | `bfloat16` on Turing. It emulates, it does not fail. |
| `InsufficientInstanceCapacity`, AWS suggests an AZ | That advice is on-demand. Walk the AZs yourself. |
| First request 50x slower than the fourth | XLA compiling per shape bucket. Warm up before measuring. |

## The short version

Take the ARM64 GPU DLAMI and `pip install 'jax[cuda12]'` — the aarch64 wheels carry `sm_75`,
the CUDA libraries come from pip, and the image only has to supply a driver. Install Python
3.12 because jax 0.11 needs it, then **resolve the interpreter and rewrite your systemd
`ExecStart`**, because the image already has a 3.12 that wins on PATH. Add `jinja2`, which
`transformers` needs and does not depend on. Serve the dense checkpoint at `float16` — Turing
has no bf16 unit and no fp8 datapath, and the fused W4A16 kernel wants 1 MB of shared memory
where the chip has 64 KiB. Warm it up before you believe a number.

You get a serving endpoint in five minutes instead of sixty-seven, with nothing to patch and
nothing to rebuild, at about a third of vLLM's throughput.

Nothing here failed loudly. The installer verified an interpreter it would never use, the GPU
probe was arithmetically incapable of passing, and the health endpoint reported green through a
server that could not produce a token. The vLLM rig taught me that the hard part was not the
packaging; the JAX rig taught me that removing the hard part does not remove the *silence*.
On hardware this far off the mainstream, the discipline that pays is not reasoning more
carefully in advance — it is making every check assert on the thing you actually care about,
and getting to a box early enough that it can tell you how wrong you were.

---

*Measured on EC2 `g5g.2xlarge` spot, `us-east-1c`, 2026-08-19. NVIDIA T4G, compute capability
7.5, 15,360 MiB, driver 580.126.09. Deep Learning ARM64 AMI OSS Nvidia Driver GPU PyTorch 2.7
(Ubuntu 22.04.5), `ami-077792d0bb6a000b8`. jax 0.11.1, jaxlib 0.11.1, jax-cuda12-plugin 0.11.1,
nvidia-cublas-cu12 12.9.2.10, nvidia-cudnn-cu12 9.24.0.43, transformers 5.15.1, Python 3.12.
Single stream, `max_num_seqs=1`, one sample per cell — directional, not a benchmark.*
