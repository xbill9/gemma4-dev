# tpu-jax-inf2-2b

Serve **`google/gemma-4-E2B-it`** with **pure JAX** on **AWS EC2 Inf2** — AWS Inferentia2,
through `jax-neuronx` and the Neuron PJRT plugin.

Platform slot is `tpu` per [`NAMING.md`](../NAMING.md): a dedicated inference accelerator on a
VM, with the part in the hardware slot.

> **Status: the serving path is NOT wired.** Created 2026-08-28 by forking
> [`tpu-pytorch-inf2-2b`](../tpu-pytorch-inf2-2b/) for its Neuron provisioning, and bringing the
> JAX engine (`ports/gemma4/`, `jax_engine.py`, `jax_openai_server.py`) across from
> [`gpu-jax-g5g-2b`](../gpu-jax-g5g-2b/). **`server.py` still deploys the inherited vLLM-Neuron
> Docker container.** Replacing that with a systemd JAX unit is the open work, and until it is
> done this rig provisions Inferentia2 and serves PyTorch, which is not what its name claims.

## Why this is worth building

Unusually for a new rig, the hard question is already answered — by runs that live in the
**parent** rig, not here.

**2026-08-02, latest Neuron/JAX stack** (`tpu-pytorch-inf2-2b/benchmarks/runs/2026-08-02-inf2-latest-stack-e2b/`):

> the engine is token-exact on Neuron *with the workaround off* at **~14 tok/s**, for both
> windowed and unwindowed KV. The fault lives in the `w4a16` + `int8`-KV serving configuration,
> not in the engine.

That is the whole thesis. The notorious Neuron failure — a too-large gather on the 4.70 GB PLE
table returning **zeros instead of an error**, producing a clean `200 OK` with zero completion
tokens — is a property of the **quantized** serving configuration. This rig deliberately serves
the **dense reference checkpoint at `fp16` with `bf16` KV**, which is the exact configuration
parity proved correct.

And it is cheap. `inf2.xlarge` is **$0.1417/hr spot average, $0.1273 at the floor** (7-day,
us-east-1, 2026-08-28) against `g5g.2xlarge`'s $0.3996 — **about a third of the price for a
comparable measured decode rate**, with 16 GiB per NeuronCore against the T4G's 15,360 MiB.

| | `gpu-jax-g5g-2b` | **this rig** |
| --- | --- | --- |
| accelerator | NVIDIA T4G, Turing | **AWS Inferentia2** |
| decode | **13.10 tok/s** (measured, this rig's own) | **~14 tok/s** (measured, in-process, parent rig) |
| memory | 15,360 MiB | **16 GiB per NeuronCore** |
| spot avg | $0.3996 | **$0.1417** |
| bf16 | none — emulates | **native** |

## The size ladder, which is a trap

Verified against `describe_instance_types` on 2026-08-28. Inferentia reports under
`InferenceAcceleratorInfo` — `AcceleratorInfo` comes back **empty**, which reads as "this
instance has no accelerator" if you query the field you'd expect.

| size | chips | cores | accel MiB | GiB/core | vCPU | host GiB | $/hr spot avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`inf2.xlarge`** | **1** | **2** | 32,768 | 16.0 | 4 | 16 | **0.1417** |
| `inf2.8xlarge` | **1** | **2** | 32,768 | 16.0 | 32 | 128 | 0.5552 |
| `inf2.24xlarge` | 6 | 12 | 196,608 | 16.0 | 96 | 384 | 2.2670 |
| `inf2.48xlarge` | 12 | 24 | 393,216 | 16.0 | 192 | 768 | 8.3988 |

**`inf2.xlarge` and `inf2.8xlarge` carry the same accelerator.** One chip, two cores, 32,768
MiB, on both. The 8xlarge is roughly **4x the spot price for zero extra compute and zero extra
accelerator memory** — the only thing it buys is host RAM.

**Accelerator memory is per core, not a pool.** 16 GiB on every size in the table, including the
twelve-chip 48xlarge. The API's 16.0 GiB/core independently confirms `HARDWARE.md`'s measured
`hbm_limit_bytes` = 17,179,869,184. **A single-device engine gets 16 GiB no matter what you
pay** — which is exactly why `NAMING.md` keeps chip count out of the rig name and puts it in
`tpu.env`.

So the default is the *smallest* size deliberately, and there is only one reason to move up:

**Host RAM.** `inf2.xlarge` has exactly 16 GiB and the one-time load peaks ~14.5 GB, so it needs
a swapfile — threshold **inclusive** at 16, the same boundary the `gpu-jax` line settled on after
`< 16` skipped the one size that needed it. If that pressure ever proves unfixable, the
8xlarge's 128 GiB is the remedy, and that is *all* it is. Decide it on a measurement, never on
the size suffix.

`_oversized_for_single_device()` flags 24xlarge and 48xlarge rather than rejecting them — the
same warning the `gpu-jax` line carries for `g5g.16xlarge` and `g5g.metal`, where the second GPU
does nothing.

## What is NOT established, quoted from the reports that measured it

The parent's own "What this does NOT establish" section is the specification for this rig:

- **Parity drove `JaxGemmaEngine` in-process, not through HTTP.** The server adds FastAPI, the
  entrypoint's environment, and its own process configuration — and the 7/31 report found the
  entire 2700x cost was *process configuration*, not the model. **Closing this gap is the point
  of the rig.**
- **Four prompts, 24 tokens, batch 1, greedy, and UNWINDOWED KV.** The serving path is windowed
  and no run has ever compared the two on device.
- **~14 tok/s describes one configuration** and must not be quoted as engine capability.

So `benchmarks/` is empty here, and it should stay empty until this rig measures something
itself.

## Two settings that are load-bearing, both measured

```
NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=0
JAX_COMPILATION_CACHE_DIR=
```

**The workaround stays off.** It costs **65x** on the current stack (2700x on jax 0.6.2), and
parity says this checkpoint and dtype combination does not need it. That is a claim to *assert*
on every run, not to assume — a silent regression here returns `200 OK` with nothing in it.

**The compilation cache must stay empty.** Setting it is a hard crash on jax 0.9.2 for Neuron:
the service crash-loops before the model loads with `RET_CHECK failure:
proto.has_host_program_shape()`. `ports/gemma4/backend.py` declares
`persistent_compilation_cache=False` for Neuron and that capability table is correct. This is
inherited directly from the `gpu-jax` line, where the cache **is** load-bearing — do not carry
that reasoning across.

## The open work, in order

1. **Replace the vLLM-Neuron Docker deploy with a systemd JAX unit**, matching the `gpu-jax`
   shape. `server.py:116 _user_data(...)` and the `docker run -d --name vllm-neuron` blocks are
   what has to go.
2. **Run the engine behind HTTP with the workaround off** and diff greedy tokens against the
   PyTorch/CPU/float32 oracle. That is the untested gap parity names.
3. **A/B `--window-kv on|off` on device.** Parity passed unwindowed; the server runs windowed.

## Read first

- [`docs/neuron-jax-quirks.md`](docs/neuron-jax-quirks.md) — twelve measured platform quirks.
  Quirk 1 (silent zero gather) and quirk 2 (the 2700x workaround) are why this rig is scoped the
  way it is. Every entry was measured on `inf2.xlarge`.
- `../HARDWARE.md` § `inf2` — the 16 GiB-per-core finding.

## License

Apache-2.0 — see [`../LICENSE`](../LICENSE).
