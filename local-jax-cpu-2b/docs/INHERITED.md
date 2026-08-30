# What these docs are, and whose numbers they carry

Forked from `gpu-jax-g4dn-2b` on 2026-08-29, which was itself forked from
`gpu-jax-g5g-2b` a day earlier. **Every measurement in this directory was taken on a
Turing GPU — a T4G, on an AWS Graviton2 host. Nothing here has been reproduced on a CPU.**

Two hops from the machine that measured them, so read the provenance line in each file
rather than trusting this table alone.

| File | Carries over? |
| --- | --- |
| `padding-window-eviction.md` | **Yes, entirely — and this is the most relevant document in the tree.** Engine-level correctness in the shared model port, with nothing hardware-specific in the mechanism. It was verified on CPU, which is what this rig *is*. |
| `bf16-weights-on-turing.md` | **The mechanism, yes. The conclusion, no — and the difference matters.** |
| `profiling-recipes.md` | **Partly.** The traps about warm-up and about `xspace_to_tool_data` returning bytes carry. The GPU kernel tables do not. |

## `bf16-weights-on-turing.md` needs reading against the grain

Its subject — bf16 weights being converted in front of every use, because the storage dtype
does not match the compute dtype — **is real here too**, and for the same reason: XLA:CPU has
no bf16 datapath and upconverts to fp32.

Its *conclusion* is the opposite of this rig's. That document is an investigation into how to
**remove** the conversion, and records three placements tried and rejected on hardware. Here
there is nothing to remove:

- **float16 is not an escape.** On Turing it was — fp16 tensor cores are a real datapath. A CPU
  has no 16-bit float datapath of any kind, so float16 storage is upconverted identically.
- **float32 storage would work and does not fit.** 18.5 GB against a host's RAM.

So the option space is *smaller* here, not larger. Do not port that document's open work item.

Its 2026-08-28 follow-up is worth reading for a different reason: the finding that
`safetensors.flax.load_file` returns arrays **already on the accelerator**, so every line of
reasoning about "where on the host to cast" was answering the wrong question. On this rig
`jax.devices()[0]` *is* the host, which collapses that distinction entirely — a device→host
round trip costs nothing because there are not two places.

## Two parent docs were deleted rather than inherited

Keeping them would have made false claims:

- **`turing-aarch64-gap.md`** — the shared-memory ceiling and the aarch64 wheel gap. Neither
  applies: there is no shared memory to budget and no GPU wheels to find. Read it in the G5g
  rig if you need the vLLM-side story.
- **`larger-models-on-t4g.md`** — sizing arithmetic against a 14.07 GB device budget. There is
  no device budget here. **Its model-size table does not transfer even in shape**, because its
  blockers were hard OOMs and here the same over-allocation is absorbed by swap: E4B and 12B
  would *load* on a large enough host, slowly, rather than failing. Whether that is useful is
  an open question and nobody has tried it.

## The one hazard this rig has that none of the parents do

**Pallas has no CPU backend, so the fused W4A16 kernel AUTO-ENABLES interpret mode** —
`JAX_E_PALLAS_INTERPRET` defaults to 1 whenever the platform is neither TPU nor GPU. On the
GPU siblings that kernel is *refused at startup* with the arithmetic attached; here it runs, in
a simulator, producing correct numbers at a speed that means nothing.

The startup banner is the only warning:

```
jax_e_model device policy: platform=cpu compute_capability=n/a
compute_dtype=bfloat16 pallas_interpret=True
```

Read it before believing any w4a16 number from this rig.
