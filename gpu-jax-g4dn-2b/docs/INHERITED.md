# What these docs are, and whose numbers they carry

Forked from `gpu-jax-g5g-2b` on 2026-08-28. **Every measurement in this directory was taken on
G5g (Graviton2 + T4G), not on this rig.**

**UPDATED 2026-08-29, after this rig's first serve.** The chip-level findings here are now
corroborated on `g4dn`: decode 13.1 tok/s against the parent's 13.10, `tpu_jax_weight_bytes`
identical to the byte, and the xprof profile reproducing 54.4% dtype conversion / 32.8% fp32
GEMV / 0.0% TensorCore, with roofline peaks identical to three decimals. So these documents
may now be read as describing **this** chip, not merely a related one — but the *numbers* in
them are still the parent's, and anything **host**-level (install time, load staging, host RSS)
was measured behind a Graviton2 and has not been reproduced here.

| File | Carries over? |
| --- | --- |
| `bf16-weights-on-turing.md` | **Yes, chip-level.** T4 and T4G are both Turing SM 7.5, so the dtype analysis is about this chip too. The *numbers* are still the parent's. |
| `padding-window-eviction.md` | **Yes, entirely.** Engine-level correctness in the shared port; nothing hardware-specific. |
| `profiling-recipes.md` | **Yes, and updated for this rig.** xprof works the same way here, but as of 2026-08-29 it and tensorboard are installed at boot rather than on demand, so the install section is this rig's own. |

Two parent docs were **deleted rather than inherited**, because keeping them would have made
false claims:

- `turing-aarch64-gap.md` — half of it is the Turing shared-memory analysis, which applies; the
  other half is the aarch64 packaging gap, which does **not**. x86_64 + CUDA is the well-trodden
  axis and none of that document's difficulty exists here. Read it in the parent.
- `larger-models-on-t4g.md` — sizing arithmetic against 15,360 MiB. It was deleted on the
  theory that this rig's T4 reports **16,384 MiB**, which would have shifted every threshold.

  **That theory was wrong, and was falsified on 2026-08-29.** The 16,384 figure came from
  `describe_instance_types`, which reports the nominal card size; `nvidia-smi` on the device
  reports `Tesla T4, 7.5, **15360 MiB**` — identical to the T4G. So that document's arithmetic
  *would* have transferred unchanged, and deleting it was the wrong call for the right-sounding
  reason. Read it in the parent; its thresholds apply here as written. The general lesson is
  the rig's own: **the device is what allocates, so the device is what you measure.**
