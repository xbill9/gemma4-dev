# What these docs are, and whose numbers they carry

Forked from `gpu-jax-g5g-2b` on 2026-08-28. **Every measurement in this directory was taken on
G5g (Graviton2 + T4G), not on this rig.** Nothing here has been reproduced on `g4dn`.

| File | Carries over? |
| --- | --- |
| `bf16-weights-on-turing.md` | **Yes, chip-level.** T4 and T4G are both Turing SM 7.5, so the dtype analysis is about this chip too. The *numbers* are still the parent's. |
| `padding-window-eviction.md` | **Yes, entirely.** Engine-level correctness in the shared port; nothing hardware-specific. |
| `profiling-recipes.md` | **Yes.** xprof works the same way here. |

Two parent docs were **deleted rather than inherited**, because keeping them would have made
false claims:

- `turing-aarch64-gap.md` — half of it is the Turing shared-memory analysis, which applies; the
  other half is the aarch64 packaging gap, which does **not**. x86_64 + CUDA is the well-trodden
  axis and none of that document's difficulty exists here. Read it in the parent.
- `larger-models-on-t4g.md` — sizing arithmetic against 15,360 MiB. This rig's T4 reports
  **16,384 MiB**, so the conclusions do not transfer unchanged.
