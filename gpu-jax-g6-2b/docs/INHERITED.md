# What these docs are, and whose numbers they carry

Forked from `gpu-jax-g5g-2b` on 2026-08-28. **Every measurement in this directory was taken on
G5g (Graviton2 + T4G, Turing SM 7.5), not on this rig** — and this rig changes the GPU
*generation*, so less carries over than in the `gpu-jax-g4dn-2b` sibling.

| File | Carries over? |
| --- | --- |
| `padding-window-eviction.md` | **Yes, entirely.** Engine-level correctness in the shared port; nothing hardware-specific. |
| `profiling-recipes.md` | **Yes.** Same xprof workflow; it is how the dtype question gets answered here. |

Three parent docs were **deleted rather than inherited**:

- `bf16-weights-on-turing.md` — its entire subject is a chip with no bf16 datapath. Ada has one.
  Inheriting it here would assert the opposite of this rig's premise. It is the *question* this
  rig exists to answer, so read it in the parent, where it was measured.
- `turing-aarch64-gap.md` — neither half applies: not Turing, not aarch64.
- `larger-models-on-t4g.md` — sizing against 15,360 MiB. This rig reports **22,888 MiB**.
