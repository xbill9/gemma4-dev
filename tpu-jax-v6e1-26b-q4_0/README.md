# tpu-jax-v6e1-26b

Port and verification artifacts for **`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`** targeting a
**TPU v6e-1**, on the pure-JAX engine. 2026-07-31.

> **Ported and verified against the HF reference on CPU. Not yet run on a TPU.** Every memory figure
> here is arithmetic, not an allocation log. See [`CLAUDE.md`](CLAUDE.md).

**This rig serves nothing** — no MCP server, no skill, no deployment path.

## Contents

| Path | What |
| --- | --- |
| `benchmarks/runs/2026-07-31-gemma4-26b-v6e1/REPORT.md` | The port report: why this size needed one, Q4_0 round-trip fidelity, correctness, the prefill FLOP cost |
| `docs/gemma4-26b-quirks.md` | §15–22 — MoE wiring, router internals, expert tensor layout, the missing checkpoint, two ways to destroy the weights |

## The short version

The 26B A4B is the odd one out **twice**: the only **sparse** checkpoint in the family, and the only
size with **no `-w4a16-ct` release**.

- **The MoE block runs *alongside* the dense MLP, not instead of it.** `enable_moe_block: true` keeps
  every layer's ordinary `mlp` and *adds* a 128-expert bank; the two outputs sum before a shared
  post-norm. Hence five norms in the feed-forward block.
- **The router reads the RAW residual; the experts read a normalized copy.** Merging them costs
  **0.36 relative error with every unit test still green** — router parity passed, expert parity
  passed, only whole-model comparison caught it.
- **`-q4_0-unquantized` is QAT data in an unquantized container.** 51.61 GB of BF16 that repacks to
  **15.27 GB** at W4A16. Group size 32 is measured, not assumed (64 fails the same test).
- **The textbook `d = amax/8` Q4_0 step is wrong here** — 4.9e-2 median error, and nothing raises.
  Searching for the level the peak actually occupies plus least-squares refinement reconstructs
  93.1% of values exactly.
- **Prefill does 16x the expert FLOPs** (128 experts / top-8) — optimal in bytes moved, wasteful in
  compute. Fixing it needs expert-sorted dispatch, which was deliberately not attempted because
  capacity-padded dispatch silently drops tokens.
- **KV is unusually cheap**: 2 global KV heads, `attention_k_eq_v`, 1024 sliding window. On this
  model KV is not the constraint — prefill temporaries are.

## Config, where it differs from the dense sizes

30 layers (25 sliding / 5 full at `i % 6 == 5`), `hidden_size` 2816, `intermediate_size` 2112,
128 experts / top-8, `moe_intermediate_size` 704, `num_global_key_value_heads` **2** (the 31B uses
4 — do not carry it over), no KV sharing, no PLE, no double-wide MLP.
