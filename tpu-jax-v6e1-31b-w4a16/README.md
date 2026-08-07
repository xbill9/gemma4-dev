# tpu-jax-v6e1-31b

Measurement artifacts for **`google/gemma-4-31B-it-qat-w4a16-ct`** on a **TPU v6e-1**, served by the
pure-JAX engine. Measured 2026-07-31/08-01 on a spot v6e-1 in `us-central1-a`.

**This rig serves nothing.** It has no MCP server, no skill and no deployment path — it is the home
for the 31B measurements and the findings drawn from them. See [`CLAUDE.md`](CLAUDE.md).

## Contents

| Path | What |
| --- | --- |
| `benchmarks/runs/2026-07-31-gemma4-31b-v6e1/REPORT.md` | The engine: memory floors, prefill cliffs, chunked prefill, the fused-kernel result |
| `benchmarks/runs/2026-07-31-gemma4-31b-v6e1/MODEL-INTEL.md` | The model: norms, `layer_scalar`, massive activations, sink reach, measured W4A16 error |
| `benchmarks/runs/2026-07-31-gemma4-31b-v6e1/jax_31b_*.py` | The harnesses, one per finding |
| `benchmarks/runs/2026-07-31-gemma4-31b-v6e1/results/`, `logs/` | Raw JSON and run logs |
| `docs/gemma4-31b-quirks.md` | Findings organized as **A** the 31B · **B** the W4A16 format · **C** XLA on v6e · **D** how the measurements lie |

## The short version

**The 31B needed no new code.** 60 layers, `hidden_size` 5376, 50 sliding / 10 full, no PLE, no KV
sharing, no double-wide MLP — a strict simplification of what the E-series engine already
implemented. Every difficulty was scale.

Four results worth knowing before reading anything else:

- **Measured W4A16 error is 6.67% and flat** — relative Frobenius error 0.0667, SNR 23.52 dB, spread
  of 0.12% across 20 (layer, projection) pairs. So **there is no cheap mixed-precision win on the
  weights.** This retracts the earlier scale-dynamic-range proxy, which had ranked `v_proj` hardest
  and suggested spending bits on late-layer attention. `v_proj` is among the *lowest*.
- **Massive activations peak at 15,665x**, parked on `<bos>` and `<|turn>` — 78% of layers put their
  peak on one of the first two tokens. The exit clamp (`layer_scalar` = 0.0317 at layer 59) exists to
  crush them before the lm_head.
- **Only 10 of 60 layers can see the sink past 1024 tokens.** Sliding layers are masked to
  `(p − 1024, p]` regardless of `window_kv`, and their measured sink mass is *exactly* zero. That
  makes windowing the sliding layers' KV provably safe, and makes evicting positions 0–1 from the
  full layers' cache unsafe.
- **Prefill is ~10 GB of temporaries before sequence length matters**, and the 4K→8K jump is a
  cliff, not a curve — the attention softmax grows exactly 4.00x per 2x in S.

## Caveats

Activation figures come from one 20-token prompt; the sliding-vs-full ratios are the robust part.
Everything describes the W4A16 QAT checkpoint. `MODEL-INTEL.md` §5 is superseded by §8 and is kept
only because the wrong version is instructive.
