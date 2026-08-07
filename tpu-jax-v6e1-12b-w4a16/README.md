# tpu-jax-v6e1-12b

Measurement artifacts for **`google/gemma-4-12B-it-qat-w4a16-ct`** on a **TPU v6e-1**
(`ct6e-standard-1t`, 33.55 GB HBM), served by the pure-JAX engine. Measured 2026-07-31.

**This rig serves nothing** — no MCP server, no skill, no deployment path. See [`CLAUDE.md`](CLAUDE.md),
which also lists two numbers in the report that do not reconcile.

## Contents

| Path | What |
| --- | --- |
| `benchmarks/runs/2026-07-31-gemma4-12b-v6e1/REPORT.md` | Port + benchmark report: parity, memory layout, decode sweep |
| `benchmarks/runs/2026-07-31-gemma4-12b-v6e1/results.json` | Raw sweep data |
| `docs/12b-exploration-2026-07-31.md` | The exploratory run: what the 12B is, and how the digit-string defect was localized |
| `ports/gemma4/jax_12b_*.py` | The seven harnesses — **record only, they do not run here** (the engine stayed in `~/tpu-jax-12b`) |

## The short version

**The 12B is the E-series architecture with every MatFormer feature switched off**, and it needed
**zero code changes to load**:

| field | E2B | 12B |
| :--- | ---: | ---: |
| `hidden_size_per_layer_input` | 256 | **0** — no Per-Layer Embeddings |
| `num_kv_shared_layers` | 20 | **0** — every layer owns its KV |
| `use_double_wide_mlp` | true | **false** |
| `attention_k_eq_v` | false | **true** |
| `num_hidden_layers` | 35 | 48 (40 sliding / 8 full, `i % 6 == 5`) |
| `sliding_window` | 512 | 1024 |

RoPE, logit softcapping (30.0), tied embeddings and the 262,144 vocab are identical.

- **Parity: 100% exact token match** against the HF PyTorch reference, once the prompt carries
  `<bos>` and the Gemma 4 chat template.
- **The `'111111'` digit output on bare prompts is not an engine bug.** The PyTorch reference does it
  too — it is a prompt-formatting requirement of the IT QAT checkpoint. (The 31B does *not* share
  this; it recovers the scaffolding by itself.)
- **8.15 GB resident** at W4A16, leaving ~25.4 GB of a 33.55 GB chip.
- **Decode is exceptionally flat: ~29.5 ms/step (33.9 tok/s) from 1K to 8K context.** Concurrency
  reaches 156 tok/s at B=8.
- Un-chunked prefill materializes the S×S attention matrix and OOMs when B×S > 8,192; chunked
  prefill lifts context to 128K.

## Caveat

The KV-cache accounting in `REPORT.md` does not reconcile — it charges the `attention_k_eq_v` layers
for a V it also says is free, and adds a window-capped figure to an uncapped one. `CLAUDE.md` has the
detail. Re-measure against a boot-time allocation log before quoting a per-token KV cost; the root
[`MODELS.md`](../MODELS.md) is canonical for that class of number.
