# `gemma4-geometry` — E2B geometry and invariants

**Status: arithmetic only, and that is its whole job.** It cannot serve a token and is not meant
to — `gemma4-engine` is the serving crate. This one exists so the model's geometry and the
expensive invariants are checkable without a GPU.

It is the first increment of the migration described in `../docs/rust-jax-runtime-survey.md`,
and it is deliberately the part that needs **no GPU, no XLA, and no aarch64**: config parsing,
KV-cache geometry, and weight dtype conversion are host-side and architecture-independent. That
is why it builds and tests on any machine, including an x86_64 dev box with no NVIDIA driver.

The survey's recommendation was to port **bottom-up against a parity harness** rather than
top-down. This is the bottom.

```
make rust-test     # 10 tests, no GPU required
make rust-lint     # cargo fmt --check + clippy -D warnings
```

## What is here

| Module | Holds |
| --- | --- |
| `config.rs` | `Gemma4EConfig`, the period-5 layer pattern, `kv_share_map()`, per-layer head dims |
| `kv.rs` | cache planning, the sliding ring, the padding invariant, the bucket ladder |
| `dtype.rs` | bf16 → f16 conversion at load |

## What is deliberately *not* here

The forward pass, attention, RoPE, the LM head and safetensors parsing. **None of it needs
writing:** `rlx-gemma` already implements Gemma 4 E2B including PLE and QAT, and
`gemma4-engine` drives it. This crate keeps the arithmetic that the rig reasons about — KV
budgets, bucket padding, the ring invariant — where it can be asserted in a unit test rather
than inferred from a serving run.

`gemma4-engine` uses it: `/health` reports `kv_bytes_at_max_seq` from `kv::total_kv_bytes`, and
the cold-shape tracker keys on `kv::bucket_for`.

## Where the E2B fixture comes from

**The monorepo root `MODELS.md`, which was verified by reading the safetensors headers** — not a
config file, not prose, and deliberately not a sibling rig's test fixture.

That distinction earned its own test. `MODELS.md` records that a 12B exploration note listed E2B at
`num_key_value_heads = 4` and `num_global_key_value_heads = 4`, filled in from memory; the headers
say **1** (full MQA), and boot-time allocation arithmetic agrees to 0.1% on two chip generations.
**`gpu-jax-g5g-2b/ports/gemma4/jax_e_smoke_test.py` still carries the 4/4 values**, together with
`hidden_size = 2048`, under the comment `# Real Gemma 4 E2B MatFormer configuration`. Those three
numbers are the Python dataclass *defaults*, not the checkpoint.

`projection_shapes_match_the_safetensors_headers` and
`the_four_kv_head_config_is_not_e2b_and_the_shapes_prove_it` pin both directions: the correct config
reproduces the header table shape for shape, and the 4/4 config misprices KV by exactly 4x.

## Why the tests assert what they do

Every test pins a number that was **independently derived or measured on the Python rig**, so a
failure is a real divergence rather than a restatement of the code:

- **KV geometry reproduces `CLAUDE.md`'s table exactly** — 30.0 MiB at ctx 4096 with
  `window_kv`, 72.0 MiB without, 54.0 MiB at ctx 8192 — and `MODELS.md`'s 18,432 B/token.
- **15 caches exist, not 35**, split 12 sliding / 3 full; KV sharing collapses to exactly two
  sources, layer 13 and layer 14.
- **The padding invariant is executable.** `ring_slot` returns `None` for a pad position rather
  than a slot, which is the fix from `docs/padding-window-eviction.md` expressed as a type
  rather than a comment. The eviction bug's failure mode was a `status="success"` token loop,
  so a rewrite that gets this wrong looks like it works.
- **The bucket ladder's worst-case padding is asserted to be 127**, below E2B's 512 window,
  which is what makes the eviction failure unreachable rather than merely unlikely.
- **`hidden_size / num_attention_heads` is asserted to be 192 and wrong.** The heads do not
  tile hidden_size, and deriving head_dim that way breaks everything downstream.
- **Projection shapes reproduce the safetensors table exactly** — `q_proj` 2048x1536 sliding and
  4096x1536 full, `k_proj`/`v_proj` 256x1536 and 512x1536. One KV head, so `out` equals head_dim.
  This is the cheapest way to tell a real E2B config from a plausible one.

## One design decision that departs from the Python port, on purpose

**Every shape-critical config field is required.** The Python `Gemma4EConfig` defaults
`hidden_size` to 2048 (E2B's real value is 1536) and `num_key_value_heads` to 4 (really 1), so a
field missing from `config.json` yields a *wrong-shaped model rather than an error*. Here serde
carries no default on any shape field, so the same input is a parse error naming the field.
`a_missing_shape_field_is_an_error_not_a_default` pins it.
