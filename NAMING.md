# Rig naming scheme

Canonical reference for how accelerator rig repos are named. Rigs currently live in three trees — `~/tpu-*`,
`~/gemma4-dev/*`, and `~/gemma4-queens/*` — and the scheme is the same regardless of which tree a rig sits in.
Location is not part of the name.

## The four slots

```
<platform>-<runtime>-<hardware>-<model>
```

```
tpu  -  vllm  -  v5e1  -  2b
 │        │       │       └─ Gemma 4 size:  2b / 4b / 12b / 26b / 31b
 │        │       └───────── accelerator + chip count:  v5e1 / v6e1 / v6e4 / inf2
 │        └───────────────── serving stack:  vllm / jax / pytorch
 └────────────────────────── where it runs:  tpu / gpu / gce
```

Read `tpu-vllm-v5e1-2b` as: **a Cloud TPU running vLLM in Docker on v5e-1 hardware, serving the 2B Gemma 4
model.**

This generalizes the older `tpu-<framework>-<accel>-<size>` pattern (documented in `~/.claude/CLAUDE.md` and
`~/gemma4-dev/tpu-pytorch-v5e1-12b/CLAUDE.md`) by making the leading `tpu` a real platform slot rather than a
fixed prefix. Slots 2–4 are unchanged, so every conforming repo name stays valid.

### Formatting rules

- All lowercase, four slots, `-` between slots, exactly three hyphens.
- No hyphen *inside* a slot — that's what keeps the name parseable by splitting on `-`. Accelerator shapes
  lose their internal punctuation: `v5e-1` → `v5e1`, `v6e-4` → `v6e4`.
- Slots are positional and never omitted. A three-slot name is ambiguous — you can't tell whether
  `tpu-jax-12b` is missing the hardware or the model.
- Order is fixed and goes outside-in: platform, then stack, then silicon, then payload.

## Slot 1 — platform

| Value | Means |
| --- | --- |
| `tpu` | Google Cloud TPU VM or Queued Resource |
| `gpu` | NVIDIA GPU host, anywhere (GCE, EC2, Cloud Run) |
| `gce` | Compute Engine with no accelerator — CPU-only serving, tooling, or control-plane rigs |

These three are the permitted values. Cloud provider is deliberately not a slot; it's implied by the hardware
slot or recorded in the rig's `tpu.env`. Add a platform value only for a genuinely different execution target.

`platform` is the accelerator family, not the cloud. An Inferentia rig takes `tpu` — it's a dedicated
inference accelerator attached to a VM, which is the same shape of thing as a TPU rig and is how the existing
repos are named (`~/tpu-jax-inf2`, `~/tpu-pytorch-inf2-2b`). The part itself lives in the hardware slot as
`inf2`. Reserve `gpu` for general-purpose GPUs and `gce` for no accelerator at all.

## Slot 2 — runtime

The serving stack that actually loads the weights.

| Value | Means |
| --- | --- |
| `vllm` | vLLM, OpenAI-compatible API on `:8000` (Docker image or pip install) |
| `jax` | JAX-native — Flax, MaxText, or a hand-rolled JAX server |
| `pytorch` | PyTorch, via `torch_xla` on TPU or CUDA on GPU |

`vllm` names the server, not the backend it compiles to — vLLM on TPU is `vllm`, never `pytorch`, even though
`torch_xla` sits underneath. Pick the layer you'd file a bug against.

## Slot 3 — hardware

Accelerator generation plus chip count, punctuation stripped.

| Value | Means |
| --- | --- |
| `v5e1` | TPU v5e, 1 chip |
| `v6e1` | TPU v6e (Trillium), 1 chip |
| `v6e4` | TPU v6e, 4 chips |
| `inf2` | AWS Inferentia2 |

For `gpu` platforms the slot takes the GPU SKU — `l4`, `t4`, `a100` — same rule: lowercase, no punctuation.
For `gce` use `cpu`.

The value is always **`inf2`**, never bare `inf` — the generation is part of the part name, both existing
rigs already spell it that way (`~/tpu-jax-inf2`, `~/tpu-pytorch-inf2-2b`), and a family-only value would be
the one hardware slot that doesn't pin a generation. Later parts take `inf3`, `trn1`, and so on.

Note the trailing digit means something different here: in `v6e4` it's the chip count, in `inf2` it's the
generation (an Inferentia2 rig, not two chips). Chip count for these parts, when it matters, belongs in
`tpu.env`, not the name.

The older wording for this slot was "`ACCELERATOR_TYPE` with the dash dropped." That holds for most rigs but
breaks on v5e, where gcloud spells the type `v5litepod-1` — dropping the dash would give `v5litepod1`, and
every real directory says `v5e1`. **The rule is marketing generation + chip count**, normalizing gcloud's
`v5litepod` spelling back to `v5e`. Siblings currently disagree on the env value itself
(`tpu-vllm-v5e1-2b` sets `ACCELERATOR_TYPE=v5litepod-1`, `tpu-pytorch-v5e1-12b` sets `v5e-1`), which is
exactly why the name can't be derived mechanically from it.

Chip count is *topology*, not tensor-parallel size. `v5e1` is one chip so `TENSOR_PARALLEL_SIZE=1`; they match
here but are separate settings.

## Slot 4 — model

Gemma 4 parameter size, lowercase `b`. Verified mappings from rigs that exist today:

| Slot | `MODEL_NAME` |
| --- | --- |
| `2b` | `google/gemma-4-E2B-it` (QAT variant: `google/gemma-4-E2B-it-qat-w4a16-ct`) |
| `4b` | `google/gemma-4-E4B-it` |
| `12b` | `google/gemma-4-12B-it` (QAT variant: `google/gemma-4-12B-it-qat-w4a16-ct`) |
| `26b` | `~/tpu-jax-26b` |
| `31b` | `~/tpu-jax-31b` |

Use **`2b`**, not `e2b` — every existing directory does, even though the checkpoint is `E2B` and the
benchmark-artifact slug is `gemma4-e2b`. (`~/gemma4-dev/tpu-pytorch-v5e1-12b/CLAUDE.md` lists `e2b` as a size
example; no directory follows it.) The `E`-prefixed form belongs to `MODEL_NAME` and benchmark filenames, not
the repo name.

Quantization is **not** a slot. A QAT/w4a16 build of the 2B model is still `…-2b`; the exact checkpoint is
whatever `MODEL_NAME` says. If a QAT and a full-precision rig ever need to coexist, suffix the directory
(`tpu-vllm-v5e1-2b-qat`) rather than bending the four slots.

## The name is a claim about `tpu.env`

The directory name is documentation; the rig's env file decides. `v5e1` is for humans — the string gcloud
wants is `v5litepod-1`, and it lives in `tpu.env` as `ACCELERATOR_TYPE`. Never copy a slot value into a CLI
flag.

Because the name is a claim, **retargeting a rig's accelerator or model makes the name stale until it's
renamed too**, and a rename is more than `mv`:

- Rename the **GitHub repo**, not just the local directory. Rigs under `~/gemma4-dev` no longer have their own
  repo — the whole tree is the `xbill9/gemma4-dev` monorepo, so for those a rename is just the directory plus
  the sweep below. Rigs in the other two trees still push individually and several have drifted:
  `~/tpu-pytorch-v6e1-2b` pushes to `xbill9/tpu-pytorch`, and `~/tpu-pytorch-inf2-2b` to
  `xbill9/tpu-pytorch-inf2`.
- In the monorepo, a rig's directory name is also its **plugin name** in the root
  `.claude-plugin/marketplace.json` and in its own `.claude-plugin/plugin.json`. Renaming the directory means
  renaming the plugin in both, or `/plugin marketplace add xbill9/gemma4-dev` serves a stale entry.
- Sweep the old name out of the repo: `grep -rn <old-name> --exclude-dir=.git --exclude-dir=dist`. Each rig is
  an independent fork, so the name is duplicated in docs, `.claude-plugin/` manifests, JSON-schema `$id`s, and
  hardcoded absolute `/home/xbill/<repo>/...` paths in test files.
- `~/gemma4-dev/tpu-pytorch-v5e1-12b/CLAUDE.md` carries the full per-file checklist plus the conventions for
  that rig's cloud resource ids.

## Benchmark artifact names

Benchmark files inside a rig use a **different, date-first scheme** — they name a measurement, not a rig, so
the four-slot rules above (exactly three hyphens, fixed positions) do not apply:

```
benchmarks/reports/<date>-<model-short>-<hw-short>.json    2026-07-21-gemma4-e2b-v6e1.json
benchmarks/runs/<date>-<what>-<hw-short>/                  2026-07-25-vllm-sweep-v6e1/
```

| Part | Rule |
| --- | --- |
| `<date>` | ISO `YYYY-MM-DD`, the run date |
| `<model-short>` | family + checkpoint variant: `gemma4-e2b`, `gemma4-12b` — **`e2b`, not the `2b` of the model slot** |
| `<hw-short>` | same value as the hardware slot: `v6e1`, `v5e1` |
| `<what>` | free-form run label, may itself contain hyphens: `vllm-sweep`, `kv-quant`, `real-http`, `jax-e2b` |

Three things that trip people up:

- **A report's `<hw-short>` is the hardware measured, not the rig it lives in.** All four rigs currently carry
  `2026-07-21-gemma4-e2b-v6e1.json`, including the v5e1 ones — the file records a v6e-1 run, and copies
  travelled with the forks. Never infer a rig's hardware from a benchmark filename sitting in it.
- **A report filename must equal its `run.id`** — see `benchmarks/serving-report.schema.json`, whose `run.id`
  description carries the same convention, and `benchmarks/README.md`, which indexes one file per
  (model, hardware) cell.
- **`<what>` is not the runtime slot.** `2026-07-25-vllm-sweep-v6e1` is a vLLM sweep that also lives in
  `pytorch` and `jax` rigs; it labels what was run, not what the rig is.

## Current inventory

**Conforming — all four slots:**

| Rig | Platform | Runtime | Hardware | Model |
| --- | --- | --- | --- | --- |
| `~/gemma4-dev/tpu-vllm-v5e1-2b` | TPU Queued Resource | vLLM in Docker | v5e-1 | `gemma-4-E2B-it` |
| `~/gemma4-dev/tpu-jax-v5e1-2b` | TPU | JAX | v5e-1 | 2B |
| `~/gemma4-dev/tpu-pytorch-v5e1-2b` | TPU | PyTorch / `torch_xla` | v5e-1 | 2B |
| `~/gemma4-dev/tpu-pytorch-v5e1-12b` | TPU | PyTorch / `torch_xla` | v5e-1 | `gemma-4-12B-it-qat-w4a16-ct` |
| `~/tpu-jax-v6e1-2b` | TPU | JAX | v6e-1 | 2B |
| `~/tpu-pytorch-v6e1-2b` | TPU | PyTorch | v6e-1 | 2B |
| `~/tpu-pytorch-inf2-2b` | Inferentia (slot `tpu`) | PyTorch | inf2 | 2B |

**Predates the scheme — not counter-examples:**

| Rig | Missing |
| --- | --- |
| `~/tpu-jax-4b`, `-12b`, `-26b`, `-31b` | hardware slot |
| `~/tpu-jax-inf2` | model slot |
| `~/tpu-jax` | hardware and model slots |
| `~/tpu-skill-claude`, `-codex`, `-agy` | not rigs — skill repos |
| `~/tpu-inference` | not a rig |

**`~/gemma4-queens/*`** is a separate git repo using the older
`<platform>-<model>-<hardware>-<host>-agent` form with mixed case and a `-devops-agent` suffix. Nothing here
renames it. If it's ever migrated, platform/hardware/model read straight off the old name but the **runtime
slot must be confirmed by reading the project** — the old names don't record it, and every one of those
directories mentions vLLM, JAX, and PyTorch somewhere in its dependencies.

| Legacy name | Translates to |
| --- | --- |
| `tpu-2B-v5e1-devops-agent` | `tpu-<runtime>-v5e1-2b` |
| `tpu-12B-v6e1-devops-agent` | `tpu-<runtime>-v6e1-12b` |
| `gpu-2B-L4-ec2-agent` | `gpu-<runtime>-l4-2b` |
| `gpu-4B-inf-devops-agent` | `tpu-<runtime>-inf2-4b` (platform is the accelerator family, not the old `gpu` label; confirm the generation by reading the project) |
| `gpu-4B-cloudrun-devops-agent` | `gpu-<runtime>-<sku>-4b` |
| `g2-4-2B-qat-L4-devops-agent` | `gpu-<runtime>-l4-2b` (QAT — see the model slot rules) |

The old scheme's host slot (`ec2`, `cloudrun`) has no home in the four slots; that detail belongs in the
project's README and env file, not its directory name.

## Adding a rig

1. Name the directory from the four slots.
2. Check the name is unique. Two rigs differing only in something the slots don't capture (quantization, zone,
   batch settings) need a suffix, not a reused name.
3. Create the GitHub repo under the **same** name — don't inherit a shorter remote from the rig you forked.
4. Set the authoritative values in `tpu.env` (`MODEL_NAME`, `ACCELERATOR_TYPE`, `TENSOR_PARALLEL_SIZE`, zone).
   The name describes; the env file decides.
5. Add a row to **Current inventory** above.
