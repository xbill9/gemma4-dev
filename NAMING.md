# Rig naming scheme

Canonical reference for how accelerator rig repos are named. Rigs currently live in three trees — `~/tpu-*`,
`~/gemma4-dev/*`, and `~/gemma4-queens/*` — and the scheme is the same regardless of which tree a rig sits in.
Location is not part of the name.

## The slots

```
<platform>-<runtime>-<hardware>-<model>[-<encoding>]
```

```
tpu  -  vllm  -  v5e1  -  2b  -  q4_0
 │        │       │       │       └─ optional weight encoding:  w4a16 / q4_0 / int4 / …
 │        │       │       └───────── Gemma 4 size:  2b / 4b / 12b / 26b / 31b
 │        │       └───────────────── accelerator + chip count:  v5e1 / v6e1 / v6e4 / inf2
 │        └───────────────────────── serving stack:  vllm / jax / pytorch
 └────────────────────────────────── where it runs:  tpu / gce / gke / gpu / cloudrun
```

Read `tpu-vllm-v5e1-2b` as: **a Cloud TPU running vLLM in Docker on v5e-1 hardware, serving the 2B Gemma 4
model.**

The first four slots are mandatory. The fifth is optional and **most rigs don't have one** — see
[Slot 5](#slot-5--encoding-optional) for the narrow case that earns it.

This generalizes the older `tpu-<framework>-<accel>-<size>` pattern (documented in `~/.claude/CLAUDE.md` and
`~/gemma4-dev/tpu-pytorch-v5e1-12b/CLAUDE.md`) by making the leading `tpu` a real platform slot rather than a
fixed prefix. Slots 2–4 are unchanged and slot 5 is purely additive, so every conforming repo name stays
valid.

### Formatting rules

- All lowercase, `-` between slots. Four slots and three hyphens without an encoding; five and four with one.
- No hyphen *inside* a slot — that's what keeps the name parseable by splitting on `-`. Accelerator shapes
  lose their internal punctuation: `v5e-1` → `v5e1`, `v6e-4` → `v6e4`. Underscore survives inside the encoding
  slot, and only there: GGUF spells the formats `Q4_0` / `Q8_0`, and stripping to `q40` neither matches
  anything upstream nor reads as a quant level — it reads like a chip count.
- The first four slots are positional and never omitted. A three-slot name is ambiguous — you can't tell
  whether `tpu-jax-12b` is missing the hardware or the model. A five-slot name is not ambiguous, because
  only the encoding can ever be the fifth.
- Order is fixed and goes outside-in: platform, then stack, then silicon, then payload, then how the payload
  was compressed.
- Slot 5 is omitted whenever the rig serves the reference checkpoint, so a bare four-slot name is a positive
  claim: `tpu-jax-v5e1-2b` and `tpu-jax-v5e1-2b-w4a16` are different rigs, and the short one is the
  unquantized build.

## Slot 1 — platform

**Slot 1 names the control plane that provisions the accelerator** — which API you call to get the hardware,
not which cloud it sits in.

| Value | Means | Provisioned by |
| --- | --- | --- |
| `tpu` | A dedicated accelerator with its own control plane: a Cloud TPU VM or Queued Resource, or AWS Inferentia via Neuron | `gcloud compute tpus tpu-vm` / `queued-resources` |
| `gce` | A Compute Engine instance, where the accelerator (if any) is a property of the **machine type** | `gcloud compute instances create --machine-type=ct6e-…` |
| `gke` | A GKE node pool carrying the accelerator, with the model served from a Pod rather than a VM | `gcloud container clusters create` / `node-pools create --machine-type=ct6e-…`, then `kubectl` |
| `gpu` | A general-purpose GPU attached to a VM, any cloud | varies (GCE, EC2) |
| `cloudrun` | A general-purpose GPU attached to cloudrun |  Cloud Run |
| `local` | A machine you physically own and are sitting at — no API to call, nothing to create, nothing to release | nothing; the hardware is already there |

These six are the permitted values. Cloud provider is deliberately not a slot; it's implied by the hardware
slot or recorded in the rig's `tpu.env`. Add a platform value only for a genuinely different execution target.

**`tpu`, `gce` and `gke` can name the same silicon, and that is the point.** `tpu-vllm-v6e1-2b`,
`gce-vllm-v6e1-2b` and `gke-vllm-v6e1-2b` are all one v6e-1 chip serving E2B under vLLM; they differ only in
which API provisions it. That difference earns a slot because Google is retiring one of them — **the Cloud TPU API is no longer
under active development, and TPU7x and later are Compute Engine or GKE only** (see `@HARDWARE.md`, which
holds the per-generation table and the migration mapping). A rig's provisioning path is therefore a fact with
a shelf life, and the name should carry it.

This revises the older reading of the slot as "the accelerator family, not the cloud" (2026-08-10). Three
consequences, none of which renames an existing rig:

- **Inferentia keeps `tpu`.** AWS Neuron is its own control plane, which is the same *shape* of thing as the
  Cloud TPU API, and both existing repos already spell it that way (`~/tpu-jax-inf2`,
  `~/tpu-pytorch-inf2-2b`). The part lives in the hardware slot as `inf2`. This is the one value where `tpu`
  does not literally mean the Cloud TPU API.
- **`gce` still covers CPU-only rigs**, its original meaning — the hardware slot settles which it is.
  `gce-<runtime>-cpu-<model>` is a CPU rig and `gce-<runtime>-v6e1-<model>` is a TPU-on-Compute-Engine rig,
  so nothing is ambiguous and no CPU rig would need renaming. (None exists yet.)
- **`gpu` wins for GPUs even though a GPU is also a machine-type property.** Mechanically an L4 on GCE is a
  Compute Engine instance and could read as `gce`, but no GPU rig has a second control plane to distinguish,
  and the hardware slot already carries the SKU. Reserve `gce` for the TPU-on-Compute-Engine case; a GPU rig
  is `gpu` wherever it runs.

**Only v5p, v6e and TPU7x can take `gce`, and for v5e that is now verified rather than inferred.**
`gcloud compute instances create --machine-type=ct5lp-hightpu-1t` was run on 2026-08-11 and rejected at
validation with `This user agent is not allowed to use the machine type [ct5lp-hightpu-1t]` — free,
conclusive, nothing created. So a `gce-*-v5e1-*` rig **is not buildable**, and the six v5e rigs stay `tpu`.

The catalog still muddies this and always will: `ct5lp-hightpu-1t` and friends do exist as machine types in
26 zones, the shared OS image family is named for v5e, and there is a Compute Engine v5-lite quota id. All
three are artefacts of the Cloud TPU API and GKE being implemented on Compute Engine underneath — none was
evidence, and the run settled it. **Do not re-derive a `gce` v5e rig from the catalog.** An earlier revision
argued the opposite way, from the missing `ct5lp-*-tpu` shape; that argument was also wrong, because the
*bare* shapes are the documented creatable ones. `@HARDWARE.md` §"Can v5e use the Compute Engine path?" has
the verbatim error and why its wording rules out zone and quota as explanations.

`gce-vllm-v5e1-2b` exists as the apparatus that settled this, and is the one rig here expected never to
provision. Keep it, and keep its name.
`@HARDWARE.md` has the generation table.

**That exclusion is about `gce`, and does not carry to `gke`** — the rejection names the *calling surface*
(`This user agent is not allowed to use the machine type`), and GKE is the surface that requests `ct5lp-*`
strings legitimately. See the next section.

### `gke` is a third control plane, not a flavour of `gce`

Added 2026-08-25 for `~/gemma4-dev/gke-vllm-v6e1-2b`. GKE is implemented *on* Compute Engine — a TPU node
pool is a set of Compute Engine VMs carrying the same `ct6e-standard-1t` machine type the `gce` rigs create
directly — so the tempting reading is that this is a `gce` rig that happens to run Kubernetes, with the
cluster recorded in the README. Three things make it its own value, and they are the same three that earned
`gce` its split from `tpu`:

- **A different API provisions the hardware.** `gcloud container clusters create` and `node-pools create`,
  not `gcloud compute instances create`. Capacity, provisioning model and failure modes are requested
  through the cluster, and a node you never asked for by name can appear — or be replaced — on the node
  pool's terms. `instances create` has no equivalent of that.
- **Discovery and access are a different shape again.** A GKE node *does* list under `gcloud compute
  instances list`, so the trap here is the opposite of the `gce` one: the discovery call succeeds and
  returns the wrong object. What serves the model is a Pod behind a Service, reached through `kubectl`
  and a port-forward or a LoadBalancer IP — not `gcloud compute ssh` to `:8000` on a `natIP`. Every
  helper a `gce` rig uses for endpoint resolution is wrong here in the same quiet way the TPU API's
  `_list_tpu_vm_nodes()` was wrong once the `gce` rigs forked off it.
- **It provisions, and it is not a `gce` create in disguise.** First run 2026-08-25: cluster plus a
  one-node `ct6e-standard-1t` pool in `europe-west4-a`, vLLM Ready ~10 minutes after apply, serving
  from a LoadBalancer on `:8000`. The node-pool create also refuses `--tpu-topology` for a
  single-host slice while GKE labels the node `gke-tpu-topology=1x1` regardless — a flag/label split
  that has no analogue on either of the other two control planes.
- **It is the only path with a future for v5e.** `instances create` refuses `ct5lp-hightpu-1t` outright and
  GKE is the consumer those machine types exist for (`@HARDWARE.md`), so `gce-*-v5e1-*` is unbuildable while
  `gke-*-v5e1-*` is the shape the six v5e rigs would have to take if the Cloud TPU API is sunset. **Untested
  here** — the v6e node pool above proves the path, but no v5e pool has been tried, so treat the v5e
  half as why the slot is worth having, not as a tested claim.

The slot names *how the accelerator is provisioned*, so `gke` is for a rig that creates its own cluster or
node pool. A manifest applied to a cluster someone else owns is not what it names. And the usual rule bites
hardest here, because the underlying VM is identical: a rig that ends up shelling to `instances create` is a
`gce` rig with a stale name, whatever Kubernetes is doing on top of it.

### `local` is the absence of a control plane

Added 2026-09-03 for the llama.cpp workstation deployment.

The other five values name an API you call to get hardware. `local` names the case where there is no such
call: the accelerator is in the machine under the desk, it is there whether or not any code runs, and
nothing in the rig provisions, waits for, or releases it. That passes this section's own test for a new
value — a genuinely different execution target — and it passes it in the direction that matters, because
**the entire provisioning half of a rig is absent** rather than merely spelled differently.

Four things follow, and together they are why this is a slot value rather than a line in a README:

- **There is no capacity to find.** Every other rig here carries machinery for a resource that might not
  exist yet — zone scanning, queued-resource state, the `~/.cache/<rig>/tpu_zones_status.md` skip list.
  A `local` rig has none of it, and a `local` rig that grows some has the wrong name.
- **Capacity is a hard ceiling, not a quota conversation.** A cloud rig that doesn't fit asks for a bigger
  machine type. A `local` rig that doesn't fit does not run. The card is the budget, and slot 3 is
  therefore load-bearing in a way it is nowhere else.
- **The endpoint is not discovered.** It is `127.0.0.1:<port>`, known before the process starts. This is the
  one place the "never hardcode an endpoint" rule in `CLAUDE.md` does not apply, because there is no
  QR → node → external IP chain to walk.
- **Nothing is billed and nothing needs tearing down**, so the care the other rigs take not to strand
  capacity is dead weight here.

`local` is a claim about who owns the machine, not about where you are typing. SSH into a cloud VM you
provisioned and it keeps that VM's platform value; a workstation is `local` whether or not you are sitting
in front of it.

## Slot 2 — runtime

The serving stack that actually loads the weights.

| Value | Means |
| --- | --- |
| `vllm` | vLLM, OpenAI-compatible API on `:8000` (Docker image or pip install) |
| `jax` | JAX-native — Flax, MaxText, or a hand-rolled JAX server |
| `jaxrust` | JAX-family graph execution driven from **Rust** — the forward pass is defined in Rust and lowered to XLA/StableHLO, not written in Python |
| `pytorch` | PyTorch, via `torch_xla` on TPU or CUDA on GPU |
| `llamacpp` | `llama-server` from `ggml-org/llama.cpp`, driven directly — one process, one GGUF file you named |
| `ollama` | The Ollama daemon, which **links llama.cpp as a library** and adds a registry, a blob store, its own chat-template renderer and model lifecycle management |

`vllm` names the server, not the backend it compiles to — vLLM on TPU is `vllm`, never `pytorch`, even though
`torch_xla` sits underneath. Pick the layer you'd file a bug against.

**`llamacpp` and `ollama` are the sharpest test of that rule anywhere in this table, because they are the same
engine.** Ollama ships `libllama.so`, and the Gemma 4 graph both execute is upstream `src/models/gemma4.cpp`.
They are still two slot values, because the layer you would file a bug against is genuinely different: an
`ollama` rig's artifact, chat template, CUDA variant and (possibly) engine are all chosen by the daemon, and
three of those four are things a benchmark can see. Verified 2026-09-02 — see either rig's `CLAUDE.md` for the
four differences and the byte counts behind them.

**Neither is spelled with its punctuation.** `llama.cpp` and `llama-cpp` both break slot parsing — the hyphen is
the field separator and the dot is not in the character set — so the value is `llamacpp`, by the same rule that
makes `jaxrust` one token. **And never `llama` on its own**: slot 2 is a stack, `llama` reads as the Llama model
family, and `gpu-llama-g5g-2b` invites exactly the misreading the slot exists to prevent.

**`jaxrust` is one slot value, not a compound, and it is never spelled `jax-rust`.** Slot values cannot contain
a hyphen, because the hyphen is the field separator: `gpu-jax-rust-g5g-2b` parses as a *five*-slot name whose
hardware slot is `rust` and whose model slot is `g5g`. The same rule is why `v5e1` is not `v5e-1`.

**Choose it by which language owns the model definition, not by what is linked in.** The `vllm` rule — file
the bug against the layer you named — decides this too. A Python JAX rig that loads a Rust extension is still
`jax`; a rig whose forward pass is written in Rust is `jaxrust`, even though XLA sits underneath both and it
is frequently the *same* `libxla_extension` binary. `pytorch` draws the identical line against `torch_xla`.

**`jaxrust` is not a performance claim.** Slot 2 has never been one — it records which stack loads the
weights, and two rigs differing only in slot 2 are an A/B pair precisely because the slot does not prejudge
the result. Rust buys memory-safety and a single static binary; the arithmetic still runs in XLA, so a
`jaxrust` rig and a `jax` rig on the same chip should be expected to land in the same place until measured
otherwise.

## Slot 3 — hardware

Accelerator generation plus chip count, punctuation stripped.

| Value | Means |
| --- | --- |
| `v5e1` | TPU v5e, 1 chip |
| `v5p1` | TPU v5p, 1 chip — only reachable via Compute Engine, see below |
| `v6e1` | TPU v6e (Trillium), 1 chip |
| `v6e4` | TPU v6e, 4 chips |
| `v6e8` | TPU v6e, 8 chips — one host on both control planes (`v6e-8` / `ct6e-standard-8t`) |
| `inf2` | AWS Inferentia2 |

For `gpu` platforms the slot depends on **who sells the machine**, and the split is
load-bearing rather than cosmetic:

- **On EC2 — the instance family.** `g5g`, `g4dn`, `g6`, `g6f`. This was a `g5g`-only carve-out
  until 2026-08-28; it is now the rule for every EC2 GPU rig. See below.
- **Everywhere else, `local` included — the GPU SKU.** `l4`, `a100`, `1650ti`: lowercase, no
  punctuation, consumer parts keeping their series digits and their suffix. This is what the five
  `gpu-vllm-l4-*` artifact rigs use and they are unaffected.

**Never the architecture or the compute capability.** `turing75` was weighed for the GTX 1650 Ti on
2026-09-03 and rejected. Compute capability 7.5 spans a 4 GB 1650 Ti and a 16 GB T4, so it drops the one
number that decides whether a checkpoint fits at all — and it is not even a single silicon capability: the
T4 is TU104 and has tensor cores, while the GTX 16-series is TU116/TU117 and has none. Naming this card
`turing75` would assert equivalence with the T4-based `g4dn` and `g5g` rigs in exactly the respect where it
differs. The architecture is a real fact; it belongs in `@HARDWARE.md`, which owns which numeric formats a
part natively supports. The name slot carries the part.

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
`v5litepod` spelling back to `v5e`.

**v5p breaks it a second way, and the answer depends on which control plane you use.** v5p slice names in the
*Cloud TPU API* count TensorCores, and a v5p chip has **two** — so its 4-chip slice is `v5p-8` and `v5p-4`
does not exist. A `v5p4` directory against a `v5p-8` flag differs by a factor of two, on purpose. v5e's trap
was a name that looked wrong; that one is a name that looks right and asks for twice the hardware.

**Compute Engine changes the available sizes, not the naming rule.** CE publishes `ct5p-hightpu-1t-tpu`, a
genuine 1-chip v5p, so `v5p1` *is* a valid rig — `~/gemma4-dev/tpu-vllm-v5p1-2b` is one — even though the
same rig is unbuildable on the TPU API, whose floor is 4 chips. The slot still means chips; what changed is
which chip counts you can buy. Either way the rule is the same and this is its sharpest case: **the slot is
documentation, `tpu.env` is configuration, and no gcloud value is ever derived from the directory name.** Siblings currently disagree on the env value itself
(`tpu-vllm-v5e1-2b` sets `ACCELERATOR_TYPE=v5litepod-1`, `tpu-pytorch-v5e1-12b` sets `v5e-1`), which is
exactly why the name can't be derived mechanically from it.

Chip count is *topology*, not tensor-parallel size. `v5e1` and `v5p1` are one chip each so both set
`TENSOR_PARALLEL_SIZE=1`; they match there but are separate settings, and a multi-chip rig's TP is a choice
about sharding rather than a fact the name carries.

## Slot 4 — model

Gemma 4 parameter size, lowercase `b`. Verified mappings from rigs that exist today:

| Slot | `MODEL_NAME` |
| --- | --- |
| `2b` | `google/gemma-4-E2B-it` (QAT build: `google/gemma-4-E2B-it-qat-w4a16-ct` → slot 5 `w4a16`) |
| `4b` | `google/gemma-4-E4B-it` |
| `12b` | `google/gemma-4-12B-it` (QAT build: `google/gemma-4-12B-it-qat-w4a16-ct` → slot 5 `w4a16`) |
| `26b` | `~/tpu-jax-26b` |
| `31b` | `~/tpu-jax-31b` |

Use **`2b`**, not `e2b` — every existing directory does, even though the checkpoint is `E2B` and the
benchmark-artifact slug is `gemma4-e2b`. (`~/gemma4-dev/tpu-pytorch-v5e1-12b/CLAUDE.md` lists `e2b` as a size
example; no directory follows it.) The `E`-prefixed form belongs to `MODEL_NAME` and benchmark filenames, not
the repo name.

The model slot is **size only**. Quantization used to have nowhere to go and got folded in here or dropped;
it now has its own optional slot below. `2b` means the 2B family whether the weights are bf16 or 4-bit — the
compression is slot 5's job, and the exact checkpoint is still whatever `MODEL_NAME` says.

## Slot 5 — encoding (optional)

**The encoding of the weights on disk**, when it isn't the reference release. This is the only optional slot
and most rigs leave it off.

One category, strictly: how the tensors are stored. Not how they were produced, and not how they are served.

| Value | Means | Example `MODEL_NAME` / artifact |
| --- | --- | --- |
| *(omitted)* | The reference instruction-tuned release, full precision | `google/gemma-4-E2B-it` |
| `w4a16` | 4-bit weights, 16-bit activations — the encoding of Google's QAT releases | `google/gemma-4-E2B-it-qat-w4a16-ct` |
| `q4_0`, `q8_0` | GGUF / llama.cpp block quant, named exactly as the file does | `gemma-4-E2B-it-Q4_0.gguf` |
| `int8`, `int4`, `fp8` | Numeric format, where nothing more specific applies | |
| `awq`, `gptq` | PTQ methods that define their own on-disk packing | |

**`qat` is not a value.** Quantization-aware training is a *training procedure*, not an encoding — a QAT
checkpoint can ship as w4a16, as int4, or dequantized back to bf16, so `qat` doesn't pin the artifact the way
every other value here does. Google's QAT release is `w4a16`; that it was trained quantization-aware is
recorded where provenance belongs, in `MODEL_NAME` (which literally spells `-qat-w4a16-ct`) and the rig's
README. This costs the discovery term — people search "Gemma QAT" — which is why the README has to say it.

**`ct` is not part of the value.** compressed-tensors is the container, not the encoding. It's the part that
can change while the tensors stay identical, it's the least likely thing to ever distinguish two rigs you'd
run side by side, and the hyphen in `w4a16-ct` would break slot parsing anyway.

There is deliberately **no `bf16` value.** Full precision is the omitted form, and a second spelling for it
would destroy the one thing the empty slot buys you — that a bare four-slot name positively means the
reference weights. `tpu-jax-v5e1-2b` is the bf16 rig.

Four rules:

- **Exactly one token, lowercase, no hyphen.** Upstream's `-qat-w4a16-ct` is three facts — procedure,
  encoding, container. The slot takes the encoding; `MODEL_NAME` carries the rest.
- **Name the encoding at the granularity the artifact does.** `w4a16` for the compressed-tensors releases,
  `q4_0`/`q8_0` for GGUF builds — a `q4_0` and a `q8_0` of the same model are two rigs and both need spelling
  out. Fall back to a bare numeric format (`int4`) only when the artifact offers nothing more specific.
- **Spell it whenever the weights aren't the reference build**, not only when a sibling forces it. The slot
  exists so the name stays a truthful claim about `MODEL_NAME`; a rig that quietly serves 4-bit weights under
  a bare `-2b` is the same stale-name problem as a retargeted accelerator.
- **Runtime parameters are not an encoding.** `kv_cache_dtype=fp8`, `--quantization`, `TENSOR_PARALLEL_SIZE`,
  `max_model_len` all live in `tpu.env`. An fp8 KV cache over bf16 weights is a runtime setting on unchanged
  tensors — that rig is not `-fp8`. `benchmarks/serving-report.schema.json` already draws this exact line:
  `model.quantization` and `model.kv_cache_dtype` are separate fields, and the slot tracks the first.
  Nor is tuning: `-it` is not an encoding, and neither is `pt`, `lora`, or a fine-tune name. Every rig here is
  instruction-tuned; if a base-weights or fine-tuned rig ever lands, it needs its own answer, not this slot.

The encoding is part of the directory name, so it flows into every derived name below — MCP server key, skill
stem, zone-status cache, plugin name. That is the point rather than a side effect: a `-w4a16` rig must not
share `~/.cache/<rig>/tpu_zones_status.md` or a `make skill-install` destination with its full-precision
sibling.

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
- The directory name is likewise the rig's **MCP server name** and the stem of its **skill name**. Three
  things derive from it, and each lands in a namespace shared with every sibling rig:

  | Derived name | Where it lands | Why it must be unique |
  | --- | --- | --- |
  | `<rig>` | MCP server key in `.mcp.json`, `.claude-plugin/plugin.json`, `.codex/config.toml` | Prefixes every tool as `mcp__<rig>__…`, so duplicates make a tool call ambiguous — and a user-scope entry shadows any project with no `.mcp.json` |
  | `<rig>-management` | `.claude/skills/`, `skills/`, `dist/<name>-skill.zip`, `~/.claude/skills/` | `make skill-install` does `rm -rf` on the destination, so a shared name means the last rig installed silently replaces the others |
  | `<rig>` | `~/.cache/<rig>/tpu_zones_status.md` | `find_tpu` skips zones it recorded as failed; rigs request different accelerator types, so one rig's failure is not evidence about another |

  All three are defaults, not locks — `MCP_SERVER_NAME`, `SKILL_NAME`, and `TPU_ZONES_STATUS_FILE` override
  them for a client or checkout already committed to a different name. Prefer fixing the client.
- Sweep the old name out of the repo: `grep -rn <old-name> --exclude-dir=.git --exclude-dir=dist`. Each rig is
  an independent fork, so the name is duplicated in docs, `.claude-plugin/` manifests, JSON-schema `$id`s, and
  hardcoded absolute `/home/xbill/<repo>/...` paths in test files.
- `~/gemma4-dev/tpu-pytorch-v5e1-12b/CLAUDE.md` carries the full per-file checklist plus the conventions for
  that rig's cloud resource ids.

## Benchmark artifact names

Benchmark files inside a rig use a **different, date-first scheme** — they name a measurement, not a rig, so
the rig-name rules above (fixed slot positions, a bounded hyphen count) do not apply:

```
benchmarks/reports/<date>-<model-short>-<hw-short>.json    2026-07-21-gemma4-e2b-v6e1.json
benchmarks/runs/<date>-<what>-<hw-short>/                  2026-07-25-vllm-sweep-v6e1/
```

| Part | Rule |
| --- | --- |
| `<date>` | ISO `YYYY-MM-DD`, the run date |
| `<model-short>` | family + checkpoint: `gemma4-e2b`, `gemma4-12b` — **`e2b`, not the `2b` of the model slot**. Append slot 5's token when the run measured a non-reference build: `gemma4-e2b-w4a16`, `gemma4-12b-q4_0` |
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
- **An encoding in `<model-short>` is likewise a claim about the weights measured, not the rig hosting the
  file.** A bare `gemma4-e2b` report sitting in a `-w4a16` rig measured the unquantized build unless the
  report's own `model.id` / `model.quantization` says otherwise — quantization is exactly the kind of
  difference a benchmark is run to expose, so never let the directory settle the question.

## Current inventory

**Conforming — all four mandatory slots:**

| Rig | Platform | Runtime | Hardware | Model | Variant |
| --- | --- | --- | --- | --- | --- |
| `~/gemma4-dev/gce-vllm-v6e1-2b` | **Compute Engine instance** (`ct6e-standard-1t`) | vLLM in Docker | v6e-1 | `gemma-4-E2B-it` | — — the A/B twin of `tpu-vllm-v6e1-2b`, see below |
| `~/gemma4-dev/gce-vllm-v6e8-2b` | **Compute Engine instance** (`ct6e-standard-8t`) | vLLM in Docker | v6e-8 | `gemma-4-E2B-it` | — — forked from `gce-vllm-v6e1-2b` and retargeted to eight chips 2026-08-19; the A/B twin of `tpu-vllm-v6e8-2b` |
| `~/gemma4-dev/gce-vllm-v6e8-31b` | **Compute Engine instance** (`ct6e-standard-8t`) | vLLM in Docker | v6e-8 | `gemma-4-31B-it` | — — forked from `gce-vllm-v6e8-2b` and retargeted from E2B to 31B 2026-08-25. **Bare four-slot name is the positive claim**: the reference bf16 release, not the `-qat-w4a16-ct` or `-q4_0-unquantized` export. Not an A/B twin of anything — nothing here serves 31B through the TPU API |
| `~/gemma4-dev/gke-vllm-v6e1-2b` | **GKE node pool** (`ct6e-standard-1t`) | vLLM in Docker | v6e-1 | `gemma-4-E2B-it` | — — third control plane for the chip `tpu-vllm-v6e1-2b` and `gce-vllm-v6e1-2b` already serve; **the name is a claim about work not yet done**, see below |
| `~/gemma4-dev/tpu-vllm-v5e1-2b` | TPU Queued Resource | vLLM in Docker | v5e-1 | `gemma-4-E2B-it` | — |
| `~/gemma4-dev/tpu-vllm-v5e1-2b-q4_0` | TPU Queued Resource | vLLM in Docker | v5e-1 | `gemma-4-E2B-it-qat-q4_0-unquantized` | `q4_0` |
| `~/gemma4-dev/tpu-vllm-v5e1-2b-w4a16` | TPU Queued Resource | vLLM in Docker | v5e-1 | `gemma-4-E2B-it-qat-w4a16-ct` | `w4a16` — the `-ct` container is not part of the slot |
| `~/gemma4-dev/tpu-vllm-v5p1-2b` | **GCE instance** (not the TPU API) | vLLM in Docker | v5p, 1 chip (`ct5p-hightpu-1t-tpu`) | `gemma-4-E2B-it` | — |
| `~/gemma4-dev/tpu-vllm-v6e1-2b` | TPU Queued Resource | vLLM in Docker | v6e-1 | `gemma-4-E2B-it` | — |
| `~/gemma4-dev/tpu-vllm-v6e8-2b` | TPU flex-start VM | vLLM in Docker | v6e-8 | `gemma-4-E2B-it` | — |
| `~/gemma4-dev/tpu-jax-v5e1-2b` | TPU | JAX | v5e-1 | 2B | — |
| `~/gemma4-dev/tpu-jax-v6e1-2b` | **GCE instance** (`ct6e-standard-1t`) | JAX | v6e-1 | `gemma-4-E2B-it-qat-w4a16-ct` | — — **slot 1 is a deliberate exception, see below** |
| `~/gemma4-dev/tpu-jax-v6e1-12b-w4a16` | TPU | JAX | v6e-1 | `gemma-4-12B-it-qat-w4a16-ct` | `w4a16` — **artifact rig**, see below |
| `~/gemma4-dev/tpu-jax-v6e1-26b-q4_0` | TPU | JAX | v6e-1 | `gemma-4-26B-A4B-it-qat-q4_0-unquantized` | `q4_0` — **artifact rig**, see below |
| `~/gemma4-dev/tpu-jax-v6e1-31b-w4a16` | TPU | JAX | v6e-1 | `gemma-4-31B-it-qat-w4a16-ct` | `w4a16` — **artifact rig**, see below |
| `~/gemma4-dev/gpu-vllm-g5g-2b` | **EC2 G5g** (Graviton2 host) | vLLM | g5g (Graviton2 + NVIDIA T4G, Turing) | `gemma-4-E2B-it` | — — hardware slot is the instance family, not the GPU SKU; carve-out below |
| `~/gemma4-dev/gpu-jax-g5g-2b` | **EC2 G5g** (Graviton2 host) | pure JAX | g5g (Graviton2 + NVIDIA T4G, Turing) | `gemma-4-E2B-it` | — — same carve-out as the vLLM sibling; serves the dense reference build, so no encoding slot |
| `~/gemma4-dev/gce-jaxrust-v6e1-2b` | **Compute Engine instance** (`ct6e-standard-1t`) | Rust-driven XLA (`rlx`) | v6e-1 | `gemma-4-E2B-it` | — — forked from `tpu-jax-v6e1-2b` 2026-08-28. **Slot 1 is `gce`, which resolves that rig's documented slot-1 exception rather than inheriting it** — it provisions only through Compute Engine, so the letter of the rule applies. Carries a Rust `rlx` engine under `rust/`; not an A/B twin of the `gpu` jaxrust rig (different chip *and* different backend) |
| `~/gemma4-dev/gpu-jaxrust-g5g-2b` | **EC2 G5g** (Graviton2 host) | Rust-driven XLA | g5g (Graviton2 + NVIDIA T4G, Turing) | `gemma-4-E2B-it` | — — forked from `gpu-jax-g5g-2b` 2026-08-28, same carve-out. **The name is a claim about work not yet done**: the engine is still the Python JAX one and the Rust runtime is unresolved, see below |
| `~/gemma4-dev/gpu-jax-g4dn-2b` | **EC2 G4dn** (x86_64 host) | pure JAX | g4dn (Intel/AMD + NVIDIA T4, Turing) | `gemma-4-E2B-it` | — — added 2026-08-28, forked from `gpu-jax-g5g-2b`. Hardware slot is the instance family per the EC2 rule below. Created as `gpu-jax-t4-2b` and renamed the same day |
| `~/gemma4-dev/gpu-jax-g6-2b` | **EC2 G6** (x86_64 host) | pure JAX | g6 (NVIDIA L4, Ada) | `gemma-4-E2B-it` | — — added 2026-08-28, forked from `gpu-jax-g5g-2b`. Family rather than SKU also keeps `g6` distinct from `g6f`, which is half an L4. Created as `gpu-jax-l4-2b` and renamed the same day |
| `~/gemma4-dev/gpu-vllm-g6-2b` | **EC2 G6** (x86_64 host) | vLLM | g6 (NVIDIA L4, Ada) | `gemma-4-E2B-it` | — — added 2026-08-28, forked from `gpu-vllm-g5g-2b`. **Slot 2 is the only slot that differs from `gpu-jax-g6-2b`**, which is what an A/B pair looks like — and it is the cleanest one in this tree, because both sides run stock (the T4G pair was not: its vLLM side used hand-reduced Triton tiles). Note this rig is the SAME SILICON as the five `gpu-vllm-l4-*` artifact rigs and the SAME RUNTIME, and still takes `g6` rather than `l4` under the EC2 rule below; do not read their numbers as its own |
| `~/gemma4-dev/gpu-vllm-g4dn-2b` | **EC2 G4dn** (x86_64 host) | vLLM | g4dn (Intel + NVIDIA T4, Turing) | `gemma-4-E2B-it` | — — added 2026-08-29, forked from `gpu-vllm-g6-2b` into a directory that was a stale copy of `gpu-jax-g4dn-2b`. **Slot 2 is the only slot that differs from `gpu-jax-g4dn-2b`** — an A/B pair — and slot 3 is the only slot that differs from `gpu-vllm-g5g-2b`, which is the pair that matters here: same Turing silicon, `aarch64` against `x86_64`. **That is the EC2-family rule earning its keep**, since `t4`-something for both would hide the only difference between them, and the difference is the finding: x86_64 pulls the amd64 manifest, which carries SM 7.5, so the aarch64 packaging gap vanishes while the Turing shared-memory ceiling does not |
| `~/gemma4-dev/gpu-pytorch-g5g-2b` | **EC2 G5g** (Graviton2 host) | PyTorch / `transformers` | g5g (Graviton2 + NVIDIA T4G, Turing) | `gemma-4-E2B-it` | — — added 2026-08-28, forked from `gpu-jax-g5g-2b`. Same `g5g` carve-out as its siblings; **slot 2 is the only slot that differs from the JAX rig**, which is exactly what NAMING.md says an A/B pair looks like |
| `~/gemma4-dev/gpu-pytorch-g4dn-2b` | **EC2 G4dn** (x86_64 host) | PyTorch / `transformers` | g4dn (Intel + NVIDIA T4, Turing) | `gemma-4-E2B-it` | — — added 2026-08-29, forked from `gpu-pytorch-g5g-2b` into a directory that was a stale copy of `gpu-jax-g4dn-2b`. **The fourth corner of the {jax, pytorch} x {g5g, g4dn} square**, so it differs from `gpu-pytorch-g5g-2b` in slot 3 only and from `gpu-jax-g4dn-2b` in slot 2 only. The latter pair is the point: identical hardware, so it is the cleanest runtime A/B in the tree — cleaner than the G5g pair, whose vLLM side used hand-reduced Triton tiles. The EC2-family rule earns its keep again here, since a `t4`-something slot would make this name collide with `gpu-pytorch-g5g-2b`'s silicon while hiding the host, and the host is what the two AMI lines differ on |
| `~/gemma4-dev/tpu-jax-inf2-2b` | Inferentia (slot `tpu`) | pure JAX (`jax-neuronx`) | inf2 | `gemma-4-E2B-it` | — — added 2026-08-28. Slot 3 is unaffected by the EC2 rule: `inf2` is both the part name and the instance family, so both readings agree |
| `~/gemma4-dev/local-llamacpp-1650ti-2b-q4_0` | **Local workstation** — no control plane, nothing provisioned | `llama-server`, driven directly | 1650ti (TU117, sm_75, 4096 MiB) | `gemma-4-E2B-it-qat-q4_0-gguf` | `q4_0` — added 2026-09-03, **the first `local` rig**, and the slot-1 value was added for it. Not a fork of anything. **Slot 3 is `1650ti` and deliberately not `turing75`**: sm_75 is shared with the T4-based `g4dn`/`g5g` rigs, but the T4 is TU104 and has tensor cores while TU116/TU117 has none, so the architecture would assert an equivalence that does not hold — and it would drop the 4096 MiB figure, which is the binding constraint on a `local` rig where capacity is a ceiling rather than a quota conversation. **Slot 5 is the weakest claim in this table**: the artifact is 67.7% Q6_K and only 31.4% Q4_0, so `q4_0` is what `MODEL_NAME` calls the export, not a description of the tensors. **Slots 1 and 3 are the only slots that differ from `gpu-llamacpp-g5g-2b-q4_0`**, which serves the same artifact through the same runtime — so the pair is a local-vs-cloud A/B on identical weights, and the two rigs currently *disagree* on whether `per_layer_token_embd` occupies VRAM — **settled 2026-09-03 in this rig's favour by measurement**, and the sibling corrected. **First light 2026-09-03**, and a full sweep the same day: 73.75 tok/s single-stream decode, 1618 MiB of 4096, and end-to-end serving throughput capped at ~45-48 tok/s because prefill does not batch |
| `~/gemma4-dev/gpu-vllm-l4-2b-w4a16` | NVIDIA L4 | vLLM | l4 | `gemma-4-E2B-it-qat-w4a16-ct` | `w4a16` — **artifact rig**, see below |
| `~/gemma4-dev/gpu-vllm-l4-4b-w4a16` | NVIDIA L4 | vLLM | l4 | `gemma-4-E4B-it-qat-w4a16-ct` | `w4a16` — **artifact rig**, see below |
| `~/gemma4-dev/gpu-vllm-l4-12b-w4a16` | NVIDIA L4 | vLLM | l4 | `gemma-4-12B-it-qat-w4a16-ct` | `w4a16` — **artifact rig**, see below |
| `~/gemma4-dev/gpu-vllm-l4-26b-w4a16` | NVIDIA L4 | vLLM | l4 | `gemma-4-26B-A4B-it-qat-w4a16-ct` **(no such Hub id — see the rig)** | `w4a16` — **artifact rig**, see below |
| `~/gemma4-dev/gpu-vllm-l4-31b-w4a16` | NVIDIA L4 | vLLM | l4 | `gemma-4-31B-it-qat-w4a16-ct` | `w4a16` — **artifact rig**, see below |
| `~/gemma4-dev/tpu-pytorch-v5e1-2b` | TPU | PyTorch / `torch_xla` | v5e-1 | 2B | — |
| `~/gemma4-dev/tpu-pytorch-v5e1-12b` | TPU | PyTorch / `torch_xla` | v5e-1 | `gemma-4-12B-it-qat-w4a16-ct` | **`w4a16` — name stale, see below** |
| `~/tpu-jax-v6e1-2b` | TPU | JAX | v6e-1 | 2B | — |
| `~/tpu-pytorch-v6e1-2b` | TPU | PyTorch | v6e-1 | 2B | — |
| `~/tpu-pytorch-inf2-2b` | Inferentia (slot `tpu`) | PyTorch | inf2 | 2B | — |

### The one rig whose slot 1 does not follow the rule

`~/gemma4-dev/tpu-jax-v6e1-2b` provisions **only** through Compute Engine — it holds no
queued-resource tools at all — so by the letter of slot 1 it should be `gce-jax-v6e1-2b`, exactly as its
`gce-vllm-v6e1-2b` sibling is. The directory name was kept on 2026-08-18 by an explicit decision, with
the trade-off understood.

This is recorded rather than quietly tolerated, and it is **not a precedent**. The rule stands: a new
Compute-Engine-provisioned TPU rig takes `gce`. Do not cite this row as evidence that `tpu` may name a
Compute Engine rig, and do not "correct" that rig's `server.py` to match its name — the name is the
exception and the code is right. Its own `CLAUDE.md` says the same.

Adding slot 5 makes exactly one existing name stale: **`tpu-pytorch-v5e1-12b` serves 4-bit weights and should
be `tpu-pytorch-v5e1-12b-w4a16`.** It has no full-precision sibling, so nothing is broken today and the
old name is not a counter-example to keep — it is a rename owed. The cost is the full sweep under
[The name is a claim](#the-name-is-a-claim-about-tpuenv): directory, plugin name in both `marketplace.json`
files and its `plugin.json`, MCP server key in `.mcp.json` / `.codex/config.toml` /
`enabledMcpjsonServers`, skill stem under `.claude/skills/` + `skills/` + `dist/`, the `~/.cache/<rig>/` zone
file, and the hardcoded `/home/xbill/tpu-pytorch-v5e1-12b/…` paths in `ports/**/*_test.py`. Do it as its own
commit, not folded into unrelated work.

### Three rigs, one chip: the provisioning A/B

`tpu-vllm-v6e1-2b`, `gce-vllm-v6e1-2b` and `gke-vllm-v6e1-2b` are deliberately identical in slots 2–4 — same
runtime, same v6e-1 chip, same E2B checkpoint — and differ only in slot 1. They exist as a matched set so the
provisioning paths can be compared directly: same model, same hardware, same serving flags, three control
planes. Keep them in step. A change to one that isn't about provisioning (serving flags, `MAX_MODEL_LEN`,
benchmark harness) should land in all three, or the comparison stops being one.

**`gke-vllm-v6e1-2b` was named on 2026-08-25 before it was built, and the code caught up the same day.**
It was forked verbatim from `gce-vllm-v6e1-2b`, so for a few hours slot 1 was a claim about shell scripts
while `server.py` still shelled to `instances create` — the weaker of the two readings this scheme allows.
That gap is closed: the Compute Engine path was removed from the rig, `server.py` provisions
`gcloud container node-pools`, and the create → list → destroy round trip was verified through the MCP
tools. The name is a claim about `server.py` again.

Two things worth keeping from how it went, because both generalise to any future rig fork:

- **A fork inherits its parent's control plane in more places than the obvious one.** Removing it here meant
  `server.py`, the Makefile's `deploy-tpu*` targets, `startup_script_template.sh`, and the test that
  *asserted* the rig was off the old path while only checking one function.
- **The name being ahead of the code is recoverable; the name being wrong about the code is not.** Recording
  the gap in this file while it existed is what made it a task rather than a discrepancy someone would later
  find and "fix" by renaming the directory.

### Artifact rigs — a rig that serves nothing

The three `tpu-jax-v6e1-*` rows above hold **measurements and findings only**: no `server.py`, no MCP
server, no skill, no plugin manifest, no `tpu.env`. They exist so that results measured on hardware
no serving rig covers have a home that names that hardware, instead of being filed under a rig they
did not come from — the mistake `benchmarks/rollup.py` was written to expose.

Two consequences for naming:

- **The name is still a full four-or-five-slot name, and slot 5 still applies.** These rigs have no
  `MODEL_NAME` to be a claim about, so the encoding slot is a claim about the **weights measured**.
  Leaving it off would positively assert the reference build, which is false for all three — hence
  `-w4a16` and `-q4_0` rather than bare `-12b` / `-26b` / `-31b`.
- **None of the derived-name machinery applies**, because there is nothing to derive it for: no
  `.mcp.json` key, no `~/.cache/<rig>/tpu_zones_status.md`, no `make skill-install` destination. The
  one thing that *is* derived is discovery — `benchmarks/rollup.py` globs `*/benchmarks/`, so an
  artifact rig appears in `ROLLUP.md` and gets a generated `INDEX.md` like any other.

Do not scaffold an artifact rig into a full one to make it match its siblings. If a serving rig for
that size is ever built, the name is already correct for it.

Note that `v6e1` in `tpu-jax-v6e1-26b-q4_0` names the **target**, not a measurement — that port was
verified on CPU and has never run on a TPU. The rig's `CLAUDE.md` says so at the top. An artifact
rig's hardware slot is the weaker claim of the two whenever its `REPORT.md` marks its figures as
projections.

### `gpu` is a real platform slot, and the cloud is still not one

The five `gpu-vllm-l4-*` rigs are the first users of platform slot `gpu`. They came from
`~/gemma4-tips` / `~/gemma4-tips-aws` on 2026-08-07 and are artifact rigs on the same terms as the
`tpu-jax-v6e1-*` ones above.

**Cloud Run, GCE and EC2 all reduce to hardware slot `l4`.** That is the rule at the top of this file
working as intended — cloud provider is not a slot, and the old scheme's host slot (`cloudrun`, `ec2`)
has no home in a rig name. The host is recorded in two places that *are* evidence: the run directory
name (`2026-06-15-vllm-grid-ec2-l4`) and the report's own `Endpoint:` line. Resolve disagreements in
favour of the endpoint — one migrated 12B report is labelled "Cloud Run" while its endpoint is a bare
public IP in the AWS tree.

Migrating that tree needed **both** the runtime-slot and the encoding-slot cautions in the legacy table
below, and one more that is specific to it: the source directories duplicated each other's artifacts so
heavily (82 report files, 20 unique) that a directory name there is not evidence of the model it
served. Only reports whose own `Model:` and `Endpoint:` lines agreed were migrated — 10 of the 20.
Filling slot 4 or slot 5 by reading a `~/gemma4-tips` directory name would have produced five rigs
misattributed at once.

### EC2 GPU rigs take the instance family, not the GPU SKU

**Widened from a `g5g`-only carve-out to the general EC2 rule on 2026-08-28**, when
`gpu-jax-g4dn-2b` and `gpu-jax-g6-2b` were added. They were first created as `gpu-jax-t4-2b` and
`gpu-jax-l4-2b` under the SKU reading below, and renamed the same day. **Do not "correct" any of
them back to a SKU.**

The original carve-out covered `gpu-vllm-g5g-2b` (2026-08-12) and `gpu-jax-g5g-2b` (2026-08-18) —
the first `gpu` rigs that were *serving* rigs rather than artifact rigs, and the first whose
hardware was not an L4. The two arguments below were written to justify `g5g` specifically. The
first turned out to be narrow and the second turned out to be general, which is why the rule
widened rather than the exception multiplying.

The default reading would be `t4g`: the GPU is an NVIDIA **T4G**, slot 3 says "GPU SKU, lowercase, no
punctuation," and `g5g` is an instance family — the same category as `ec2` and `cloudrun`, which this
file excludes precisely because they say nothing about the silicon. Two facts beat it:

- **`t4g` is already taken, by CPUs.** AWS ships `t4g.nano`…`t4g.2xlarge`: Graviton2 **burstable CPU**
  instances with no GPU at all. They sit in the same instance-family namespace as `g5g` — AWS's own
  Graviton inventory lists `T4g` and `G5g` side by side. So `t4g` is the one GPU SKU whose spelling
  collides head-on with a well-known GPU-less instance type. **This argument is narrow** — `l4` and
  `a100` have no such problem — and on its own it would justify only `g5g`. The second argument is
  the one that generalizes.
- **`g5g` loses no information, because the family is 1:1 with the silicon.** The usual objection to a
  family name is that it is lossy — `ec2` could be any chip. G5g is the **only** Graviton+GPU family
  AWS has ever shipped, and there is no Graviton3 or Graviton4 GPU instance, so `g5g` names exactly
  one CPU+GPU pairing: Graviton2 + T4G. It is a *more* specific claim than `t4g`, not a vaguer one,
  because it pins the aarch64 host as well.

That second point is what makes the carve-out earn its keep rather than merely be tolerable. **The
host architecture is load-bearing on this rig in a way it is on no other.** The reason nothing here
runs out of the box is that no published CUDA build covers aarch64 *and* SM 7.5 together
(`@HARDWARE.md`). A slot value naming only the GPU would describe the half of the problem that is not
the problem.

Two limits on how far this generalizes:

- **This does not reopen the cloud slot.** Cloud provider is still not a slot, and `ec2`, `cloudrun`
  and `gce` are still not values for slot 3. What earns `g5g` its place is the 1:1 mapping to silicon,
  which no cloud name has.
- **On EC2 it now DOES license the family name, and that is the point.** The family is the unit you
  actually provision, quota and pay for — you request a `g6.xlarge`, never "an L4" — and it pins the
  host architecture, which the SKU does not. `g5g` and `g4dn` are the same Turing-class silicon on
  `aarch64` and `x86_64` respectively; naming both `t4`-something would hide the only difference
  between them, which is the entire reason the pair exists as an A/B. The family also separates
  `g6` from `g6f`, a whole L4 from half of one, which `l4` cannot.
- **A family that stops being 1:1 with its silicon needs revisiting.** If AWS ever ships a second
  Graviton+GPU family, `g5g` loses the property that earned it. Record that here if it happens.

The chip keeps its own name everywhere it is the chip under discussion — `@HARDWARE.md`'s section is
headed T4G, and so is the rig's `tpu.env`. Only the slot is the family. The same holds for the newer
pair: `gpu-jax-g6-2b` talks about the L4 and about Ada throughout its README, and its test class is
`AdaConstraintTests`, because those are claims about the chip rather than about the rig.

**Predates the scheme — not counter-examples:**

| Rig | Missing |
| --- | --- |
| `~/tpu-jax-4b`, `-12b`, `-26b`, `-31b` | hardware slot — **12b/26b/31b artifacts migrated** to the three artifact rigs above on 2026-08-07; the source repos keep the engine code and are unchanged |
| `~/tpu-jax-inf2` | model slot — **artifacts migrated** to `tpu-pytorch-inf2-2b` on 2026-08-07 |
| `~/tpu-jax` | hardware and model slots |
| `~/tpu-skill-claude`, `-codex`, `-agy` | not rigs — skill repos |
| `~/tpu-inference` | not a rig |

**`~/gemma4-tips` and `~/gemma4-tips-aws`** use the same legacy `-devops-agent` form and are **not a
naming reference**: their directory names misattribute models and hardware on a large scale
(`g2-48-26B-qat-L4` holds the *31B* report; one 12B report sits in 13 directories spanning 2B–31B).
The citable artifacts were migrated to the five `gpu-vllm-l4-*` rigs on 2026-08-07; the rest carry no
model line and are unattributable. Do not translate a name from this tree — read the report's own
`Model:` and `Endpoint:` lines.

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
| `g2-4-2B-qat-L4-devops-agent` | `gpu-<runtime>-l4-2b-<encoding>` (the old name records QAT, which is provenance, not an encoding — **read `MODEL_NAME` for the actual format** before filling slot 5, same caveat as the runtime slot) |

The old scheme's host slot (`ec2`, `cloudrun`) has no home in this scheme; that detail belongs in the
project's README and env file, not its directory name.

## Adding a rig

1. Name the directory from the four mandatory slots. For slot 1, pick the **control plane you will actually
   provision through** — `tpu` for the Cloud TPU API, `gce` for a Compute Engine TPU machine type, `gke` for
   a GKE node pool, `local` for hardware you already own and provision nothing for. This is a claim about the code in `server.py`, and a rig that switches paths needs the
   rename. Three of these can name one chip, so slot 1 is the only thing telling them apart.
2. Add slot 5 if the weights aren't the reference build — one lowercase token naming the **encoding**
   (`w4a16`, `q4_0`, `int4`), read off `MODEL_NAME`. Not `qat`, not the container, not a runtime flag.
3. Check the name is unique. Two rigs differing only in something no slot captures (zone, batch settings,
   KV-cache dtype) need a suffix of their own, not a reused name and not a misused encoding slot.
4. Create the GitHub repo under the **same** name — don't inherit a shorter remote from the rig you forked.
5. Set the authoritative values in `tpu.env` (`MODEL_NAME`, `ACCELERATOR_TYPE`, `TENSOR_PARALLEL_SIZE`, zone).
   The name describes; the env file decides — including the encoding, which must match `MODEL_NAME`.
6. Add a row to **Current inventory** above, filling the Variant column with `—` if there isn't one.
