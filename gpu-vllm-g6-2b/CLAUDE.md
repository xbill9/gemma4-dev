# CLAUDE.md — `gpu-vllm-g6-2b`

Serving rig: **`google/gemma-4-E2B-it`** under **vLLM** on **AWS EC2 G6** — an **x86_64**
host with an **NVIDIA L4** GPU (Ada, SM 8.9, 24 GB nominal / **23034 MiB** measured by the
JAX sibling on the same silicon).

This is a full rig: `server.py`, an MCP server, a skill, a plugin manifest, and `tpu.env`.
It is **not** one of the `gpu-vllm-l4-*` artifact rigs, despite sharing the `gpu` platform
slot, the runtime slot **and the actual GPU** with them. See **Measurement**.

> **THIS RIG SERVED FOR THE FIRST TIME ON 2026-08-30**, on `g6.2xlarge` spot in `us-east-1d`,
> and was torn down the same session. `benchmarks/runs/2026-08-30-first-serve-g6/` and
> `benchmarks/reports/2026-08-30-gemma4-e2b-g6.json` are its own, measured on its own hardware
> slot. **Both of the fork's open premises held** — the published image covers SM 8.9, and
> Triton's 512-wide tile fits Ada unpatched. Anything NOT marked MEASURED below is still
> arithmetic or inherited.

## Why this rig exists

**It is the runtime control for `gpu-jax-g6-2b`.** That rig MEASURED **48.3–48.5 tok/s** on
this exact silicon under pure JAX on 2026-08-28. Same chip, same checkpoint, different
runtime — which would be **the only clean runtime comparison in this tree.**

The T4G pair was never clean — though **not for the reason written here until 2026-08-30.**
The tile clamp was blamed, but it applies to *every* vLLM-on-T4G number including the good
ones, so it never distinguished anything. The real defect: **43.1/44.24 are not benchmarks.**
43.1 is one sample from a first-serve smoke test; 44.24 has no benchmark artifact at all and
was measured to prove a swapfile lets `g5g.xlarge` boot. The T4G pair does have a clean
comparison — `gpu-vllm-g5g-2b/benchmarks/runs/2026-08-14-rust-frontend-g5g/`, three runs of
`vllm bench serve` on one host: c=1 TPOT 31.44 ms (~31.8 tok/s decode), c=8 168.33 tok/s.
Here both sides run stock.

**REALISED 2026-08-30: vLLM 0.28.0 measured 46.09 tok/s single-stream against the JAX rig's
48.3–48.5 — about 5% slower, both sides stock.** No Triton patch, no hand-reduced tiles, no
from-source build on either side.

**Single-stream is the wrong place to judge vLLM, and this run shows why:** 46.09 → 360.17
tok/s from concurrency 1 to 8 is **7.8x on 8x the load (98% scaling efficiency)**, TPOT nearly
flat (19.85 → 21.71 ms). So the runtime question is not "which is faster" but "at what
concurrency" — and **nothing has measured the JAX side under load, so that half of the
comparison still does not exist.**

Against that clean T4G sweep, same harness and same runtime: **1.45x at c=1 (31.8 → 46.09),
1.81x at c=4 (~97 → 175.83), 2.14x at c=8 (168.33 → 360.17).** The gap widens with
concurrency. Caveat that survives: the T4G side ran with hand-reduced tiles, this side is
stock.

## The fork deletes the sibling's reason for existing

`gpu-vllm-g5g-2b` is a hard rig. This one should not be, and the difference is entirely
packaging.

**G5g needs aarch64 and SM 7.5 together, and no published CUDA artifact provides both.**
Read from the published image config on 2026-08-12:

| Manifest | `TORCH_CUDA_ARCH_LIST` | SM 7.5? | SM 8.9? |
| --- | --- | :---: | :---: |
| `linux/amd64` | `7.5 8.0 8.6 8.9 9.0 10.0 12.0` | **yes** | **yes** |
| `linux/arm64` | `8.0 8.7 8.9 9.0 10.0 11.0 12.0` | no | yes |

vLLM's Dockerfile sets no `+PTX`, so there is no JIT fallback — an unsupported device fails
with `no kernel image is available for execution on the device`.

**G6 is x86_64 and SM 8.9. It wants the amd64 manifest, and that manifest carries 8.9.**
Both axes are covered. So this rig has:

- **no from-source build** (the sibling spends ~67 minutes),
- **no CUDA toolkit** to install from the sbsa repo,
- **no Rust toolchain** for `vllm-rs`,
- **no prebuilt AMI to maintain**, and
- **no `serving=` mode.** The sibling's `build`/`stock` split is gone: `build` would have
  nothing to do and `stock` nothing to fail at. A parameter whose only value is the default
  is a knob that suggests a choice nobody has.

**Do not copy `ami-0b44b90b3d02430ee` into anything here.** That is the sibling's prebuilt
image: arm64, SM 7.5, and carrying a Triton patch. It cannot boot a G6.

## The one thing that might still bite: Triton shared memory

This is the sibling's *actual* blocker, and unlike the packaging half it is **not obviously
gone**. It is the first thing to check here.

Gemma 4's attention is heterogeneous — sliding **256**, global **512** — and vLLM handles that
with FA4 or Triton only. FA4 being unavailable, Triton is **forced** and cannot be overridden:

```
triton.runtime.errors.OutOfResources: shared memory, Required: 98304, Hardware limit: 65536
Gemma4 model has heterogeneous head dimensions
{'sliding_attention': 256, 'full_attention': 512}. FA4 not available, forcing TRITON_ATTN.
```

Triton at `head_size=512` wants **~96 KiB (98,304 B) of shared memory per block**.

| | per-block shared memory | 96 KiB tile fits? |
| --- | ---: | :---: |
| Turing (SM 7.5) | 64 KiB | **no** — 65,536 < 98,304 |
| **Ada (SM 8.9)** | **~99 KiB** | **expected yes — UNVERIFIED** |
| Ampere (SM 8.0) | 164 KiB | yes |

**SETTLED 2026-08-30: IT FITS, UNPATCHED.** vLLM forced `TRITON_ATTN` exactly as predicted,
and the engine started with **no `OutOfResources`**. Ada's ~99 KiB cleared the ~96 KiB tile.
**The sibling's patch to `vllm/v1/attention/ops/triton_unified_attention.py` is NOT needed
here and must not be ported in** — it lives only on that rig's instance, is not upstream, and
is not in this repo.

The margin is still ~3 KiB, so this is a fact about *this* tile at *this* head size, not a
general Ada guarantee: a larger `head_size` or a retiled Triton kernel could re-cross it.
Note also that on Turing the *static* limit is 48 KiB and a kernel must opt in to the dynamic
attribute to reach 64; do not assume the nominal figure is what a kernel gets.

**`VLLM_ATTENTION_BACKEND` is deliberately unpinned here.** Pinning a backend is how the
sibling ended up carrying a patch. Two further facts, MEASURED there 2026-08-12: vLLM v0.27
**does not recognize the variable at all** (`Unknown vLLM environment variable detected`), and
it forces `TRITON_ATTN` for this model regardless. Setting it did nothing.

## Ada is not Turing — the dtype policy inverted at the fork

**Turing has no bf16 and no fp8 datapath. Ada has both.** The sibling's settings encode those
absences, so copying them here is wrong in both directions.

| | T4G sibling (SM 7.5) | **this rig (SM 8.9)** |
| --- | --- | --- |
| compute dtype | `float16` | **`bfloat16`** |
| KV cache dtype | `auto` → float16 | **`auto` → bfloat16**; fp8 reachable, unused |
| device memory | 15360 MiB | **23034 MiB** |
| per-block shared memory | 64 KiB | **~99 KiB** |

**`DTYPE=bfloat16`, and the reason is the checkpoint, not the datasheet.** E2B ships bf16.
Setting float16 would make vLLM convert every weight on load — and `gpu-jax-g6-2b` MEASURED
exactly that mismatch costing **54% of decode** on Turing, dropping to **0.0%** once storage
and compute dtype agreed. Matching the checkpoint is the cheap default.

On the sibling, note, bfloat16 did not *fail* — PyTorch upconverts and vLLM logs
`Casting torch.bfloat16 to torch.float16` and proceeds. Silent cost, not an error.

**fp8 KV is newly reachable and is deliberately off.** MEASURED 2026-08-30: vLLM allocated
**9.65 GiB of KV = 1,076,849 tokens = 65.73x concurrency at 16K context**, so KV is nowhere
near the binding constraint, and nothing here has measured fp8's accuracy cost.

**That works out to 9622 B/token (9.40 KiB), and `@MODELS.md` derives 18,432 B — a 1.92x gap
that is NOT resolved.** Do not "fix" either number from the other. The 18 KiB figure is
derived from layer geometry (12 sliding x 256 + 3 full x 512) and cross-checked *exactly*
against `total_hbm_avail_gb` on v5e-1 and to 0.1% on v6e-1, so it is not loose arithmetic.
The 9.40 KiB figure is a **divided-out average** — vLLM's reported pool over vLLM's reported
token capacity — on a **different serving path** (vLLM CUDA, not `tpu_inference`), and the
likely explanation is that vLLM v1 charges sliding-window layers their *window* rather than
the full context, which makes the two numbers answer different questions. **Whoever needs
this next should read vLLM's KV-cache-group accounting before treating either as the other's
correction.** Recorded in `@MODELS.md` as an open discrepancy. Enable it with a measurement, not because the
part supports it. `fp8_e5m2` in particular is DEGRADED on the L4 artifact rigs: two mantissa
bits, visibly truncated output.

## The image tag — a source ref in a tag field, and it cost a launch

**`VLLM_IMAGE=vllm/vllm-openai:v0.28.0`** (MEASURED 2026-08-30; vLLM 0.28.0, torch 2.13.0+cu130,
transformers 5.15.1, CUDA 13.0).

**The tag this rig shipped with, `v0.27.2rc0`, IS NOT A PUBLISHED IMAGE TAG.** The first launch
died in cloud-init at:

```
failed to resolve reference "docker.io/vllm/vllm-openai:v0.27.2rc0": not found
```

`v0.27.2rc0` is the sibling's **`VLLM_REF`** — a *git* ref that rig **compiled from source**,
which is exactly why it never needed to exist on Docker Hub. The fork copied a **source ref
into an image-tag field**. Published releases are `v0.27.0`, `v0.27.1`, `v0.28.0`; **there is
no `v0.27.2` of any kind.** The paragraph below already said the sibling "built its real image
from `VLLM_REF=v0.27.2rc0`" — the evidence was sitting in the comment the whole time.

**MEASURED on the sibling: v0.26.0 dies** with `AmbiguousGlobalPerLayerAttributeError` against
current `transformers`, because Gemma 4's `head_dim` is per-layer. The `per_layer_config`
handling that fixes it landed in **v0.27.2rc0**. That is a constraint of the **model**, not the
chip, so it carries across the fork unchanged — but it is a floor on **the fix**, not on that
literal string. `v0.28.0` is above it.

**The guard failed open.** `test_image_is_at_or_above_the_measured_vllm_floor` forbids
`v0.27.1` and `v0.26` and checks the `vllm/vllm-openai:` prefix — it **never asserts the tag
resolves**. A blocklist cannot catch a value that is in no list. Same shape as the Codex gates
naming tools that do not exist: *a check that names only what is wrong passes everything it
has not heard of.*

## Instance sizing

`INSTANCE_TYPE=g6.xlarge`. **Every size is supported** — `_validate_instance_type` only
enforces the size list. **No size has been launched.**

**Two traps relative to the G5g table, both of which invert an inherited verdict:**

- **Host RAM DOUBLED at every matching suffix.** `g6.xlarge` has **16 GiB** where
  `g5g.xlarge` had 8. The sibling *rejects* `g5g.xlarge` outright — first on the theory that
  8 GiB "cannot stage 9.5 GiB of weights", later corrected to a swapfile problem. **Neither
  applies here**, and `g6.xlarge` is a reasonable default where `g5g.xlarge` was not.
- **`g6.16xlarge` is SINGLE-GPU** where `g5g.16xlarge` had two. GPU count is **not monotonic
  in the size**: 12xlarge and 24xlarge have 4, 48xlarge has 8, and 16xlarge has 1. Never
  infer it from the name; a wrong tensor-parallel size fails at engine start.

**G6 is 4 GiB of RAM per vCPU; G5g was 2.** Any inherited `RAM // 2` vCPU shortcut silently
**doubles** the answer — `check_g6_quotas` had exactly that bug at the fork and now reads
`_vcpu_count` from the table.

**The swapfile block is now dead code**, because no G6 size falls below the 16 GiB gate. It is
kept rather than deleted because the threshold is a claim about the **checkpoint** (~10.2 GB
to mmap), not the host, and a larger checkpoint would need it again. That makes it **untested
code**, which this lineage has been burned by: `mkswap -q` is a **busybox** flag util-linux
rejects with `invalid option -- 'q'`, and under `set -e` it killed cloud-init *before anything
logged*. It sat latent for as long as only one unlaunched size rendered the block.
`test_no_mkswap_q_flag` guards it.

## AMI resolution

**The architecture requirement flipped at this fork.** The sibling needs **arm64**; an arm64
image cannot boot a G6 at all.

`DLAMI_SSM_PARAMETER` is the **x86_64 base** GPU DLAMI, and is single-valued and
authoritative. **Base, not PyTorch**: this rig serves from a docker image carrying its own
CUDA and torch, so a PyTorch DLAMI is GBs of unused image. The DLAMI supplies the NVIDIA
driver and docker, nothing else.

**This exact path was VERIFIED ON HARDWARE by `gpu-jax-g6-2b` on 2026-08-28** — driver
**595.91.07**, Ubuntu 26.04, 66-second bootstrap. That is the one piece of this rig's
provisioning that is not a guess.

**`/latest/` in a DLAMI path does not mean latest.** It is the newest build *within one
PyTorch-and-Ubuntu line*, and AWS eventually stops rebuilding a line. The sibling pinned
`pytorch-2.7-ubuntu-22.04`, which froze at a **2026-05-02** image while reading as "track
latest".

**Changing `DLAMI_SSM_PARAMETER` requires changing `DLAMI_NAME` in the same commit.** The
sibling's filter requires `Deep Learning ARM64 AMI` *contiguously*, and the base images are
named `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 26.04)` — so it would match none
of them and the fallback would quietly resolve a **different image**. A revert that reports
success. `test_tpu_env_agrees_with_server_defaults` covers both keys together.

**Never hardcode an AMI id** — resolve it at launch.

## Root volume

**100 GB gp3 at 500 MiB/s and 6000 IOPS.** PORTED FROM `gpu-jax-g6-2b`, where it is MEASURED
rather than assumed: two unrelated load stages there both landed on ~125 MB/s — the signature
of a volume ceiling rather than CPU or network — and raising it took `read_shards` from 73.5 s
to **24.7 s**, a clean 3.0x on the same read.

**This rig adds a multi-GB image pull on top of the 10.2 GB checkpoint**, so the ceiling
should bind at least as hard here. UNMEASURED.

Two things to keep:

- **gp3 requires `throughput <= IOPS * 0.25`, and it is enforced at run-instances time** — so
  violating it fails a **launch**, not just a disk. A test pins the inequality.
- **`get_deployment_config` and `create_g6_instance` must provision the same volume.** On the
  sibling the former PRINTED `VolumeSize=200` while the latter LAUNCHED 100, which is how a
  manual reproduction quietly fails to reproduce. Both now render from `ROOT_VOLUME_*`.

## Engineering rules

- boto3 and the standard AWS credential provider chain — never shell out to the AWS CLI.
- SSM Run Command for remote administration; no inbound SSH rule, no private key.
- Require explicit subnet, security-group, and instance-profile ids. Do not create broad
  network or IAM policy.
- Scope instance discovery to `ManagedBy=gpu-vllm-g6-2b`.
- Hugging Face tokens live in Secrets Manager and are fetched at boot. **Never** in user data —
  instance metadata is readable by anything on the box. `set +x` wraps the fetch because the
  script runs under `set -x` and bash traces assignments *with their values*. Two tests assert
  it, including the ordering.
- Launches default to spot. **Surface capacity errors rather than retrying silently.**
- Never hardcode an endpoint; `get_endpoint` resolves it from the instance.
- **Termination is cheap here**, unlike on the sibling: there is no built image to lose with
  the root volume, only an image pull and the model cache. **Do not import that rig's
  "weigh stop against terminate" reasoning, or its AMI-maintenance reasoning.**
- **Do not health-check by testing for a non-empty response.** On this rig's lineage a broken
  deploy answered `': ok: ok: ok…'` — degenerate repetition that passes an emptiness check.

## Spot capacity: quota is not capacity

MEASURED 2026-08-28 while provisioning the JAX sibling: **`g6.xlarge` spot was exhausted in
all five `us-east-1` AZs** with quota to spare. `g6.2xlarge` in `us-east-1d` succeeded, and
was chosen from `aws ec2 get-spot-placement-scores` (score 3 against 1 elsewhere) rather than
by launching until one worked.

Two things that generalise from the G5g lineage and are worth budgeting for:

- **Reclamation is highly variable.** One instance was taken at **21 minutes**, another ran
  **19.2 hours**, same type and region. **Neither is typical** — quote the range, and
  checkpoint continuously rather than sizing work to an assumed lifetime.
- **Price is not a proxy for availability.** On 2026-08-27 the only AZ with capacity was also
  the most expensive.

The sibling has a standing argument that its ~67-minute build makes spot unusable and an AMI
bake necessary. **That argument does not carry here** — there is no build, so a reclamation
costs an image pull.

## Commands

Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v` (38 tests,
all passing as of 2026-08-28). They are fully offline — no AWS, no network, no GPU — and are
written to pin **what changed at the fork**, because that is where a silent copy-paste lands:
the dtype flip, the x86_64 AMI filter, the vanished build constants, the image-tag floor, the
non-monotonic GPU count, the vCPU derivation, and that the Codex approval gates name tools
that actually exist.

`make lint` runs `ruff check` and `bash -n` on the shell scripts.

`make skill` regenerates the snapshots under `.claude/skills/` and `skills/`. **`SKILL.md` is
a hand-written SOURCE** — `refresh_skill.py` will not recreate it, so `rm -rf .claude/skills`
destroys it permanently. `test_skill_is_complete_in_both_copies` guards both copies and fails
if any generated file is stale.

There is no `make deploy` on purpose: provisioning resolves an x86_64 AMI at launch time, and
a Makefile would have to hardcode one.

## MCP registration lives in four places

`.mcp.json`, `.claude-plugin/plugin.json`, `.codex/config.toml`, and
`.claude/settings.local.json`'s `enabledMcpjsonServers`. All four must name the server
`gpu-vllm-g6-2b`, which prefixes every tool as `mcp__gpu-vllm-g6-2b__…`. **All four agree as
of 2026-08-28**, and a test asserts it.

**Only `.mcp.json` is generated** by `project-setup.sh`; it and `settings.local.json` are
gitignored. The other two are committed.

**`.codex/config.toml` was written fresh at this fork rather than copied, and the reason is a
real failure on the JAX rig's identical fork.** There, the file survived untouched and was
wrong three ways at once: wrong server name, a skill path that did not exist, and — the
dangerous one — **approval gates naming `*_g5g_*` tools against actual `*_g6_*` tools.** The
gates matched nothing, so **every destructive tool was ungated while appearing to be gated.**

**A gate on a tool name that does not exist fails open and says nothing.** That generalises
past this file: a rename silently converts a safety control into a no-op, and nothing tests
it. `test_codex_gates_name_tools_that_exist` now does.

**`project-setup.sh` derives `SKILL_STEM` from the rig directory** and must never carry a
literal — on the JAX rig's fork a hardcoded stem still named the old rig, so the script could
not find the skill and died with `cannot locate the bundled skill`. **The rig was
unregisterable, not merely misregistered.** The Makefile, `refresh_skill.py` and this script
must agree on one derived name.

`AGENTS.md` and `GEMINI.md` cover the same ground for other tools. There is no generator:
**`CLAUDE.md` is authoritative where they disagree.**

## Measurement

**This rig has exactly one measurement, and it is its own:**
`benchmarks/runs/2026-08-30-first-serve-g6/` with
`benchmarks/reports/2026-08-30-gemma4-e2b-g6.json` (schema 1.1, validates). Headline:
**46.09 tok/s single-stream, 360.17 tok/s at concurrency 8**, on `g6.2xlarge` spot.

**Benchmark JSON travelled with the forks in this tree**, and several rigs carry numbers
measured on hardware they are not. The sibling's `2026-08-12-first-serve-g5g` and
`2026-08-14-rust-frontend-g5g` were **not copied**.
`test_benchmarks_carries_no_other_rigs_runs` used to assert `runs/` was *empty*; now that this
rig has served, it asserts every artifact here carries the **`-g6`** slot instead. The guard's
intent never changed — it was always about foreign hardware, not about the count.

Naming is `benchmarks/runs/<date>-<what>-g6/` — `<hw-short>` equals the hardware slot,
and it is the hardware **measured**, not the rig hosting the file. The JAX sibling's
`tune_loop.py` hardcoded `-g5g` through its own fork and filed an L4 result under the T4G's
name before it was caught.

`benchmarks/README.md` and `serving-report.schema.json` are **synced copies** —
`make benchmarks-sync` at the monorepo root overwrites them, so edit the root originals.

**Four numbers you will be tempted to reuse, and must not:**

- **48.3–48.5 tok/s** — `gpu-jax-g6-2b`, MEASURED 2026-08-28. **Same chip, same checkpoint,
  different runtime.** This is the comparison this rig exists to make, and quoting it as
  though it were this rig's would destroy exactly the thing being measured.
- **43.1 / 44.24 tok/s** — the T4G sibling, 2026-08-12/13. Different silicon, and obtained
  **with hand-reduced Triton tiles**, so it is not a stock-vLLM figure either.
  **CORRECTED 2026-08-30 — neither figure is a benchmark, do not compare against either.**
  `43.1` is one sample from the 2026-08-12 first-serve run, whose own report says "single-run,
  single-stream, no repeats and no variance figure", taken with a 19-token prompt. `44.24` has
  **no benchmark artifact anywhere in the tree** — it survives only in `gpu-vllm-g5g-2b/server.py`'s
  swap comment and `tests/test_server.py`, where it was measured 2026-08-13 to show that
  `g5g.xlarge` + a 16 GiB swapfile reaches a healthy endpoint at all. The tile-clamp caveat is real
  but does not distinguish them: it applies to every vLLM-on-T4G number, the good ones included.
  **Compare against `gpu-vllm-g5g-2b/benchmarks/runs/2026-08-14-rust-frontend-g5g/`** — `vllm bench
  serve`, three runs, one `g5g.4xlarge`: c=1 TPOT 31.44 ms (~31.8 tok/s decode), c=4 ~97 tok/s,
  c=8 168.33 tok/s.
- **~44 tok/s on one Inferentia core** from `~/gemma4-tips-aws` — different harness,
  different silicon.
- **Anything from the five `gpu-vllm-l4-*` artifact rigs.** They are the **same GPU and the
  same runtime** as this rig, which makes them the most tempting and the most dangerous:
  their provenance is the weakest in the tree. `~/gemma4-tips` duplicated its own artifacts,
  82 reports reduce to 20 unique, and its directory names misattribute both model and chip.
  **Same chip is not same measurement.** Never read a model or a chip off one of those
  directory names; prefer a report's `Endpoint:` line.

A config flag being accepted is not evidence it did anything. Cross-check against an absolute
physical bound — **300 GB/s of GDDR6 and 23034 MiB is the whole envelope here** — not against
another config.

**MEASURED 2026-08-30, and it reframes the comparison this rig exists for: decode is NOT
bandwidth-bound at B=1.** E2B streams **3.382 GB/token** (matmuls 2.576 + tied LM head 0.805);
the 4.698 GB PLE table is an indexed **gather**, not a stream — see `@MODELS.md`, *"Resident is
not streamed"*. At 46.09 tok/s that is 155.9 GB/s of 300, i.e. **52% MBU against an implied
ceiling of 88.7 tok/s**.

So **vLLM (46.09) and JAX (48.3–48.5) both sit at about half the memory roofline**, and the ~5%
between them is an **overhead / kernel-launch difference, not a memory-system one.** Do not
explain that gap with bandwidth. It also explains why batching pays so well here: c=8 reaches
360.17 tok/s, far past the single-stream ceiling, because streamed weights amortise across the
batch.

**Never divide resident weights by bandwidth on this checkpoint** — 46.09 tok/s × 9.8 GiB
resident implies 485 GB/s on a 300 GB/s bus, which is impossible, and that impossibility is
itself the proof that resident is the wrong denominator.

## What to do first

1. `check_g6_quotas`, then `get-spot-placement-scores` to pick a size and AZ.
2. `create_g6_instance` → `get_install_progress`.
3. **`verify_gpu_arch`.** MEASURED 2026-08-30: **it passes** — a real bf16 matmul ran. Read
   its output carefully though: the arch list is `sm_75/80/86/90/100/120` and **`sm_89` is not
   in it**; Ada runs the `sm_86` cubins by same-major-version binary compatibility. The matmul
   is the evidence, not the list.
4. **The Triton shared-memory error does NOT occur.** SETTLED 2026-08-30 — the ~96 KiB tile
   fits Ada's ~99 KiB unpatched.
5. `verify_model_health`, then a sweep. **The single-stream comparison is done (46.09 vs
   48.3–48.5).** The open work is the other half: **nothing has measured the JAX sibling under
   concurrency**, and vLLM's 7.8x scaling to c=8 is where the runtimes actually diverge.

**Benchmark from ON the instance against localhost**, as the 2026-08-30 run did — it keeps WAN
latency out of TTFT/ITL, and it works whether or not the security group lets you in.
