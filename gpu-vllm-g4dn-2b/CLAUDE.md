# CLAUDE.md — `gpu-vllm-g4dn-2b`

Serving rig: **`google/gemma-4-E2B-it`** under **vLLM** on **AWS EC2 G4dn** — an **x86_64**
(Intel) host with an **NVIDIA T4** GPU (Turing, SM 7.5, 16 GB nominal / **15360 MiB** measured
on the T4G sibling, not here).

This is a full rig: `server.py`, an MCP server, a skill, a plugin manifest, and `tpu.env`.
It is **not** one of the `gpu-vllm-l4-*` artifact rigs, despite sharing the `gpu` platform
slot and the runtime slot with them.

> **THIS RIG HAS SERVED NOTHING.** The directory was a stale copy of `gpu-jax-g4dn-2b`; the
> vLLM side was forked from `gpu-vllm-g6-2b` on 2026-08-29, which had itself served nothing.
> Every number here is arithmetic or inherited from a sibling. `benchmarks/runs/` is
> deliberately empty and no run directory was copied. **Nothing below has been checked on this
> hardware.**

## Why this rig exists: it isolates one of the two G5g problems

`gpu-vllm-g5g-2b` is the hardest rig in this tree because it hits **two independent problems
at once** and has to solve both to serve a single token. This rig keeps one and deletes the
other, which is the only reason it is worth building.

| | Problem 1: SM 7.5 in the published image? | Problem 2: Triton tile vs shared memory |
| --- | --- | --- |
| `gpu-vllm-g5g-2b` — aarch64, SM 7.5 | **NO** → ~67-minute from-source build | **NO** → unlanded tile clamp |
| `gpu-vllm-g6-2b` — x86_64, SM 8.9 | yes | yes (~99 KiB), **unverified** |
| **this rig** — x86_64, SM 7.5 | **YES** | **NO** → tile clamp, mandatory |

**Problem 1 is a property of the published binaries, not the silicon.** `vllm/vllm-openai`
publishes one manifest list with two platforms: `linux/amd64` is compiled for
`7.5 8.0 8.6 8.9 9.0 10.0 12.0` and `linux/arm64` for `8.0 8.7 8.9 9.0 10.0 11.0 12.0`. G4dn
is Intel, so it pulls the amd64 manifest — **the one that carries 7.5**. Same image, same tag,
different answer purely because of the host architecture. The Dockerfile sets no `+PTX`, so
on arm64 there is not even a JIT fallback, which is why the G5g rig must rebuild.

So this rig has **no from-source build, no CUDA toolkit, no Rust toolchain, no prebuilt AMI
to maintain, and no `serving=` mode.**

**Problem 2 is untouched**, because it is a property of the model and the chip. And unlike on
Ada it is not a margin to check: **65,536 < 98,304 is arithmetic**, and the same silicon has
already produced the failure on a sibling.

**It is also the runtime control for `gpu-jax-g4dn-2b`** — same chip, same host, same
checkpoint, different runtime. That rig has measured nothing either, so the comparison is
*available*, not made.

## The Turing patch is the whole rig

**`docs/turing-shared-memory.md` is the write-up. Read it before touching
`patch_triton_turing.py` or `_user_data`.**

Gemma 4's head dims are heterogeneous — sliding **256**, global **512**. Only FA4 and Triton
handle that; FA4 is unavailable, so vLLM **forces `TRITON_ATTN`** and its tile at
`head_size=512` wants **98,304 B** of shared memory per block. Turing allows **65,536** at
most, and only if the kernel opts into the dynamic attribute — the *default static* limit is
**49,152**, which is what `torch.cuda.get_device_properties().shared_memory_per_block`
reports. **Always cite 64 KiB with that qualifier**, or a reader who checks torch concludes
the doc is wrong.

**The backend is not the knob.** MEASURED on the G5g rig 2026-08-12: vLLM v0.27 does not
recognize `VLLM_ATTENTION_BACKEND` at all (`Unknown vLLM environment variable detected`), and
forces Triton for this model regardless. Setting it did nothing. `ATTENTION_BACKEND` is left
empty here for that reason, and an empty value is **not exported** — vLLM seeing the variable
set to `""` is not the same as not seeing it.

### What is genuinely new here: the delivery, not the patch

The patch body is the G5g rig's. What changes is that **that rig can only get it in by
compiling vLLM** — its image has no SM 7.5 kernels, so the engine has to be rebuilt anyway and
the patch rides along on a 67-minute build. Here the kernels are already present and **exactly
one pure-Python file is wrong**, so cloud-init:

```
docker pull <stock>
  → resolve the module path INSIDE the image      (never hardcoded — site-packages
                                                   carries the image's python version)
  → docker run --entrypoint cat → patch → patch_triton_turing.py
  → FROM <stock>; COPY patched <resolved path> → docker build
  → verify the clamp is present IN THE BUILT IMAGE
  → serve the DERIVED tag
```

Seconds, not an hour. **Three of those steps exist because of a specific way this fails
silently**, and none should be removed:

- **The path is resolved, not hardcoded.** It moves with the image's python version.
- **A failed patch kills cloud-init** (`|| exit 1` under `set -e`). The alternative is a
  derived tag that serves unpatched and dies ~10 minutes later at engine start, having
  reported success the whole way.
- **The clamp is verified inside the built image.** A wrong `COPY` destination builds cleanly
  and leaves the module unpatched.

### `patch_triton_turing.py` refuses rather than no-ops, on purpose

A patch that silently matches nothing is worse than one that fails, because it moves the
failure ten minutes downstream and attributes it to the wrong thing. The script exits 2 with
the surrounding source attached when an identifier is missing, when the launch-site anchor is
gone, or when the pipeline-stage variable is ambiguous.

**That last check is subtler than it looks and is worth keeping.** The launch site reads
`num_stages=launch_num_stages`, and a line-oriented pattern reads that as an *assignment* to
`num_stages`. Picking it would write `num_stages = 1` into the enclosing scope, binding a
local nothing reads — **half the fix, applied silently, reported as success.** The script uses
`ast` and considers only real assignment targets. `TRITON_STAGES_VAR` overrides it.

### VERIFIED against real upstream source 2026-08-29 — and the first anchor was wrong

Pulled `vllm/v1/attention/ops/triton_unified_attention.py` at `v0.28.0` (byte-identical to
`main`) and ran the patch against it. Three findings, and the second one nearly shipped:

- **The clamp is still needed.** `_get_tile_size` has **no shared-memory awareness** — it
  returns 32/16/32 from `head_size` and element size alone. Upstream has not fixed this.
- **The launch-site anchor was WRONG, and it would have failed silently.** The obvious reading
  of the G5g rig's patch is "insert before the kernel launch". But upstream copies the tile
  constants into a local *well before* the launch:

  ```python
  if not use_3d:  tile_size = TILE_SIZE_PREFILL      # consumed here
  else:           tile_size = TILE_SIZE_DECODE       # and here
  if launch_num_stages is not None:
      launch_kwargs["num_stages"] = launch_num_stages  # and here
  kernel_unified_attention[grid](...)                  # the launch
  ```

  Clamping at the launch would rewrite three variables **nothing reads afterwards**. The
  marker would be present, the in-image verification would pass, `verify_triton_patch` would
  report ✅, and the kernel would still ask for 98,304 bytes. **Exactly the silent half-fix
  the script is built to refuse — arrived at by the script's own author.**

  So the insertion point is now **derived**: immediately after the last assignment to either
  tile constant, with an explicit refusal if anything *reads* them before that point, and a
  second refusal if nothing reads them after. On real source it lands at line 997, with the
  three reads at 1079, 1082 and 1182 behind it.

- **The kernel was renamed.** `kernel_unified_attention_2d`/`_3d` were merged into one
  `kernel_unified_attention`, so the old pattern matched nothing at all. That refusal is what
  sent the placement question back to first principles.

**What the clamp actually does at this model's shapes** (fp16, `BLOCK_M=16`, budget 60000) —
it is deliberately narrow:

| layer | head | path | tile in → out | bytes in → out |
| --- | ---: | --- | --- | --- |
| `sliding_attention` | 256 | prefill/decode | 32 → 32 | 40,960 → unchanged |
| `full_attention` | 512 | decode | 16 → 16 | 49,152 → unchanged |
| **`full_attention`** | **512** | **prefill** | **32 → 16** | **81,920 → 49,152** |

Only Gemma 4's 512-wide global prefill path is touched, and it lands under the 65,536 hard
limit. A test pins that; if it ever clamps everything, the budget is wrong, not the tiles.

`get_install_progress` still recognises a refusal and says `This is NOT a slow launch`.

`TURING_SMEM_BUDGET=60000`, not 65536: the tile arithmetic does not count the kernel's
accumulators, so budgeting the hard limit still overflows.

## The inherited image tag did not exist

**`vllm/vllm-openai:v0.27.2rc0` is not a real tag.** `gpu-vllm-g6-2b` pins it and this rig
copied it at the fork. CHECKED 2026-08-29 against both registries: **404 on Docker Hub**, and
no such git tag in `vllm-project/vllm` — the published sequence goes `v0.27.0`, `rc1`, `rc2`,
`v0.27.1`, then `v0.28.0rc1`, `rc2`, `v0.28.0`, `v0.28.1rc0`. **Cloud-init would have died at
`docker pull`, on the first stage, before any of this rig's machinery ran.**

**How it survived is the interesting part.** The version-floor test asserted the tag was *not*
`v0.27.1` and *not* `v0.26` — both trivially true of a tag that was never published. **A
version floor that never checks the artifact exists is this tree's "an accepted flag is not
evidence" rule one level up, in the test itself.**

**Now pinned `v0.28.0`** — a real release, comfortably above any `v0.27.x` floor. The floor
claim itself is unaffected and still holds: v0.26.0 dies with
`AmbiguousGlobalPerLayerAttributeError` because Gemma 4's `head_dim` is per-layer, and that is
a constraint of the **model**, so it applies on every chip here.

**Not `nightly`, and that was checked rather than assumed.** `nightly` is a *moving* tag and
this rig renders deterministic user data on purpose. For
`triton_unified_attention.py`, `main` is **byte-identical** to `v0.28.0` and the patch applies
to both — so nightly would buy nothing and cost reproducibility.

**The premise is now VERIFIED on the real v0.28.0 manifest**, not inherited:

```
linux/amd64   TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 8.9 9.0 10.0 12.0    <- SM 7.5 PRESENT
linux/arm64   TORCH_CUDA_ARCH_LIST=8.0 8.7 8.9 9.0 10.0 11.0 12.0   <- SM 7.5 ABSENT
```

That is the fork's whole justification, and it still holds at the current release rather than
only at the `v0.27.1` the G5g rig read on 2026-08-12.

**`gpu-vllm-g6-2b` still carries the phantom tag** and will fail the same way. Not fixed here
— it is a different rig, and this file is not its documentation.

## Turing is not Ada — the fork parent's dtype policy is wrong here

`gpu-vllm-g6-2b` is the fork parent and runs **Ada (SM 8.9), which has bf16 and fp8**. Turing
has **neither**. Copying its dtype settings is the single most likely error in `server.py`.

| | G6 parent (SM 8.9) | **this rig (SM 7.5)** | G5g sibling (SM 7.5) |
| --- | --- | --- | --- |
| compute dtype | `bfloat16` | **`float16`** | `float16` |
| KV cache dtype | `auto`; fp8 reachable, unused | **`auto`; fp8 NOT reachable** | `auto` |
| device memory | 23034 MiB | **15360 MiB** | 15360 MiB |
| per-block shared memory | ~99 KiB | **64 KiB opt-in / 48 KiB static** | 64 / 48 KiB |
| host | x86_64 | **x86_64** | aarch64 |

**bfloat16 does not fail here — it upconverts, and that is why the wrong value is dangerous.**
MEASURED on the G5g rig: PyTorch runs bf16 on Turing by upconverting, and vLLM logs
`Casting torch.bfloat16 to torch.float16` and proceeds. `float16` is correct because it is
what **executes**, not because bf16 errors. State the reason correctly or someone tests torch,
watches it pass, and deletes the guard.

**One consequence that is NOT the G6 rig's, and must not be described using its numbers.**
The checkpoint ships bf16 and compute here is float16, so vLLM converts every weight at load.
`gpu-jax-g4dn-2b`'s lineage measured a dtype mismatch costing **54% of decode** on this exact
silicon — but that was a **JAX loader holding bf16 and converting at every USE, per step**.
vLLM converts **once, at load**. **Do not quote the 54% in this rig.** The chip transfers;
the mechanism does not.

**fp8 is not "available but unused" here** the way it is on the G6 rig — there is no datapath
at all. int8 is Turing's only low-precision compute win (`@HARDWARE.md`), and vLLM does not
expose it as a KV cache dtype.

## Instance sizing

`INSTANCE_TYPE=g4dn.xlarge`. **Every size is supported** — `_validate_instance_type` only
enforces the size list. **No size has been launched.**

| size | GPUs | GPU mem | vCPU | RAM |
| --- | ---: | ---: | ---: | ---: |
| `g4dn.xlarge` | 1 | 16 GB | 4 | 16 GB ← default |
| `g4dn.2xlarge` | 1 | 16 GB | 8 | 32 GB |
| `g4dn.4xlarge` | 1 | 16 GB | 16 | 64 GB |
| `g4dn.8xlarge` | 1 | 16 GB | 32 | 128 GB |
| `g4dn.12xlarge` | **4** | 64 GB | 48 | 192 GB |
| `g4dn.16xlarge` | **1** | 16 GB | 64 | 256 GB |
| `g4dn.metal` | **8** | 128 GB | 96 | 384 GB |

Three traps, two shared with G6 and one that is this rig's own:

- **GPU count is NOT monotonic in the size.** 12xlarge has 4, 16xlarge has 1, metal has 8.
  Never infer it from the suffix; a wrong tensor-parallel size fails at engine start.
- **G4dn is 4 GiB of RAM per vCPU; G5g was 2.** Any inherited `RAM // 2` vCPU shortcut
  silently **doubles** the answer. `_vcpu_count` reads the table.
- **Host RAM is DOUBLE its g5g namesake at every suffix.** `g4dn.xlarge` has 16 GiB where
  `g5g.xlarge` had 8, so that rig's xlarge rejection does not carry.

### The swap gate deliberately disagrees with the JAX rig on the same instance type

`gpu-jax-g4dn-2b` gates swap **at-or-below** 16 GiB. This rig gates **strictly below**, so
`g4dn.xlarge` gets no swapfile. That is not an oversight:

- The JAX rig OOMs at exactly 16 GiB in `quantize_ple_table`, which upcasts a 4.70 GB PLE
  table to float32 while the whole tree is resident. **A property of its loader.** vLLM has no
  equivalent step.
- The vLLM failure is different: the kernel refuses to **mmap** the 10.2 GB checkpoint against
  7.5 GiB of RAM and no swap (`Cannot allocate memory (12)`), MEASURED on `g5g.xlarge`
  2026-08-13. The G5g rig measured a 16 GiB host needing no swapfile.

**Do not harmonise the two gates.** They encode different failures, and copying the JAX one
here provisions swap nothing needs. A test pins the distinction.

**Consequence: no G4dn size trips the gate, so the swap block never renders and is UNTESTED
CODE.** Kept because the threshold is a claim about the *checkpoint* (~10.2 GB to map), not
the host. This lineage has been burned by exactly that: `mkswap -q` is a **busybox** flag
util-linux rejects with `invalid option -- 'q'`, and under `set -e` it killed cloud-init
*before anything logged*, latent for as long as only one unlaunched size rendered the block.
`test_no_mkswap_q_flag` guards it.

## AMI resolution

**The architecture axis is what separates this rig from `gpu-vllm-g5g-2b`** — an arm64 DLAMI
cannot boot a G4dn, and an x86_64 one cannot boot a G5g.

`DLAMI_SSM_PARAMETER` is the **x86_64 base** GPU DLAMI, single-valued and authoritative.
**Base, not PyTorch**: this rig serves from a docker image carrying its own CUDA and torch, so
a PyTorch DLAMI is GBs of unused image. The DLAMI supplies the NVIDIA driver and docker.

**This exact path was VERIFIED ON HARDWARE by `gpu-jax-g6-2b` 2026-08-28** — driver
595.91.07, Ubuntu 26.04, 66-second bootstrap. **On a G6, not on a G4dn.**

**`/latest/` in a DLAMI path does not mean latest.** It is the newest build *within one
PyTorch-and-Ubuntu line*, and AWS eventually stops rebuilding a line. The G5g rig pinned
`pytorch-2.7-ubuntu-22.04`, which froze at a **2026-05-02** image while reading as "track
latest".

**Changing `DLAMI_SSM_PARAMETER` requires changing `DLAMI_NAME` in the same commit.** The G5g
filter requires `Deep Learning ARM64 AMI` *contiguously*, and the base images are named
`Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 26.04)` — so it would match none of them
and the fallback would quietly resolve a **different image**. A revert that reports success.
`test_tpu_env_agrees_with_server_defaults` covers both keys together.

**Never hardcode an AMI id** — resolve it at launch. Note the legacy tips-tree rigs hardcode
`ami-012ba162b9cd2729c`, which *is* x86_64 and so would boot here. That makes it a worse trap
here than on the G5g rig, where it fails immediately.

## Root volume

**100 GB gp3 at 500 MiB/s and 6000 IOPS.** PORTED FROM `gpu-jax-g5g-2b`, where it is MEASURED
rather than assumed: two unrelated load stages both landed on ~125 MB/s — the signature of a
volume ceiling rather than CPU or network — and raising it took `read_shards` from 73.5 s to
**24.7 s**, a clean 3.0x on the same read. **UNMEASURED HERE**, and this rig puts *three*
multi-GB reads on that volume: the image pull, the derived build, and the checkpoint.

- **gp3 requires `throughput <= IOPS * 0.25`, enforced at run-instances time** — violating it
  fails a **launch**, not just a disk. A test pins the inequality.
- **`get_deployment_config` and `create_g4dn_instance` must provision the same volume.** On
  the G5g rig the former PRINTED `VolumeSize=200` while the latter LAUNCHED 100, which is how
  a manual reproduction quietly fails to reproduce. Both render from `ROOT_VOLUME_*`.

## Engineering rules

- boto3 and the standard AWS credential provider chain — never shell out to the AWS CLI.
- SSM Run Command for remote administration; no inbound SSH rule, no private key.
- Require explicit subnet, security-group, and instance-profile ids. Do not create broad
  network or IAM policy.
- Scope instance discovery to `ManagedBy=gpu-vllm-g4dn-2b`.
- Hugging Face tokens live in Secrets Manager and are fetched at boot. **Never** in user data
  — instance metadata is readable by anything on the box. `set +x` wraps the fetch because the
  script runs under `set -x` and bash traces assignments *with their values*. Two tests assert
  it, including the ordering.
- Launches default to spot. **Surface capacity errors rather than retrying silently.**
- Never hardcode an endpoint; `get_endpoint` resolves it from the instance.
- **Termination is cheap here.** Nothing is compiled — a relaunch costs an image pull, a
  seconds-long derived build and the model download. **Do not import the G5g rig's "weigh stop
  against terminate" reasoning, or its AMI-maintenance reasoning.** Both exist because that rig
  loses a 67-minute build with the root volume.
- **Do not health-check by testing for a non-empty response.** MEASURED on this lineage
  2026-08-12: a broken deploy answered `': ok: ok: ok…'` — degenerate repetition, 16 tokens,
  non-empty, and completely wrong. `verify_model_health` carries a crude degeneracy check for
  exactly that, and **it is not a quality metric** — it catches one specific documented
  failure shape.

## Spot capacity: quota is not capacity

Two things measured elsewhere in this family, both worth budgeting for:

- **Quota is not capacity.** G-family spot in `us-east-1` has been exhausted in every AZ but
  one with quota to spare, and **the one AZ with capacity was the most expensive**. Price is
  not a proxy for availability. Use `aws ec2 get-spot-placement-scores`.
- **Reclamation is highly variable.** One instance was taken at **21 minutes**, another ran
  **19.2 hours**, same type and region. **Neither is typical** — quote the range.

The G5g rig has a standing argument that its ~67-minute build makes spot unusable and an AMI
bake necessary. **That argument does not carry here** — there is no build, so a reclamation
costs an image pull and a seconds-long derived build.

## Commands

Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v`. They are
fully offline — no AWS, no network, no GPU, **no docker** — and are written to pin what
changed at the fork, because that is where a silent copy-paste lands: the dtype flip back to
float16, the g4dn size table and its non-monotonic GPU count, the swap gate that deliberately
disagrees with the JAX rig, and every silent failure mode of the patch (marker drift, payload
round-trip, the 16 KB user-data cap, the `num_stages` keyword-argument trap).

`make lint` runs `ruff check server.py patch_triton_turing.py refresh_skill.py tests` then
`bash -n` on **four** shell scripts. **A new top-level module is silently unlinted until it is
added to that list** — on `gpu-jax-g5g-2b` `profile_decode.py` sat outside it and was red for
a day.

`make skill` regenerates the snapshots under `.claude/skills/` and `skills/`. **Four files are
generated**: `server.py`, **`patch_triton_turing.py`**, `project-setup.sh`, `requirements.txt`.
The patch script is in that list because `create_g4dn_instance` reads it from beside
`server.py` and ships it in user data — **an installed skill copy without it cannot launch an
instance at all.**

**`SKILL.md` is a hand-written SOURCE** — `refresh_skill.py` will not recreate it, so
`rm -rf .claude/skills` destroys it permanently, which is what happened on a sibling during a
rename. `test_skill_is_complete_in_both_copies` guards both copies and fails if any generated
file is stale.

There is no `make deploy` on purpose: provisioning resolves an x86_64 AMI at launch time, and
a Makefile would have to hardcode one.

## MCP registration lives in four places

`.mcp.json`, `.claude-plugin/plugin.json`, `.codex/config.toml`, and
`.claude/settings.local.json`'s `enabledMcpjsonServers`. All four must name the server
`gpu-vllm-g4dn-2b`, which prefixes every tool as `mcp__gpu-vllm-g4dn-2b__…`. **All four agree
as of 2026-08-29**, and a test asserts it — including that no live (non-comment) line still
names `g5g`, `gpu-jax` or `gpu-vllm-g6`.

**Only `.mcp.json` is generated** by `project-setup.sh`; it and `settings.local.json` are
gitignored. The other two are committed.

**`.codex/config.toml` was written fresh rather than edited, and the reason is a real failure
on the JAX rig's identical fork.** There the file survived untouched and was wrong three ways
at once: wrong server name, a skill path that did not exist, and — the dangerous one —
approval gates naming `*_g5g_*` tools against actual `*_g4dn_*` tools. The gates matched
nothing, so **every destructive tool was ungated while appearing to be gated.**

**A gate on a tool name that does not exist fails open and says nothing.** That generalises
past this file: a rename silently converts a safety control into a no-op.
`test_codex_gates_name_tools_that_exist` now checks it.

**`project-setup.sh` derives `SKILL_STEM` from the rig directory** and must never carry a
literal — on the JAX rig's fork a hardcoded stem still named the old rig, so the script could
not find the skill and died with `cannot locate the bundled skill`. **The rig was
unregisterable, not merely misregistered.**

`AGENTS.md` and `GEMINI.md` cover the same ground for other tools. There is no generator:
**`CLAUDE.md` is authoritative where they disagree.**

## Measurement

**This rig has no measurements.** `benchmarks/runs/` is empty and a test keeps it that way
until something is measured here. Naming will be `benchmarks/runs/<date>-<what>-g4dn/` —
`<hw-short>` equals the hardware slot, and it is the hardware **measured**, not the rig
hosting the file.

`benchmarks/README.md` and `serving-report.schema.json` are **synced copies** —
`make benchmarks-sync` at the monorepo root overwrites them, so edit the root originals.

**Five numbers you will be tempted to reuse, and must not:**

- **43.1 / 44.24 tok/s** — `gpu-vllm-g5g-2b`, 2026-08-12/13. **The closest number in the tree
  and still not this rig's**: same GPU generation and the same runtime, but a different host
  architecture, a from-source build, and it was obtained **with hand-reduced Triton tiles**.
  This rig also reduces tiles, which makes the comparison *closer* than usual and therefore
  more tempting to over-claim. It is the number to beat, not a baseline to inherit.
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
- **48.3–48.5 tok/s** — `gpu-jax-g6-2b`, MEASURED 2026-08-28. Different chip (L4/Ada) and a
  different runtime.
- **12.4–13.1 tok/s** — the `gpu-jax-g5g-*` runs. Same chip generation, **pure JAX**, and a
  single-stream engine capped at `MAX_NUM_SEQS=1`. Not comparable to a batched vLLM figure.
- **~44 tok/s on one Inferentia core** from `~/gemma4-tips-aws` — different harness, different
  silicon.
- **Anything from the five `gpu-vllm-l4-*` artifact rigs.** Same runtime, different chip, and
  the weakest provenance in the tree: `~/gemma4-tips` duplicated its own artifacts, 82 reports
  reduce to 20 unique, and its directory names misattribute both model and chip. Never read a
  model or a chip off one of those directory names; prefer a report's `Endpoint:` line.

A config flag being accepted is not evidence it did anything. Cross-check against an absolute
physical bound — **320 GB/s theoretical / 277 GB/s measured streaming read on the T4G, and
15360 MiB, is the whole envelope here** (`@HARDWARE.md`). Quote **277**, not 320: decode is
bandwidth-bound, so the achieved figure is the bound that matters.

## A root-file correction this rig depends on

**`@HARDWARE.md` says "the Turing-capable vLLM attention backend is `XFORMERS`".** That line
predates the 2026-08-12 measurement on `gpu-vllm-g5g-2b` showing vLLM **forces `TRITON_ATTN`**
for Gemma 4 and **does not recognize `VLLM_ATTENTION_BACKEND` at all** in v0.27. It should be
corrected at the root rather than restated in a rig — filed here because this rig's entire
design follows from the corrected version, not the stale one.

## What to do first

1. `check_g4dn_quotas`, then `aws ec2 get-spot-placement-scores` to pick a size and AZ.
2. `create_g4dn_instance` → `get_install_progress`.
3. **`verify_gpu_arch`.** Expected to **PASS** here, the opposite of the G5g rig where the same
   tool confirms an absence. That inversion is half the fork's premise and is unchecked. It
   probes with **float16**, not bfloat16 — a bf16 probe would pass by upconversion and tell you
   nothing.
4. **`verify_triton_patch`.** This is the check that matters, and passing step 3 says nothing
   about it — the two problems are independent.
5. `verify_model_health`, then a sweep — and only then is the JAX comparison real.
