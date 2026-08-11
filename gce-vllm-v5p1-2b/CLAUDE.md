# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file MCP server (`server.py`, FastMCP) that acts as a devops agent for serving Gemma 4
(`google/gemma-4-E2B-it`) with vLLM on a single-chip Google Cloud **TPU v5p** flex-start Compute Engine
instance (`ct5p-hightpu-1t-tpu`). Its tools shell
out to `gcloud` and talk HTTP to the vLLM OpenAI-compatible endpoint on port 8000.

**This rig was forked from `tpu-vllm-v5e1-2b` on 2026-08-10 and repointed at v5p.** The code is that rig's
code; what changed is the hardware, the zone set, the quota ids and the tensor-parallel size. Anything below
labelled as *measured* was measured on v5e-1 unless it says otherwise — **nothing here has been run on v5p
yet**, and the v5e-1 memory numbers in particular do not transfer (16 GiB/chip against 95).

The live-demo rig is still `tpu-vllm-v5e1-2b`, not this one.

## Commands

```
make install    # pip install -r requirements.txt
make run        # python server.py (stdio MCP server)
make test       # python test_agent.py — unittest, NOT pytest
make lint       # ruff check . && ruff format --check . && mypy .
make format     # apply ruff formatting and autofixes
make tools      # regenerate GemmaTools.md from the @mcp.tool() decorators
make benchmark  # discovers the TPU IP, then runs benchmarking_suite.py against it
make query PROMPT="..."
```

`make lint` only *checks* formatting — `make format` is what writes it. `make test` passes clean (36 tests);
keep it that way. **`make lint` does not** — it fails on two pre-existing `B` findings in `to-medium.py`
(`E731` lambda, `B905` bare `zip`), inherited unchanged from `tpu-vllm-v5e1-2b`, where it fails identically.
`server.py` and `test_agent.py` are clean, so `ruff check server.py test_agent.py` is the useful signal until
someone fixes that file.
A `PostToolUse` hook in `.claude/settings.json` already runs `ruff format` on every `.py` file Claude edits.

## Style

- ruff is both linter and formatter; no black. `line-length = 120`, but `E501` is in the ignore list, so the
  formatter enforces width and the linter does not.
- Lint rules are `E, F, B, I` — import sorting comes from ruff's `I`, not a separate isort.
- mypy is deliberately non-strict: `check_untyped_defs = true` but `attr-defined` is globally disabled.
- Python 3.13 is the minimum; ruff targets `py313` and mypy runs at `python_version = "3.13"`.
- Existing code uses `Optional[str]` from `typing` rather than `X | None`. The target no longer requires this
  — it's now just consistency with the surrounding code, so match what's already in the file you're editing.
- Every subprocess call goes through `run_command(cmd: list[str])` — list args via
  `asyncio.create_subprocess_exec`, never `shell=True`. Keep it that way.
- MCP tools are `async def` and return markdown strings with emoji status prefixes (`✅`, `❌`, `📡`).

## Tool catalog is generated — don't hand-edit it

`GemmaTools.md` and the `get_help` tool both build their tool list from `mcp.list_tools()`, so they cannot
drift from the `@mcp.tool()` decorators. After adding or removing a tool, run `make tools` to refresh the
doc. `README.md` intentionally lists only a handful of highlights and points at `GemmaTools.md` for the rest.

Source of truth either way: `grep -n "^@mcp.tool" server.py`.

## Upstream TPU tuning docs — read, but check the version first

**<https://docs.vllm.ai/en/v0.11.1/configuration/tpu/>** is the upstream TPU configuration and tuning
page: `max-num-seqs` as "concurrent decode slots", `max-num-batch-tokens` down for latency / up for
throughput, XLA warmup and the compiled-graph cache, `VLLM_TPU_BUCKET_PADDING_GAP` (use increments of
128), the TP rule, and a pointer to the vLLM auto-tuner. The directional guidance is sound and matches
what this rig measures.

**It is pinned at v0.11.1 and this rig runs `0.26.1rc1.dev125+ga7a204cc6`. Verify anything from it
against the build before acting.** Checked 2026-08-09:

| From the page | Status on this build |
| :--- | :--- |
| `VLLM_TPU_MOST_MODEL_LEN` — "pass 32k to `--max-model-len` and set this to 2048" | **Gone.** Absent from vLLM's `envs.py` at the pinned commit *and* from tpu_inference's 81-var `envs.py`. Setting it is a silent no-op |
| `VLLM_TPU_BUCKET_PADDING_GAP` | Real, but lives in **tpu_inference**'s envs, not vLLM's |
| `VLLM_XLA_CACHE_PATH` | Present in vLLM; this stack is the **JAX** path, whose cache is `/root/.cache/vllm` in the container (~169 MB) |
| "v5e/v6e support INT8 W8A8, INT8 W8A16, FP8 KV cache" | **Contradicted by measurement** — qwix int8 reaches no allocation on either path, and `--kv-cache-dtype fp8_e4m3` measures a 1.000x capacity ratio. See `@QUANTIZATION.md`; the boot allocation log wins over the doc. Measured on v5e-1; **not re-run on v5p**, so treat it as the prior, not a v5p result |

The version string is the handle for checking: `0.26.1rc1.dev125+g**a7a204cc6**` embeds vLLM's git SHA
(commit `a7a204cc6ec99b3…`, 2026-07-30), so upstream source at that exact commit is readable and
authoritative. The **image** ID in this file (`sha256:2a4a1f82…`) is a config-blob ID, *not* a manifest
digest — `docker pull` by it fails with `unexpected media type`. tpu_inference carries no such handle,
so its source can only be read from `main` and is the weaker half of any source claim.

Serving-parameter analysis, measurements and probe plan live in `SERVING-PARAMS.md` — **all of it measured
on v5e-1**, inherited with the fork. Re-measure before quoting any of it as a v5p number.

## Upstream v5p references

Checked 2026-08-10. These describe the chip and the VM, not this rig, and the usual rule applies: a doc
claim loses to a boot log.

| Doc | What it settles | Watch out for |
| :--- | :--- | :--- |
| [TPU v5p](https://docs.cloud.google.com/tpu/docs/v5p) | Topologies, 95 GiB HBM/chip, 2765 GBps, 2 TensorCores/chip, 4 chips per `ct5p-hightpu-4t` host, 2x2x1 as the floor | It lists per-chip peak as bf16 459 TFLOPs **and FP8 459 TFLOPs** — equal, so it buys no speed even if real. `@HARDWARE.md` treats native fp8 as a v7-and-later property; do not read this row as an fp8 win |
| [TPU regions and zones](https://docs.cloud.google.com/tpu/docs/regions-zones) | v5p is `us-central1-a`, `us-east5-a`, `europe-west4-b` — nothing else | **Understates the hardware.** The live `accelerator-types list` publishes `v5p-8` in ≥9 zones; the page's three happen to match the *reachable* set (quota ∩ hardware) for this project. Same understatement as v6e, where the page names 8 against the API's 18 — read the API |
| [Request TPU Flex-start VMs](https://docs.cloud.google.com/tpu/docs/request-using-flex-start) | v5p flex-start is `us-east5-a` only; up to a 7-day run; flex-start goes through the queued-resources API | The only source for the one-zone flex-start claim — unconfirmed by creation here |
| [TPU OS images](https://docs.cloud.google.com/tpu/docs/tpu-os-images) | v5e/v5p/v6e share one image family, `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` (Ubuntu 22.04, kernel 6.8) in `ubuntu-os-accelerator-images` | It does **not** give `--runtime-version` strings. Ours (`v2-alpha-tpuv5`) came from `gcloud compute tpus versions list`, which is authoritative |
| [TPU Monitoring Library](https://docs.cloud.google.com/tpu/docs/tpu-monitoring-library) | `from libtpu.sdk import tpumonitoring` — `duty_cycle_pct`, `tensorcore_util`, `hbm_capacity_total`, `hbm_capacity_usage`, `hlo_exec_timing`, `collective_e2e_latency`. Supports v5p | Runs **on the TPU VM, inside the container**, so it needs working SSH (see below) — unlike `/metrics`, which is plain HTTP. At `TENSOR_PARALLEL_SIZE=1` there are no cross-chip collectives, so `collective_e2e_latency` is inert here; `duty_cycle_pct` and `hbm_capacity_usage` are the useful two |
| [Profile TPU VMs](https://docs.cloud.google.com/tpu/docs/profile-tpu-vm) | XProf (`pip install xprof tensorboard_plugin_profile`), on-demand capture via `profiler.start_server`, or `xprofiler create -z $ZONE -l $GCS_PATH` for a hosted viewer | Says nothing about profiling a vLLM serving process specifically. Same SSH constraint |

## This rig runs on Compute Engine, because the Cloud TPU API is deprecated

**Checked 2026-08-10.** Google's own words: *"The Cloud TPU API is no longer under active development. This
includes the Google Cloud CLI for the Cloud TPU API and the Cloud Client Libraries for the Cloud TPU API."*
It gets bug and security fixes only. *"New hardware generations, starting with TPU7x (Ironwood), are
supported only through Compute Engine or GKE."*
([Cloud TPU resources in Compute Engine](https://docs.cloud.google.com/tpu/docs/tpus-in-compute-engine))

**Migrated 2026-08-10, before this rig ever provisioned anything.** `server.py` now speaks
`gcloud compute instances` throughout; no `gcloud compute tpus` call remains in it. Sibling rigs still use the
old API and still work — v5p is fully served by it — but they have no path to TPU7x and no way to ask for
fewer than 4 v5p chips. The concepts map like this:

| Cloud TPU API (what the sibling rigs use) | Compute Engine (what this rig uses) |
| :--- | :--- |
| TPU VM | VM instance |
| Single-host slice | VM instance, or a MIG for autoscaled inference |
| Multi-host slice | MIG with accelerator topology in a workload policy |
| **Queued Resource** | **Flex-start VM** |
| `--accelerator-type=v5p-8` + `--runtime-version` | `--machine-type=ct5p-hightpu-1t-tpu` + `--image-family` |
| lowercase `--provisioning-model=flex-start` | **UPPERCASE** `--provisioning-model=FLEX_START` |
| quota on `tpu.googleapis.com` (`TPUV5P…ForTPUAPI`) | quota on `compute.googleapis.com` (`TPU-V5P-per-project-zone`) |
| implicit large boot disk | `--boot-disk-size`, which you must set yourself |
| cloud-platform scope by default | `--scopes`, which you must set yourself or Secret Manager fails at boot |

Compute Engine supports **v5p, v6e and TPU7x only** — so this rig can migrate and `tpu-vllm-v5e1-2b` cannot.
Confirmed from the machine-type list rather than the doc: the GCE-native shapes carry a `-tpu` suffix and
`guestAcceleratorType: tpu-v5p`, and they exist for `ct5p-*` and `ct6e-*` but **not** for v5e's `ct5lp-*`,
which only has the legacy plain shapes.

### The image

Family **`ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e`** in project **`ubuntu-os-accelerator-images`** —
Ubuntu 22.04, kernel 6.8, shared across v5e/v5p/v6e. Current image as of 2026-08-10 is
**`ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e-v20260803`** (built 2026-08-03, `READY`); every earlier build in
the family is `DEPRECATED`, and they ship roughly weekly. TPU7x is a separate 24.04 family
(`ubuntu-accel-2404-amd64-tpu-tpu7x`). Pin the **family**, not the dated name, or you inherit a deprecated
image within the fortnight.

There is also an older family spelling still present, `ubuntu-accelerator-2204-amd64-with-tpu-v5e-v5p-v6e`.
Use the `ubuntu-accel-…` form; that is what the current images are published under.

### Why this rig is one chip

The Cloud TPU API's floor is `v5p-8` — 4 chips, no smaller slice exists. **Compute Engine publishes
single-chip and two-chip v5p machine types**, live in every v5p zone on 2026-08-10:

| Machine type | vCPU | RAM | Chips | `guestAcceleratorType` |
| :--- | ---: | ---: | ---: | :--- |
| `ct5p-hightpu-1t-tpu` | 52 | 112 GB | **1** | `tpu-v5p` |
| `ct5p-hightpu-2t-tpu` | 104 | 224 GB | 2 | `tpu-v5p` |
| `ct5p-hightpu-4t-tpu` | 208 | 448 GB | 4 | `tpu-v5p` |
| `ct5p-hightpu-4t` | 208 | 448 GB | 4 | `ct5p` (legacy shape) |

A single v5p chip is **95 GiB of HBM** — about 6x a v5e-1 — so E2B fits on one with enormous headroom at
`TENSOR_PARALLEL_SIZE=1`. That is why this rig takes `ct5p-hightpu-1t-tpu`. It removes the TP question
entirely — E2B is MQA with `num_key_value_heads=1`, so the API path's forced TP=4 would have needed that one
KV head replicated four ways, a stack capability nobody here had verified — and it cuts the chip bill 4x
against a control plane that makes you take four whether the model uses them or not.

**The `-tpu` suffix is load-bearing.** v5p publishes suffixed shapes at 1, 2 and 4 chips but a bare shape
only at 4, so a 1-chip request must carry it.

**Not verified by creation.** A machine type appearing in `machine-types list` is not proof of capacity, nor
that TPU-on-GCE is enabled for this project — the same "quota is not availability" rule that governs the zone
table applies here. It is listed, not provisioned. **The first `create_tpu_instance` call is the real test**,
and the two things most likely to break are exactly what the old path hid: the boot disk and `--scopes`.

## Gotchas

**The JAX compile cache is container-local and thrown away on every `docker rm`.** It is
`/root/.cache/vllm` inside the container (~169 MB), nothing in `startup_script_template.sh` mounts it,
and compilation is **738 s of the ~1,000 s** to healthy. So every restart, config change and spot
preemption repays the full compile. Mounting `-v ~/.cache/vllm:/root/.cache/vllm` fixes restarts;
durable storage would fix preemptions too. On spot capacity this is the largest cold-start lever there is.

**`startup_script_template.sh` is consumed by `str.format()`.** Placeholders are `{project_id}`, `{zone}`,
`{model_name}`, `{hf_secret_id}`, `{tensor_parallel_size}`, `{max_model_len}`, `{max_num_batched_tokens}`,
`{limit_mm_per_prompt}`. Any other literal `{` or `}` added to that bash file — a shell brace expansion, a
`${VAR}`, a JSON literal — raises at format time and breaks the deploy. Escape as `{{` / `}}`.

**The startup script fetches the HF token itself; never add a `{hf_token}` placeholder back.** The rendered
script is uploaded as instance metadata, so a baked-in token would be readable from the instance. It reads
`hf-token` from Secret Manager at boot via the metadata server, retrying for 30 minutes so an IAM grant
applied after creation still lands. The VM's service account needs
`roles/secretmanager.secretAccessor` on the secret. Tracing (`set -x`) is off across the whole token section —
keep it that way, and never interpolate `$HF_TOKEN` into a logged string.

**Serving flags live in one place.** `_vllm_serve_flags()` builds the vLLM arg list from `MAX_MODEL_LEN`,
`MAX_NUM_BATCHED_TOKENS`, `LIMIT_MM_PER_PROMPT`, and `TENSOR_PARALLEL_SIZE`; the startup script takes the same
values as placeholders. Both deploy paths and the generated one-liner therefore agree. Don't reintroduce a
second hardcoded flag list. Note the JSON value needs different quoting inside a single-quoted argument —
that's what the `mm_limit` parameter is for.

**`--gpu-memory-utilization` is unset. The 0.92-is-the-ceiling result is a v5e-1 result and does not
transfer to v5p.** On v5e-1 the engine default of 0.92 gave a 14.49 GiB cap out of 15.75 GiB total —
8.97 GiB weights + 5.52 GiB KV = 321,376 resident tokens — and 0.95 was measured on 2026-08-07 and did not
boot: the KV pool sized exactly as arithmetic predicted, then XLA died ~13 min later loading
`jit_structured_decode_fn`, wanting 384.11 MB against 346.77 MB free.

**The transferable lesson is the mechanism, not the number: the knob governs weights + KV only, and compiled
program images come out of whatever the cap leaves behind, invisible to it.** At least ~836 MB of them on
v5e-1, against the ~1,290 MB that 0.92 left and the ~799 MB that 0.95 left. That headroom is what a tighter
cap eats, on any chip.

**Why it does not carry to v5p:** the failure was a 1.26 GiB absolute headroom running out on a 16 GiB chip.
v5p has **95 GiB per chip**, so the same 0.92 fraction leaves tens of GiB behind and the constraint that
produced this result is simply absent. Do not port the 0.92 ceiling here as if it were a v5p limit — and
equally, do not assume 0.95+ is safe without measuring, because the program-image term is still invisible to
the knob. The probe is expensive either way: the v5e-1 failure landed ~17 min in, *after* the full compile,
not at allocation. Full v5e-1 write-up lives in the source rig at
`../tpu-vllm-v5e1-2b/benchmarks/runs/2026-08-07-gpu-mem-util-v5e1/`.

That run also confirmed E2B's KV cost by difference — 27,520 extra tokens for 0.47 extra GiB is
**18 KiB/token**, cancelling the weights term. The retracted 15 KiB/token figure came from a boot log
line describing one layer group as if it described the whole hybrid model; see `@MODELS.md`.

**SSH: the old finding was about a command this rig no longer calls.** The migration moved every SSH tool
from `gcloud compute tpus tpu-vm ssh` to plain `gcloud compute ssh`, which is a different code path, so the
failure below does not automatically carry over. Probed on 2026-08-10 against a *nonexistent* target, both
commands returned a clean `NOT_FOUND` rather than crashing — which means the crash, if it survives at all,
happens after resource lookup. **Unresolved until there is a real instance to SSH into.** The historical
finding, kept because it explains the diagnostics design:

> **`gcloud compute tpus tpu-vm ssh` did not work from the dev sandbox.** It crashed with
> `ConnectionResetError` on its own internal API call, while plain gcloud API calls were unaffected. That
> broke every MCP tool shelling out through it — `manage_vllm_docker`, `get_vllm_docker_logs`,
> `get_tpu_system_logs`, `run_vllm_benchmark` — with an error describing nothing about the real cause.

Direct `ssh -i ~/.ssh/google_compute_engine xbill@<external-ip>` works regardless, and the important point
stands either way: **most diagnostics need no SSH at all.** `/v1/*` and `/metrics` are plain HTTP, and
`kv_cache_size_tokens`, `num_gpu_blocks`, `gpu_memory_utilization` and `cache_dtype` all come off
`vllm:cache_config_info` in `/metrics`. Prefer those over the SSH tools whichever way the question lands.

**`create_tpu_instance` is non-destructive; `manage_tpu_instances` is not.** The latter deletes every
TPU instance in the zone that isn't the named primary — and sibling rigs share this zone.
`create_tpu_instance` touches only the name it was given, so `find_tpu`'s zone sweep is safe. Keep that
split. The blast radius grew with the migration: it now deletes VMs directly rather than Queued Resources.

**`tpu_zones_status.md` is mutable state, not documentation.** `find_tpu` rewrites it in place to record which
zones have failed, and reads it back to skip known-bad zones. Do not hand-edit it as if it were docs.

**Endpoint discovery is dynamic, and provisioning-path agnostic.** `_discover_vllm_node()` lists **TPU VM
instances** in `ZONE`, filtered to TPU machine types by `TPU_MACHINE_PREFIXES` — that filter is new and
necessary, because an unfiltered instance list returns every VM the project runs in the zone, where the old
`tpu-vm list` was implicitly TPU-only. It ranks candidates (this rig's names first, then `RUNNING`), probes
each on `/v1/models`, and returns the first that answers, as
`VllmNode(name, url, serving)`; `discover_vllm_url()` is the thin wrapper. Never hardcode an endpoint — the
IP changes every time the node is recreated. Use the `get_vllm_endpoint` tool.

Two rules fall out of sibling rigs sharing this zone: a node of **ours** that is up but not yet answering is
returned with `serving=False` so callers can poll it while vLLM boots, and a node that is **not** ours is
never returned unless a probe confirmed it is serving. That second rule is stricter than the old code, which
would hand back the first `ACTIVE` QR in the zone even if it belonged to the jax or pytorch rig.

Until 2026-08-06 discovery listed queued resources only, so a healthy hand-provisioned spot VM reported as
"No ACTIVE Queued Resource found" and every query tool refused to run against a TPU that was serving fine.
The Compute Engine migration retires that whole class of bug: there is no second namespace left to be blind
to, because the instance *is* the node. This rig has never provisioned anything, so it has no legacy node.

**`_resolve_node_id()` is the matching fix for the SSH-based tools** (`manage_vllm_docker`,
`run_vllm_benchmark`, `get_vllm_docker_logs`, `get_tpu_system_logs`). It tries an instance by that name,
then a listed instance named exactly `resource_id` or `<resource_id>-node`, then — last resort — the node
that discovery confirmed is serving vLLM, so the default `resource_id` still reaches a deployment named by an
earlier convention. The `<resource_id>-node` form is kept deliberately: it is what the Queued Resource era
derived, so a node left over from that path is still reachable. `_get_node_id()` is now just an existence
check on the instance — on this path there is nothing left to unwrap.

**Resource names are derived from the rig directory.** `RIG_NAME` in `server.py` is
`os.path.basename(...)` of the rig directory; `RESOURCE_ID` defaults to it and is the default `resource_id`
of every MCP tool, and the Makefile's `SERVICE_NAME` is `$(notdir $(CURDIR))`. So in this directory both
resolve to `tpu-vllm-v5p1-2b`. `RIG_NAME` also supplies the default `MCP_SERVER_NAME`, which names the
FastMCP server. Nothing here reads a *slot* out of
the directory — that is still forbidden. The whole name is read as an identifier, which is what keeps
sibling rigs off each other's capacity in a shared project and zone.

The derivation is a default, not a lock: `RESOURCE_ID` in `tpu.env` (or the environment), `MCP_SERVER_NAME`
in either, and `SERVICE_NAME`
on the make command line all win. **Renaming the rig directory orphans anything already provisioned** —
the tools will look for the new name and the old resource keeps billing. Pin the old name in `tpu.env`
before renaming, or destroy first.

**The MCP server name has to match the key the client registers it under**, because that key is what
prefixes every tool: `mcp__tpu-vllm-v5p1-2b__find_tpu`. All the rigs used to register as `tpu-devops`, so a
tool call was ambiguous whenever more than one was loaded, and a user-scope `tpu-devops` shadowed this rig
entirely (it has no committed `.mcp.json`). `mcp_config.json` is the committed example of the entry;
`make mcp-config` writes a real `.mcp.json` using `MCP_SERVER_NAME` (default `$(notdir $(CURDIR))`), merging
into any existing file rather than replacing it. `.mcp.json` is gitignored at the monorepo root.

Note the ordering constraint in `server.py`: `load_dotenv` now runs *before* `FastMCP(...)` is constructed,
because `MCP_SERVER_NAME` set in `tpu.env` would otherwise arrive too late to name the server. Don't move
the FastMCP construction back above the dotenv block.

**The Makefile path and the MCP tools finally point at the same object.** This is new as of the Compute
Engine migration and worth knowing, because the old behaviour was a long-standing trap. `make endpoint` /
`status` / `benchmark` / `query` `describe` an instance named `$(SERVICE_NAME)` = `tpu-vllm-v5p1-2b`, and the
MCP tools now create and manage an instance of exactly that name. Before the migration those were two
different TPU nodes — the Makefile looked for a `tpu-vm` while the agent created a Queued Resource whose node
was `<name>-node` — so `make status` reliably failed to see anything the agent had deployed.

`make deploy-tpu` and `create_tpu_instance` build the same `gcloud compute instances create` call from the
same `tpu.env` values (`MACHINE_TYPE`, `IMAGE_FAMILY`, `BOOT_DISK_SIZE`, `INSTANCE_SCOPES`). The one
remaining difference is the provisioning model knob: the Makefile has its own `TPU_PROVISIONING_MODEL`
(now `STANDARD` | `SPOT` | `FLEX_START`, matching Compute Engine's vocabulary), while the MCP tools take
`provisioning_model` as `flex-start` | `spot` | `on-demand` and translate. Same enum underneath, two
spellings above.

**Raw `/v1/completions` returns an empty completion on `-it` models.** `make query` and
`benchmarking_suite.py` use it, so an empty result there is expected, not a broken deploy. `server.py`
correctly uses `/v1/chat/completions` throughout — keep new code on the chat endpoint; raw completions are
only useful for prefill-only benchmarks.

**The comparison/plot scripts carry labels from two hardware generations ago.** This rig's code went
v6e-1 → v5e-1 → v5p without those scripts being touched: `compare_chips.py`, `compare_benchmarks.py`, and
`plot_grid.py` hardcode "v6e-4"/"v6e-1" titles and read CSVs out of sibling `../tpu-*-v6e*-devops-agent/`
directories that this rig does not have. Don't read any of those labels as describing v5p.

The v5e-1 and v6e-1 measurement artifacts that came with the fork were **removed** from
`benchmarks/reports/` and `benchmarks/runs/` on 2026-08-10 — they were byte-identical duplicates of the
tracked copies in `../tpu-vllm-v5e1-2b/`, and `benchmarks/rollup.py` counts per rig, so leaving them here
would have credited this rig with nine runs and a report it never measured. Both directories are empty and
waiting for the first real v5p run.

**This rig can reach v5p in exactly three zones, and flex-start narrows that to one.** Three separate limits
stack, and confusing them is the standing trap in this rig:

| Limit | Zones | How known |
| :--- | :--- | :--- |
| **Quota** (`TPUV5PPerProjectPerZoneForTPUAPI`) | 10 — `us-central1-a/b/c/f`, `us-east1-b/c`, `us-east5-a`, `europe-west4-a/b/c` | stated value of 128 cores, live 2026-08-10 |
| **Hardware** (`v5p-8` published) | **≥9** — `europe-west1-b/d`, `europe-west4-b`, `us-central1-a`, `us-east1-d`, `us-east5-a/b/c`, `us-south1-a` | `gcloud compute tpus accelerator-types list --filter="type=v5p-8"`, 2026-08-10 |
| **Reachable** (quota ∩ hardware) | **3** — `us-central1-a`, `us-east5-a`, `europe-west4-b` | the intersection of the two rows above |
| **Flex-start** | **1** — `us-east5-a` | Google's Flex-start VM doc |

**The hardware row is not three — that figure is the intersection, and this table used to mislabel it.**
Corrected 2026-08-10. The chip is installed in at least nine zones; the project holds stated quota in ten;
only three are in both. Seven quota zones have no v5p installed, and six zones with v5p carry no stated quota
for it — including `us-east5-b` and `-c`, next door to the rig's default. `tpu_zones_status.md` is seeded
from the **reachable** row, with the seven quota-only zones marked `No` so `find_tpu` skips them.

`@HARDWARE.md` holds the full breakdown, including the one zone (`europe-west1-c`) where the GCE machine-type
catalog and the TPU API disagree outright.

`GOOGLE_CLOUD_ZONE` defaults to `us-east5-a` for that reason — it is the only zone where the default
`PROVISIONING_MODEL=flex-start` can succeed. To use `us-central1-a` or `europe-west4-b`, pass
`provisioning_model="spot"` or `"on-demand"` as well as the zone; a flex-start request there will be refused.

**None of the above is verified by attempting creation** — unlike the v5e-1 result it replaces, which was
confirmed on 2026-08-04 by watching the API reject `v5litepod-1` outside `us-west4-a`. Treat the flex-start
row as documentation until a real request confirms it.

**The instance path takes three provisioning models.** `_provisioning_flags()` in `server.py` is the
one place that maps `flex-start` / `spot` / `on-demand` to gcloud flags; every creation tool
(`create_tpu_queued_resource`, `manage_queued_resource`, `find_tpu`) takes a `provisioning_model` argument
defaulting to `PROVISIONING_MODEL` in `tpu.env`. Two things do not generalize across them:

- **Only flex-start passes `--max-run-duration`** — gcloud documents that flag as flex-start-only. A spot or
  on-demand node has no automatic stop and bills until it is preempted or destroyed. `--valid-until-duration`
  bounds the *request*, not the run, so it is shared by all three.
- **Spot is metered by a different quota**, `TPUV5PPreemptiblePerProjectPerZoneForTPUAPI`
  (`TPU_SPOT_QUOTA_ID`), not `TPU_QUOTA_ID`. `find_tpu` and `get_zones_with_available_quota` pick the id from
  the provisioning model, so don't pass `quota_id` explicitly unless you mean to override that.
- **The v5p quota ids are not the v5e ids with the generation swapped.** v5e says `TPUV5sLitepod…`; v5p says
  `TPUV5P…` — capital `P`, no `Litepod` infix. And the **unit differs**: the v5p metric counts *cores*, so one
  `v5p-8` draws 8 against the default 128, where the v5e metric counted chips. Don't reason about how much
  quota a slice consumes by analogy with the other rig.

`tpu_zones_status.md` rows now carry a `[model]` prefix in the detail column and `find_tpu` only skips a zone
whose recorded failure was under the *same* model — a zone that rejects flex-start is not evidence about spot,
which is the whole reason spot exists here. Untagged rows predate this and read as flex-start.

The Makefile's separate hand-provisioned `tpu-vm` path has its own knob, `TPU_PROVISIONING_MODEL`
(`standard` | `spot` | `reservation-bound` — gcloud's vocabulary there, with no flex-start), surfaced as
`make deploy-tpu-spot` / `make deploy-tpu-ondemand`. Don't confuse the two spellings.

**`estimate_deployment_cost` reads live pricing — never reintroduce a rate table.** It queries the Cloud
Billing Catalog API (Compute Engine service `6F81-5844-456A`, where TPU SKUs live), matching on region,
`usageType`, and a description pattern per provisioning model. The previous hardcoded table said v5e was
$0.12/chip-hr against a $1.20 list rate — wrong by 10x, and undetectable from inside the code. If no SKU
matches, the tool says so rather than falling back to a guess; keep that property. Requires a working
`gcloud auth print-access-token` and the Cloud Billing API enabled.

Two naming traps in the catalog: flex-start is sold as **"DWS Defined Duration"** (Dynamic Workload
Scheduler) and drops the `Tpu` prefix (`DWS Defined Duration V5p`), and spot is `usageType: Preemptible`
spelled `TpuV5p attached to Spot Preemptible VMs`. The `Reserved …` and `Commitment v1: …` SKUs describe the
same chip in the same region and two of them are also `OnDemand`, so the patterns are anchored with `^`.
Pass `tpu_type="v5p"` (the default here) — the family is spelled `TpuV5p`, and the catalog names regions by
city, so `us-east5` is `… running in Columbus` and `europe-west4` is `… running in Netherlands`.

**A price existing does not mean capacity is obtainable**, and v5p demonstrates it twice over: the catalog
publishes a `DWS Defined Duration V5p running in Iowa` SKU for `us-central1`, yet Google's Flex-start doc
lists `us-east5-a` as the only v5p flex-start zone; and `europe-west4` quotes both on-demand and spot v5p
rates across a region whose only v5p zone is `europe-west4-b`. Verified against the live catalog 2026-08-10.

**`tpu.env` is the single source of truth for deployment parameters.** Project, region, zone, model,
accelerator type, and tensor-parallel size are defined once there and consumed by `server.py` (via
`load_dotenv`), `mcp-run.sh` (which is what the `mcp_config.json` files launch), the `Makefile` (via
`-include`), and `set_env.sh`. Change the zone there, not in five places. A real environment variable always
beats the file in all four consumers — `load_dotenv` doesn't overwrite, the wrapper only exports what's unset,
and the Makefile uses `?=` — so `make status ZONE=...` still works for a one-off. Defaults are `us-east5-a` /
`us-east5`. Still check what is actually running before assuming: `list_queued_resources` and
`discover_vllm_url` only look in the configured zone.

**`--tensor-parallel-size` is 1, and on this rig that is not a compromise.** One `ct5p-hightpu-1t-tpu` is
one v5p chip with 95 GiB of HBM — roughly 6x a v5e-1 — and E2B's bf16 weights are 8.97 GiB, so there is
nothing to shard and enormous room left for KV. If you see `4` anywhere it is copy-paste from the Cloud TPU
API path, whose 4-chip floor this rig exists to avoid.

**How v5p is spelled here.** Everything below lives in exactly one place — `tpu.env` — and is
env-overridable. Verified against the live API on 2026-08-10.

| Context | This rig |
| :--- | :--- |
| Prose, directory name | `v5p-1` / `v5p1` (one chip) |
| `MACHINE_TYPE`, `--machine-type` | **`ct5p-hightpu-1t-tpu`** |
| Image | `--image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e --image-project=ubuntu-os-accelerator-images` |
| Quota service | **`compute.googleapis.com`** — not `tpu.googleapis.com` |
| Quota id | `TPU-V5P-per-project-zone` |
| Spot quota id | `PREEMPTIBLE-TPU-V5P-per-project-zone` |
| Provisioning model | **UPPERCASE** `FLEX_START` / `SPOT` / `STANDARD` |
| Billing catalog family | `TpuV5p`, `DWS Defined Duration V5p` |

**What the directory name does not tell you.** `v5p1` is one chip under the monorepo's chip-count rule
(`@NAMING.md`). On the deprecated API path this rig could not have existed: its smallest v5p is `v5p-8` —
four chips — because slice names *there* count TensorCores and a v5p chip has two. A directory reading
`v5p4` against a flag reading `v5p-8` was the old trap, and the rule that survives both control planes is
the same one: **never derive a gcloud value from the directory name.**

**Don't destroy the instance unless asked.** Teardown is not part of routine debugging, and flex-start
capacity can take a long time to come back. Two behaviour changes from the old path: the delete is now
immediate and synchronous rather than an `--async` queued-resource delete, and a flex-start instance deletes
*itself* at `--max-run-duration`. An instance vanishing on its own after 4h is expected, not a failure.

## Auth and env

Requires both `gcloud auth login` (for the `gcloud` subprocess calls) and `gcloud auth application-default
login` (ADC, for the `google-cloud-secret-manager` client). `set_env.sh` must be **sourced**, not executed.
`init.sh` is a one-time bootstrap that blocks on `read` in its error path — don't run it non-interactively.

Env vars `server.py` reads: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_ZONE`, `GOOGLE_CLOUD_REGION`, `MODEL_NAME`,
`ACCELERATOR_TYPE`, `TPU_RUNTIME_VERSION`, `TPU_QUOTA_ID`, `TENSOR_PARALLEL_SIZE`, `LOCAL_DOCKER_IMAGE`,
`MAX_MODEL_LEN`, `MAX_NUM_BATCHED_TOKENS`, `LIMIT_MM_PER_PROMPT`, `TPU_NETWORK`, `TPU_SUBNETWORK`,
`RESOURCE_ID` and `MCP_SERVER_NAME` (both default to the rig directory name). The HF
token lives in GCP Secret Manager under the secret id `hf-token` — never log, return, or commit it.

`TPU_NETWORK` / `TPU_SUBNETWORK` default to empty, which means gcloud uses the project's default network.
`aisprint-491218` has only the auto-mode `default` network — it has no custom VPC. Setting these to a network
that doesn't exist fails creation in every zone, which is what a screenful of failed zones in
`tpu_zones_status.md` usually means.

## Tests

`test_agent.py` mocks the whole `mcp` module and the Google Cloud clients before importing `server`. Keep unit
tests offline: mock the cloud, subprocess, and network boundaries rather than reaching out. Because `mcp` is a
`MagicMock`, anything calling `mcp.list_tools()` needs an explicit `AsyncMock` patch — see `test_get_help`.

## Git

The git root is the **parent** directory, `/home/xbill/gemma4-dev` (`xbill9/gemma4-dev`) — this rig is one
subdirectory of that monorepo, alongside the `tpu-jax-*` and `tpu-pytorch-*` rigs. `git add .` from here
stages only this subdirectory; run git commands from the repo root when you mean the whole tree.

This rig was forked out of `/home/xbill/gemma4-queens`, which is still a separate repo with the older
`-devops-agent` naming. Nothing here is shared with it any more — don't look for this project's history
there.

**This rig has no benchmark artifacts of its own yet.** `benchmarks/reports/` and `benchmarks/runs/` are
empty. The v5e-1 and v6e-1 material that arrived with the fork was removed on 2026-08-10 (see the gotcha
above); the tracked originals live in `../tpu-vllm-v5e1-2b/benchmarks/`, and that is where to read them.

New runs go under `benchmarks/runs/<date>-<what>-<hw>/` with `<hw>` = `v5p1`, and a schema-1.1 report in
`benchmarks/reports/<date>-gemma4-e2b-v5p1.json`. Scripts that *write*
(`run_sweep.py`, `run_grid_benchmark.py`, `run_fast_sweep.py`, `benchmarking_suite.py --output`,
`plot_grid_benchmark.py`) still use bare filenames in the CWD on purpose: pinning a writer at an
archived run dir would overwrite a recorded measurement. File new output into a new dated run dir.

`benchmarks/serving-report.schema.json`, `benchmarks/README.md`, and `benchmarks/INDEX.md` are all
generated or synced from the monorepo root — see the root `CLAUDE.md`. Don't hand-edit them here.

`mypy` excludes `benchmarks/runs/` (`pyproject.toml`): each run dir carries its own `aggregate.py`,
and the shared module name is a fatal collision that aborts the whole mypy run. ruff still covers them.

`AGENTS.md` in this directory is maintained by a different tool and overlaps with this file — if you change a
convention here, check whether it needs the same change there. It has already drifted on two points: it claims
`ZONE`/`REGION` are hardcoded in `server.py` (they read the environment, `server.py:26-27`) and that
`get_help()` is hand-maintained (it is generated from `mcp.list_tools()`). This file is correct on both.
