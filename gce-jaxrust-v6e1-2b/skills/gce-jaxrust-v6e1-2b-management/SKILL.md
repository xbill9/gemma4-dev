---
name: gce-jaxrust-v6e1-2b-management
description: Manage Google Cloud TPU capacity on Compute Engine and serve Gemma 4 from Rust (rlx + libtpu's PJRT plugin), with Python JAX and vLLM as the alternative workloads. Use when the user asks about provisioning, finding, probing, listing, or destroying TPU VMs / flex-start capacity, Rust or JAX on TPU, XLA/PJRT/StableHLO from Rust, starting or debugging vLLM on TPU (v6e, v5p), TPU quotas and zones, TPU cost estimates, benchmarking TPU serving, or the TPU devops MCP agent. Triggers include "TPU", "flex-start", "v6e", "Trillium", "Rust on TPU", "PJRT", "libtpu", "XLA", "JAX on TPU", "vLLM on TPU", "TPU quota", "TPU stockout".
---

# TPU Management — Compute Engine path

Operate Google Cloud TPU infrastructure: acquire capacity, run the Rust/XLA engine
(the default), the Python JAX engine, or Gemma 4 under vLLM, and tear down. Two ways
to act:

1. **Preferred — MCP agent tools.** If the `gce-jaxrust-v6e1-2b` MCP server is
   connected in this session, use its tools (catalog below). They wrap the correct
   `gcloud` invocations, discovery, and retry/cleanup logic.
2. **Fallback — direct `gcloud`.** If the MCP server is not connected, either offer to
   register the bundled server (see "Registering the MCP server") or run the equivalent
   `gcloud` commands from `references/tpu-guide.md`.

## This rig provisions through Compute Engine only

There is **no queued-resource path here** — no `create_tpu_queued_resource`, no
`find_tpu`, no `gcloud alpha compute tpus`. Capacity comes from
`gcloud compute instances create --machine-type=ct6e-standard-1t`.

The Cloud TPU API is no longer under active development — bug and security fixes
only, no published sunset date — and **new generations from TPU7x (Ironwood) are
Compute Engine or GKE only**. So a rig's provisioning path is a fact with a shelf
life, and this one is already on the surviving side of it.

**Not every chip has a Compute Engine path, and the catalog will tell you yes when
the answer is no.** `ct5lp-hightpu-1t` and its siblings are listed in 26 zones, the
shared OS image family is literally named `…-tpu-v5e-v5p-v6e`, and there is a
Compute Engine v5-lite quota metric — and a create is still refused outright:

```
ERROR: (gcloud.compute.instances.create) Could not fetch resource:
 - This user agent is not allowed to use the machine type [ct5lp-hightpu-1t].
```

That is not a quota error and not a does-not-exist error. Those artefacts exist
because the TPU API and GKE are implemented on Compute Engine underneath. **Catalog
presence is not creatability.** v6e, v5p and TPU7x have a Compute Engine path;
**v5e does not**, so v5e work belongs on a `tpu-*` rig. The agent refuses a v5e
accelerator with that explanation rather than a generic "unsupported".

One documentation trap, and it misleads in both directions: *Request TPU Flex-start
VMs* states "You must use the queued resources API to use TPU Flex-start VMs". That
is **true for v5e and out of date for everything else** — the page sits in the
deprecated API's doc set. Believe the Compute Engine *provisioning models* page for
v5p, v6e and TPU7x instead.

## The flag mapping

| Cloud TPU API | Compute Engine (this rig) |
| :--- | :--- |
| `--accelerator-type=v6e-1` | `--machine-type=ct6e-standard-1t` |
| `--runtime-version=v2-alpha-tpuv6e` | `--image-family=…` **plus** `--image-project=` |
| `--valid-until-duration` | `--request-valid-for-duration` |
| `--provisioning-model=flex-start` | `--provisioning-model=FLEX_START` (SCREAMING_CASE) |
| `--max-run-duration` (flex-start only) | `--max-run-duration` (**any** model) + `--instance-termination-action=DELETE` |
| — | `--scopes=cloud-platform` (**required** if the startup script reads a secret) |
| — | `--maintenance-policy=TERMINATE` (**required** — TPU instances cannot live-migrate) |
| QR produces a `<id>-node` | **the instance IS the node** |
| `gcloud compute tpus tpu-vm list` | `gcloud compute instances list` |
| `gcloud compute tpus tpu-vm ssh` | `gcloud compute ssh` |

Serving does not change: same chip, same engine build, same flags. **Treat a
migration as a refactor, not a performance decision.**

## Bundled files

- `mcp/server.py` — the FastMCP DevOps agent (snapshot of the repo-root `server.py`;
  the live copy at the repo root is authoritative if the two differ).
- `mcp/project-setup.sh` — one-command installer: copies this skill into a target project and
  registers the MCP server (see "Registering the MCP server").
- `mcp/startup_script_jaxrust_template.sh` — the startup script for `workload="jaxrust"`
  VMs, the default. Installs a pinned Rust toolchain plus `protoc`/`clang`/`binutils`,
  then libtpu with `--no-deps` — no JAX, no jaxlib. It asserts the installed `libtpu.so`
  exports `GetPjrtApi` before reporting ready, because a file of that name is not yet a
  PJRT plugin. It stops at a prepared environment: nothing is fetched, built or served.
- `mcp/startup_script_jax_template.sh` — the startup script for `workload="jax"` VMs
  (installs a current CPython + `jax[tpu]` on the bare VM, no docker and no HF token,
  and asserts a TPU device is visible before reporting ready).
- `mcp/startup_script_template.sh` — the startup script for `workload="vllm"` VMs
  (installs Docker if missing, pulls `vllm/vllm-tpu:nightly`, serves the model).
- `mcp/startup_script_cpu_template.sh` — the CPU debug box script (same JAX stack minus libtpu).
- `references/tpu-guide.md` — the TPU getting started guide: prerequisites,
  flex-start capacity zones per TPU family, `gcloud` creation templates,
  persistent-disk + startup-script patterns, quota metrics and request procedure,
  troubleshooting/FAQ. Read it when working without the MCP tools, diagnosing
  provisioning failures, or answering quota/capacity/billing questions. **Where it
  describes the queued-resources API, it describes the path this rig left** — the
  flag mapping above is the translation.

## Registering the MCP server

Easiest path — run the bundled installer (idempotent; installs this skill into the
target project and writes the `gce-jaxrust-v6e1-2b` entry into the project's `.mcp.json`,
using the system `python3` — it warns if the pip deps below are missing but never
creates a venv):

```bash
mcp/project-setup.sh /path/to/project --project <gcp-project-id>   # one project
mcp/project-setup.sh --global                                      # all projects (user scope)
# from the skill repo root: make init TARGET=/path/to/project ARGS='--project <id>'
```

Run `mcp/project-setup.sh --help` for all options (`--model`, `--accelerator`, `--tp`,
`--server-name`, `--skip-deps`). Then restart Claude Code in the target project and
approve the server when prompted; `/mcp` should list `gce-jaxrust-v6e1-2b`.

Manual alternative:

```bash
claude mcp add gce-jaxrust-v6e1-2b \
  --env GOOGLE_CLOUD_PROJECT=<project-id> \
  --env MODEL_NAME=google/gemma-4-E2B-it \
  --env ACCELERATOR_TYPE=v6e-1 \
  --env TENSOR_PARALLEL_SIZE=1 \
  --env GOOGLE_CLOUD_ZONE=europe-west4-a \
  -- python .claude/skills/gce-jaxrust-v6e1-2b-management/mcp/server.py
```

Requires: `pip install -r mcp/requirements.txt`, an authenticated `gcloud` CLI, and
the Compute Engine API enabled. (`gcloud alpha` may report as a missing component
while working fine — the component manager is disabled by design on apt installs and
alpha ships in the base package. Ignore it.)

Config comes from env vars — `GOOGLE_CLOUD_PROJECT` (falls back to the active gcloud
config), `GOOGLE_CLOUD_ZONE` (default `europe-west4-a`), `GOOGLE_CLOUD_REGION`,
`MODEL_NAME`, `ACCELERATOR_TYPE`, `TENSOR_PARALLEL_SIZE`, `INSTANCE_NAME`,
`PROVISIONING_MODEL`, `REQUEST_VALID_FOR`, `MAX_RUN_DURATION`, `BOOT_DISK_SIZE_GB`,
`IMAGE_FAMILY`, `IMAGE_PROJECT`, `GCE_QUOTA_ID`, `GCE_SPOT_QUOTA_ID` — and `get_help`
prints the live values. A Hugging Face token must exist as Secret Manager secret
`hf-token` (save one with `save_hf_token`) before creating a `workload="vllm"` VM;
`workload="jax"` VMs need none.

## The four provisioning models

The choice drives what you pay, which quota you spend, and how you fail.

| Model | How you get capacity | Max run | Quota spent | v6e, europe-west4 |
| :--- | :--- | :--- | :--- | :--- |
| `flex-start` | queues, up to a 2h wait | 10 min – 7 days | preemptible → standard | **$1.35**/chip-hr |
| `spot` | immediately, if available | unlimited | preemptible | $1.78/chip-hr |
| `on-demand` | immediately, if available | unlimited | standard (family) | $2.97/chip-hr |
| `reservation-bound` | reserved ahead, if approved | up to 90 days | with the reservation | **no list rate** |

Rates read from the Cloud Billing Catalog 2026-08-11.

- **Flex-start is the default worth reaching for, and it is the cheapest.** It
  undercuts spot on v6e in both regions priced, and is less than half on-demand. You
  trade immediacy: the request queues rather than failing, and the instance
  self-terminates at `--max-run-duration`.
- **Spot is not the cheap option here, despite the name.** It costs *more* than
  flex-start on v6e, and the ordering does not invert on v5e either ($0.607 vs
  $0.60). Its real advantage is that it **does not queue** — which is what makes it
  the capacity probe. Read the rate rather than assuming.
- **Reservation-bound has no list rate in the billing catalog.** What it costs is
  whatever the reservation was priced at. `estimate_deployment_cost` says "read the
  reservation" rather than substituting the on-demand SKU.

## Standard lifecycle

1. **Status first.** `get_system_status`. Never create before checking what exists.
2. **Acquire capacity.** `create_tpu_vm_instance` for a known-good zone, or
   `find_tpu_vm` to sweep. The sweep is usually the right move: candidate zones are
   the intersection of "the catalog publishes this machine type here" and "the region
   holds quota on the pool this provisioning model spends".
3. **Wait for ready — `RUNNING` is not it.** `wait_for_jaxrust_ready` /
   `wait_for_jax_ready` / `wait_for_vllm_ready`, by workload. See the section below.
4. **Build and prove the chip.** On the Rust path: `deploy_jaxrust_engine` uploads
   `rust/`, builds it release-mode on the VM, and finishes by running `xla-probe` —
   which compiles a StableHLO matmul on the chip and checks the numbers coming back.
   Re-run it any time with `verify_rust_tpu`. Then `manage_jaxrust_server`
   action='start'. On the other paths: `verify_jax_tpu` for a JAX box,
   `manage_vllm_docker` for a serving one.
5. **Verify.** `verify_model_health`, `get_vllm_endpoint`, `get_model_details`,
   `query_gemma4` (`include_stats=True` for TTFT/throughput). Health checks,
   queries and benchmarks auto-target whatever model the server actually loaded (via
   `/v1/models`), so they keep working after a deploy-time `model_name` override.
6. **Benchmark (optional).** `run_vllm_benchmark`; `save_result=True` also returns
   the run as a `throughput.sweep[]` entry for `benchmarks/serving-report.schema.json`.
7. **Tear down.** `destroy_tpu_vm_instance`. Billing runs until deletion. The
   two-object lifecycle has collapsed — no queued resource owning a node you did not
   name, and teardown needs no `--force`.

## Three things that do not fail loudly

Almost nothing on this path fails loudly. These three are why.

### RUNNING does not mean ready

This is the most misleading signal here. A queued resource reached ACTIVE only once
its node was up. **An instance is RUNNING the moment the VM boots** — before the
startup script has pulled an image, loaded a model, or done anything at all. During
a completely dead boot the instance list still says:

```
NAME              ZONE            MACHINE_TYPE       STATUS
gce-jaxrust-v6e1-2b   europe-west4-a  ct6e-standard-1t   RUNNING
```

It says that indefinitely. Nothing distinguishes a dead boot from a healthy one
except reading the startup log or curling the port. **Any readiness check that
trusted ACTIVE is now wrong** — by several minutes on a good day and forever on a
bad one. Use `wait_for_jax_ready` / `wait_for_vllm_ready`, which read the serial
console, or `get_tpu_vm_serial_log` to watch by hand.

### PENDING is either no quota or no capacity, and they look identical

A flex-start create in a region with zero quota queues indefinitely. A flex-start
create in a region with 1536 chips of quota and no hardware does exactly the same.
Neither reports the actual problem, and it is not even consistent — the same create
in a third zone came back immediately with an explicit stockout. **You cannot infer
the cause from the behaviour, and "did my create succeed" is not a quota test.**

The routine:

1. **Probe capacity with a spot create** — `probe_zone_capacity`. Spot does not
   queue; it fails fast and names the reason, so it is a free check that takes
   seconds. A `stockout` means real scarcity and no amount of quota will help. If it
   provisions instead, capacity exists (the tool deletes the probe immediately) and
   the problem is quota. Spot and flex-start draw on the same preemptible pool, so
   this probes the **zone**, not your entitlement.
2. **Try the sibling zones, not just the region.** Quota is regional; capacity is
   zonal, and they diverge sharply — a stockout in `-a`, a stockout in `-c`, and a
   working instance in `-b`, all within minutes against the same regional quota.
3. **Check both quota metrics** (below), not one.
4. **Only then request more quota.**

### Quota is a ceiling, not an allocation

**Holding quota does not mean the hardware is there.** Single v6e chips have been
scarce nearly everywhere: three of five zones checked held full quota and no chips at
all. One zone provisioned an instance, and a request a minute later was refused for
stockout. **Availability moves faster than you can test against it**, let alone plan
around. Treat quota as permission to ask; flex-start's queue is the mechanism that
actually gets you a chip, because it waits rather than failing.

## Quota: which metric, and why one reading misleads

Three separate traps, in order.

**1. TPU API quota does not come with you.** The two control planes meter against
completely disjoint pools. This project holds 512 v6e chips in `us-east5` on the TPU
API and, on Compute Engine, nothing at all in the same region for the same silicon.

**2. Which quota you spend depends on the provisioning model.**

| Provisioning model | Quota id it spends |
| :--- | :--- |
| `flex-start` | `PREEMPTIBLE-TPU-V6E-per-project-region`, **falling back to** the family quota |
| `spot` | `PREEMPTIBLE-TPU-V6E-per-project-region` |
| `on-demand` | `TPUS-PER-TPU-FAMILY-per-project-region`, `tpu_family=CT6E` |

Flex-start spending the *preemptible* pool is counterintuitive — flex-start is not
preemptible in behaviour; once granted it runs uninterrupted for up to seven days —
and nothing in the flag names hints at it. The second sentence of the docs matters as
much as the first: *"If your project lacks preemptible quota, then standard quota is
consumed."* So **a region is usable for flex-start if either pool has room.** Do not
write a region off on one listing alone.

There is **no non-preemptible per-generation v6e id at all**, which is why on-demand
falls back to the generic family quota. v4, v5e and v5p each publish their own pair,
so this is a v6e/TPU7x quirk rather than a rule.

**3. The obvious command does not show either of them.**
`gcloud compute regions describe` answers confidently and wrongly — its quota list
carries only the older v5-era metrics (`TPU_LITE_PODSLICE_V5` and friends), none of
which governs v6e. Use `get_zones_with_available_quota` (which reads **both** ids), or
ask by name:

```bash
gcloud alpha quotas info describe PREEMPTIBLE-TPU-V6E-per-project-region \
    --service=compute.googleapis.com   # flex-start and spot
gcloud alpha quotas info describe TPUS-PER-TPU-FAMILY-per-project-region \
    --service=compute.googleapis.com   # on-demand
```

**Read both, because their defaults are opposite.** A region absent from the family
listing inherits **0**; a region absent from the preemptible listing inherits **1536**.
A region that looks dead in one listing may have plenty of headroom in the other. An
unset value also reads identically to a zero one, so a blank does not tell you the
hardware is missing — check `machine-types list` for that.

### Requesting quota

```bash
gcloud quotas preferences create \
  --service=compute.googleapis.com --project=YOUR_PROJECT \
  --quota-id=PREEMPTIBLE-TPU-V6E-per-project-region \
  --dimensions="region=us-east5" \
  --preferred-value=32 \
  --preference-id=preemptible-tpu-v6e-us-east5 \
  --justification="..."
```

The dimension keys differ per metric — the family quota takes region **and**
`tpu_family`, the preemptible one takes region alone. Read them off
`gcloud quotas info describe <quota-id>` rather than guessing. Check filings with
`gcloud quotas preferences list`.

What to expect: decisions come back **within seconds**, automated, with the verdict in
`quotaConfig.stateDetail`. **The size of the ask is not the variable** — the same
0 → 32 request was approved in two regions and denied in three, and retried at 8
chips the denials were identical. The outcome tracked the *region* every time; there
is no magic number. **Denials may track capacity**: the two denying regions that were
then probed both refused a spot create for lack of capacity, so do not read a denial
as a judgement about your project. And **you cannot ask for less than you hold** —
`FAILED_PRECONDITION: decreases effective quota unsafely` means that region already
sits at the 1536 default and you asked for 32. Read the current value per metric first.

## MCP tool catalog (by task)

**Capacity & lifecycle:** `create_tpu_vm_instance` (Compute Engine create with the
proven flags: 200GB boot disk, cloud-platform scopes, TERMINATE maintenance policy;
`workload="jaxrust"` by default), `find_tpu_vm` (zone sweep, records failures),
`probe_zone_capacity` (spot probe — stockout vs quota, in seconds),
`get_zones_with_available_quota` (both metrics), `wait_for_jaxrust_ready`,
`wait_for_jax_ready`, `verify_jax_tpu`, `wait_for_vllm_ready`, `create_cpu_debug_vm`,
`list_tpu_vm_instances`, `destroy_tpu_vm_instance`, `get_tpu_vm_serial_log`,
`get_tpu_vm_endpoint`, `get_deployment_command`, `estimate_deployment_cost`, `find_gpu`

**Rust engine:** `deploy_jaxrust_engine` (upload + release build + probe),
`verify_rust_tpu` (re-run the probe), `manage_jaxrust_server` (start/stop/status/logs)

**Serving:** `manage_vllm_docker`, `get_vllm_endpoint`, `save_hf_token`

**Health, logs & diagnostics:** `get_system_status`, `verify_model_health`,
`get_model_details`, `get_metrics`, `get_vllm_docker_logs`, `get_tpu_system_logs`,
`get_cloud_logging_logs`, `analyze_cloud_logging` (Gemma-4-powered log triage)

**Inference & benchmarking:** `query_gemma4` (`include_stats=True` for
latency/throughput), `run_vllm_benchmark`

Every agent in this repo also exposes `get_help` for its live configuration.

## Rust/XLA on TPU — the default workload

This is what the rig is for: Gemma 4 served from a process with no Python in it. The
graph is built in rlx's JAX-shaped IR, lowered to HLO, and executed through libtpu's
PJRT plugin.

```
create_tpu_vm_instance(workload="jaxrust")   # RUNNING is not ready
wait_for_jaxrust_ready                       # toolchain + libtpu installed. Nothing built.
deploy_jaxrust_engine                        # uploads rust/, builds release, RUNS THE PROBE
manage_jaxrust_server(action="start", model_path="/path/to/checkpoint")
```

Each step answers a strictly smaller question than the next one down. That is the whole
design: when serving fails, the last step that still passes tells you where to look.

**What "ready" means here, and what it does not.** `wait_for_jaxrust_ready` returning
green means a pinned Rust toolchain is installed, `protoc`/`clang`/`binutils` are
present, and a `libtpu.so` exporting `GetPjrtApi` is on disk with `LIBTPU_PATH` pointing
at it. It does **not** mean anything has compiled, and it does not mean the chip has
executed an instruction. `deploy_jaxrust_engine` settles both, in that order.

**The probe asserts on a computed value.** `dlopen` on `libtpu.so` succeeds on a host
with no chip attached, exactly as `import jax` succeeds with no TPU backend — which is
why the JAX path asserts on `jax.devices()` rather than on the import. `xla-probe` goes a
step further: it compiles a StableHLO matmul, runs it, and checks that every element of
the result is what arithmetic says. A plugin that loads, a client that creates and a
device that lists are three separate facts, and none of them says the MXU computed
anything. It also prints each device's `largest_free_block_bytes` next to the HBM limit,
which is the number that decides whether a weight tensor can actually be placed.

**Field notes on the Rust path:**

- **`protoc` and `clang` are mandatory and fail late.** `pjrt-sys` runs `prost-build` and
  `bindgen` in its build script; without `protoc` the build dies with ``Could not find
  `protoc` `` minutes in, nowhere near the cause. The startup script installs both. A
  build failure with that message means the VM was not booted with the `jaxrust` workload.
- **The rlx model crates trail the framework.** `rlx` publishes 0.2.14, but `rlx-gemma`
  0.2.11 pins `rlx-runtime =0.2.11` exactly, so the workspace is held at 0.2.11
  throughout. Bumping `rlx` alone fails version selection.
- **`rlx-gemma` is GPL-3.0-only** while the rig is Apache-2.0, so a binary built with the
  `gemma` feature is a GPLv3 combined work. `JAXRUST_CARGO_FEATURES` controls it; drop
  `gemma` and the server still starts and reports the device but refuses generation.
  `xla-probe` never links it. See `rust/NOTICE.md`.
- **Only one process may hold the TPU.** A crashed or backgrounded run keeps the chip;
  `manage_jaxrust_server` action='stop' before retrying.
- **Started is not serving.** The engine compiles the graph before it binds port 8000, so
  a port that is not answering yet is a load in progress. Watch for
  `JAXRUST-SERVER: listening` in action='logs'.
- **Nothing on this path has run on a chip yet.** As of 2026-08-28 the workspace compiles
  and the templates render; that is all. Do not write up a result from it that you have
  not seen.

## JAX on TPU (bare dev VMs) — the parity oracle

For Python JAX work — kernels, benchmarks, and the reference implementation the Rust
engine is differenced against — provision with `workload="jax"`:

1. `create_tpu_vm_instance()` or `find_tpu_vm()`. The startup script installs
   `JAX_PYTHON_VERSION` (default 3.13) from deadsnakes and pip-installs
   `JAX_PIP_SPEC` (default `jax[tpu]`) into it directly — no venv, per this repo's
   standard; the dedicated interpreter provides the isolation. It upgrades the Python
   packaging stack and all configured extras, and installs the latest compiler/build
   tools available from the configured Ubuntu repositories. Default extras include the
   tokenizer stack (`transformers`, `tokenizers`, `sentencepiece`, Jinja 3.1+)
   required by the validation harness. No docker, no Hugging Face token, no 200 GB
   image pull, so it boots in a couple of minutes rather than ~8.5.
   The script then upgrades `libtpu` independently to the newest wheel on Google's JAX
   releases index, overriding the conservative version bundled by the `jax[tpu]`
   extra; its TPU device assertion is the compatibility gate.
2. `wait_for_jax_ready` polls the serial console for
   `JAX-BOOTLOADER: TPU environment ready.` and fails fast on
   `JAX-BOOTLOADER: FAILED`. `verify_jax_tpu` re-runs the device check over SSH.
3. Run workloads with `python3.13` (or whatever `JAX_PYTHON_VERSION` is set to),
   **not** `python3` — the system interpreter is 3.10 and has no JAX.

**Why the script asserts on `jax.devices()`:** importing `jax` succeeds on a host
with no TPU backend, so an import-only check reports success on a broken VM. The
script exits non-zero unless a device with `platform == "tpu"` is present, and only
then prints the ready marker.

**Field notes:**

- Ubuntu 22.04's system Python is 3.10, which resolves `jax[tpu]` to an old
  release (0.6.x). If you need a specific JAX, pin `JAX_PIP_SPEC`
  (e.g. `jax[tpu]==0.11.0`) — and remember `libtpu` comes from the JAX releases
  index, not PyPI.
- The system `pip` (22.x) predates `--break-system-packages`; the script
  bootstraps pip into the new interpreter with `get-pip.py` instead.
- `add-apt-repository` is not on the accelerator image by default — the script
  installs `software-properties-common` first.
- Direct SSH often times out; use `--tunnel-through-iap`.
- Only one process may hold the TPU. A crashed or backgrounded run keeps the chip
  ("The TPU is already in use by process with pid N"); kill it before retrying.

## The image is not the runtime version

`ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` has **no Docker on PATH at first boot**.
A startup script written against the TPU API's `v2-alpha-tpuv6e` runtime — which
evidently shipped it — dies 100 seconds in when ported verbatim:

```
+ sudo docker pull vllm/vllm-tpu:nightly
sudo: docker: command not found
ERROR: Failed to pull vLLM Docker image after multiple retries. Exiting.
```

…while the instance reports RUNNING throughout. The bundled `workload="vllm"`
template installs `docker.io` first. **Fix it in three places, not one:** the startup
script, any Docker command your tooling runs over SSH (the recovery tool you reach for
after a failed boot must not fail the same way), and any copy-pasteable deploy
one-liner you emit.

**The general form:** your startup script was written against a runtime version that
gave you things for free. Whatever yours assumes, the instance will sit there
reporting RUNNING while it fails.

## vLLM on TPU — required flags (Gemma 4)

When composing or reviewing a vLLM serve command for TPU, use:
`--tensor-parallel-size` matching the chip count, `--max-model-len 16384`,
`--disable_chunked_mm_input`, `--max_num_batched_tokens 4096`,
`--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4`,
and `--limit-mm-per-prompt '{"image":4,"audio":1}'` for multimodal
(the agent uses `{"image":0,"audio":0}` for text-only serving).
Image: `vllm/vllm-tpu:nightly`, run with `--privileged --net=host --shm-size 10gb`
and `HF_HOME=/dev/shm`.

Upstream references: [vLLM TPU docs](https://docs.vllm.ai/projects/tpu/en/latest/),
[Recommended Models & Features](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/)
(the support matrix — check it before serving quantized checkpoints),
[vLLM Recipes](https://recipes.vllm.ai) (per-model deployment guides), and the
[tpu-inference GitHub repo](https://github.com/vllm-project/tpu-inference)
([releases](https://github.com/vllm-project/tpu-inference/releases) track newly
landed quantization/model support).

Known-broken (verified on `vllm-tpu:nightly`, Jul 2026): the Gemma 4 **E2B QAT**
checkpoints do not load on TPU in any form — `-qat-w4a16-ct` fails with
"compressed-tensors scheme for layer 'per_layer_model_projection' is not yet
supported in the JAX path", and `-qat-q4_0-unquantized` fails on both the JAX and
`MODEL_IMPL_TYPE=vllm` (torchax) paths with "weights not initialized from
checkpoint: layers.15-34 self_attn.k_norm.weight" (the export omits k_norm for
the upper KV-sharing layers). Serve the plain `google/gemma-4-E2B-it` instead — the
JAX path in this repo is the one that loads the QAT export.

## Discovery and SSH moved, and the old calls go quiet

A `ct6e-*` instance is an ordinary Compute Engine instance that happens to carry a
TPU, so the old API cannot see it:

```console
$ gcloud compute instances list --filter="name=gce-jaxrust-v6e1-2b"
NAME              ZONE            MACHINE_TYPE       STATUS
gce-jaxrust-v6e1-2b   europe-west4-a  ct6e-standard-1t   RUNNING

$ gcloud compute tpus tpu-vm list --zone=europe-west4-a
$
```

Empty. **No error, no warning** — tooling on the old call simply believes nothing is
running. Two field shapes move with it: status is `status: RUNNING` rather than
`state: READY`, and the external IP moves from
`networkEndpoints[].accessConfig.externalIp` to
`networkInterfaces[].accessConfigs[].natIP`. A copied-across status check does not
throw; it just sorts every healthy instance to the bottom of a ranking, which you
notice the day you have two.

SSH moves too, and it is the call site people miss — because everything that manages
your container, tails logs, reads `journalctl` or runs a benchmark goes through it,
and those are precisely the tools you reach for *when something has already gone
wrong*. **Grep for the old command; do not trust one test over one function.**
`tests/test_server.py::OffTheCloudTpuApiTests` scans the whole source text for
`tpu-vm` and `queued-resources` for exactly this reason.

## Three flags that fail late

- **`--scopes=cloud-platform`** — required if your startup script reads a secret.
  Without it the VM boots fine and then spins for ~30 minutes before giving up, so the
  symptom is a slow startup followed by what looks like a token problem. Also grant the
  default compute SA `roles/secretmanager.secretAccessor` on `hf-token`.
- **`--boot-disk-size`** — the image default is 10 GB, which will not hold a vLLM TPU
  image. Fails after a clean boot, mid-pull. If already created, recover without losing
  the capacity grant: `gcloud compute disks resize <name> --size=200GB` then
  `gcloud compute instances reset <name>` — never delete and recreate, which forfeits
  the grant and restarts the max-run clock.
- **`--maintenance-policy=TERMINATE`** — required, because a TPU instance cannot
  live-migrate.

Two more worth knowing: flex-start instances run for a minimum of 10 minutes and a
**maximum of seven days**, so set `--max-run-duration` explicitly rather than
discovering the boundary. And you **cannot suspend one** — a standalone flex-start
instance can be stopped, but suspend and recreate are unavailable. Keep state you care
about on a separate disk or in GCS.

## Troubleshooting quick reference

| Symptom | Likely cause | Check |
| :--- | :--- | :--- |
| PENDING for hours | quota or capacity — identical from outside | `probe_zone_capacity`; stockout means capacity, and usually it is |
| `not allowed to use the machine type` | that generation has no Compute Engine path | use a `tpu-*` rig for that chip |
| RUNNING but nothing serves | startup script died | `get_tpu_vm_serial_log`; curl the port |
| `docker: command not found` | the CE image ships no Docker | install `docker.io` before pulling |
| Out of disk mid-pull | 10 GB image default | `--boot-disk-size` |
| Secret access hangs 30 min | missing `--scopes=cloud-platform` | recreate with the scope |
| Flex-start VM disappeared | it reached `--max-run-duration`, max seven days | set the duration explicitly; it is not unlimited |
| `tpu-vm list` returns nothing | wrong API for a `ct6e-*` instance | `gcloud compute instances list` |
| SSH says not found | wrong SSH surface | `gcloud compute ssh`, not `tpus tpu-vm ssh` |
| Quota looks fine but nothing works | reading `regions describe`, which shows v5 metrics only | Cloud Quotas API, by metric name |
| `decreases effective quota unsafely` | requesting less than you hold | read the current value first |
| `gcloud alpha` reported missing but works | component manager disabled by design | ignore it; alpha ships in the base package |

## Cautions

- `destroy_tpu_vm_instance` deletes infrastructure. Billing runs until deletion and
  cannot be paused — always confirm teardown of idle resources with the user, and
  remind them an instance left running self-deletes at `--max-run-duration`.
- `probe_zone_capacity` **creates a real instance** when capacity exists, and deletes
  it immediately. If the delete fails it says so loudly — that instance is billing.
- Data on the VM is lost at teardown. Persist on a separate disk or GCS.
- A v6e chip has 32 GB HBM, twice a v5e chip, and `_min_chips_for_model` is sized on
  the target chip's HBM: a bf16 12B fits one v6e chip and would OOM on a v5e-1, and
  int4 is not an automatic pass (26B int4 is still ~13 GB). Unknown checkpoints are
  never blocked.
- The chip-count guard applies to `workload="vllm"` only — a `workload="jax"` VM
  loads no model, so there is nothing to size.
