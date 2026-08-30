# local-jax-cpu-2b

Serve **`google/gemma-4-E2B-it`** with **pure JAX on the local CPU** — no accelerator, no
cloud, no provisioning.

> **Status: this rig has served nothing yet.** Forked from
> [`gpu-jax-g4dn-2b`](../gpu-jax-g4dn-2b/) on 2026-08-29 and retargeted from an EC2 GPU to
> whatever machine it is checked out on. The engine and the model port are inherited
> unchanged; the entire cloud control plane is **deleted**, not disabled. 123 tests pass
> offline and `benchmarks/` is deliberately empty.

## Why this exists

**It is the zero point.** Every other rig in this monorepo measures an accelerator, and none of
them can separate the chip from the engine — because there is no run without a chip. This is
the only place a number for this JAX port can be attributed entirely to the model code and the
XLA CPU backend.

It is also the only rig with **no hourly cost and no capacity risk**, which makes it the right
place to reproduce an engine-level bug before spending a capacity cycle on one. That is not
hypothetical: `docs/padding-window-eviction.md` describes a silent correctness bug in the
shared model port that was found on a GPU and *verified on CPU*, because nothing about the
mechanism is hardware-specific. That verification could have happened here first.

| | every other rig | **this rig** |
| --- | --- | --- |
| control plane | a cloud API that can refuse | **none — nothing is provisioned** |
| cost | per hour | **none** |
| capacity risk | spot reclamation, quota, zone | **none** |
| what fails first | the accelerator's memory budget | **host RAM, and it does not raise** |
| hardware in the name | a SKU the rig provisions | **whatever machine you are on** |

## What is different, and it is mostly subtraction

`server.py` went from 1,674 lines to about 1,000, and the deletions are the point: no EC2
launch, no AMI resolution, no spot handling, no SSM Run Command, no Secrets Manager, no S3
compilation-cache sync, no systemd unit, no cloud-init, no `boto3`. There is no
`deploy_jax_server`, because the payload runs in place.

What replaced them is the same *shape* of problem, one layer down:

| Cloud rig | Here |
| --- | --- |
| `check_g4dn_quotas` | `check_host_capacity` |
| `verify_gpu_arch` | `verify_cpu_backend` |
| `create_*_instance` / `terminate_*` | `start_jax_server` / `stop_jax_server` |
| `get_install_progress` (cloud-init) | `check_dependencies` |
| `get_jax_logs` (systemd journal) | `get_jax_logs` (one logfile) |
| AMI + instance-type selection | `fetch_checkpoint` |

## Memory is the constraint, not speed

It will be slow, and that is expected. What will actually stop a serve is RAM — and the
failure mode is worse than a cloud rig's, because **exceeding a quota is refused at the API
and exceeding host RAM is accepted and paid for in swap.** A thrashing serve looks exactly
like a loading one.

| Config | Weights |
| --- | ---: |
| dense (`ple_bits=0`) | 9.257 GB |
| **`ple_bits=4` — the default here** | **5.752 GB** |
| `ple_bits=4` + `int8_lm_head` | 6.155 GB |

Plus about **1.6 GB** of prefill transient at a bucket below 4K. Run `check_host_capacity`
before a load; `start_jax_server` refuses outright when the arithmetic does not close.

**`INT8_LM_HEAD` is OFF here, inverting every GPU sibling's default.** It *adds* 0.403 GB to
buy +2.3% throughput by halving bytes read per decode step — a memory-bandwidth trade, and the
wrong one when the constraint is resident bytes.

## bfloat16 is forced by memory, not chosen for speed

XLA:CPU has no bf16 datapath; it upconverts to fp32 in front of every use — the same tax that
is 54% of decode on the Turing siblings. The difference is that **there is no escape here**:

- **float16 does not help.** A CPU has no 16-bit float datapath of any kind. On Turing, fp16
  tensor cores were the escape; there is nothing to escape into.
- **float32 storage would avoid the conversion and does not fit** — 18.5 GB of weights.

So do not open the ticket the GPU rigs have open. `docs/bf16-weights-on-turing.md` records
three placements tried and measured there; here the option space is smaller, not larger.

## Quick start

```bash
python3 -m pip install -r requirements.txt      # system-wide; no virtualenv
python3 -m unittest discover -s tests -v        # 123 offline tests
./project-setup.sh                              # register the MCP server
```

Then, through the MCP agent:

```
check_host_capacity → check_dependencies → verify_cpu_backend → fetch_checkpoint
→ start_jax_server → get_jax_logs → verify_model_health → query_model / get_metrics
```

`make serve` runs the same command in the foreground; `get_serve_command` prints it.

## Naming

`local-jax-cpu-2b` — platform `local`, runtime `jax`, hardware `cpu`, model `2b`.

**`local` was added to `NAMING.md` on 2026-08-29** as a sixth platform value. The spec's rule
is that a new value is earned by a genuinely different execution target, and this is the one
target with no control plane at all: nothing is provisioned, nothing is billed, and there is no
API that can refuse. `gce` would have claimed a Compute Engine control plane this rig does not
have.

The bare four-slot name is a positive claim: this is the **dense reference checkpoint**, not a
quantised export. `PLE_BITS=4` is a runtime lever, not a weight encoding, so it does not earn a
fifth slot.

## Measurement discipline

**A report from this rig must record the host in its own fields.** Every sibling's hardware
slot names a SKU the rig provisions, so the rig name carries the hardware; `cpu` does not — it
names whatever machine this checkout is on. Two runs of this rig on two machines are not
comparable, and the directory name cannot tell you so. `tune_loop.py` writes the CPU model,
core count and RAM into every `summary.json` for exactly this reason.

Three numbers you will be tempted to reuse and must not: **13.10 tok/s** (`gpu-jax-g5g-2b`, a
T4G), **43–44 tok/s** (the vLLM sibling on the same silicon, with reduced Triton tiles), and
anything from `~/gemma4-tips` (that tree duplicated its own artifacts and its directory names
misattribute both model and chip).
