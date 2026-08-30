---
name: local-jax-cpu-2b-management
description: Serve Gemma 4 E2B under pure JAX on the local CPU — no accelerator, no cloud, no provisioning. Use when the user asks about running Gemma 4 or JAX locally, serving a model on CPU, starting/stopping the local JAX server, whether a model fits in host RAM, or the local-jax-cpu-2b MCP agent. Triggers include "local CPU", "run it locally", "no GPU", "CPU inference", "JAX on CPU", "does it fit in RAM", "swap".
---

# local-jax-cpu-2b management

Serve `google/gemma-4-E2B-it` under **pure JAX on the local CPU**, through the
`local-jax-cpu-2b` MCP server. Same engine as every JAX rig in this monorepo
(`ports/gemma4/` behind `jax_engine.py` and an OpenAI-compatible FastAPI server), running as
an ordinary background process owned by the invoking user.

**There is no control plane.** Nothing is provisioned, nothing is billed, and no API can
refuse. Every sibling rig's server.py is mostly cloud lifecycle; that is all deleted here.
Do not look for `create_*_instance`, quotas, AMIs, spot, SSM or Secrets Manager — none of
them exist, and none is owed.

## What has been measured

**Nothing yet.** `benchmarks/runs/` is empty on purpose. Every throughput number in this
rig's inherited prose (12.4 / 12.8 / 13.10 tok/s) was measured by `gpu-jax-g5g-2b` on a
**Turing GPU**, and none of it is this rig's. Do not difference a number from here against
one of those and read it as a CPU-vs-GPU result.

The host facts recorded in `tpu.env` and `CLAUDE.md` — RAM, cores, the btrfs swap incident,
the anonymous-checkpoint verification — were measured here on 2026-08-29.

## Start here, every time

**Run `check_host_capacity` before starting a serve.** It is the analogue of the cloud rigs'
quota check and it exists because the failure mode is worse than theirs: exceeding a cloud
quota is *refused at the API*, and exceeding host RAM is **accepted and paid for in swap**.
A thrashing serve is indistinguishable from a loading one, so the arithmetic has to happen
before the load, not after.

Then `verify_cpu_backend`, which runs a real matmul on the device rather than checking that a
flag was accepted, and asserts the device really is a `CpuDevice`. That last part matters
here: this machine also hosts the GPU sibling rigs, so a CUDA plugin in site-packages would
silently hand this rig a GPU and every number it produced would be mislabelled.
`JAX_PLATFORMS=cpu` is set for exactly that reason.

## Memory is the constraint, not speed

It will be slow. That is expected and it is not the interesting part. What will actually stop
a serve is RAM.

| Config | Weights |
| --- | ---: |
| dense (`ple_bits=0`) | 9.257 GB |
| `ple_bits=4` — **this rig's default** | 5.752 GB |
| `ple_bits=4` + `int8_lm_head` | 6.155 GB |

Plus roughly **1.6 GB** of prefill transient at a bucket below 4K, which is flat rather than
per-token in that range.

**`INT8_LM_HEAD` is OFF here, inverting every GPU sibling's default.** It *adds* 0.403 GB to
buy +2.3% throughput by halving the bytes read per decode step — a memory-bandwidth trade,
and the wrong one when the constraint is resident bytes and the surplus goes to swap.

**Never size this rig's context from KV arithmetic.** With `window_kv` on, only 3 of E2B's 35
layers hold full context and the whole KV cache at 2048 tokens is about 18 MiB — roughly
ninety times smaller than the prefill transient.

## The dtype rule, and why it inverts the GPU siblings'

- **`bfloat16`, not `float16`** — and this is forced by memory, not chosen for speed.
  XLA:CPU has no bf16 datapath and upconverts to fp32 in front of every use, the same tax the
  Turing rigs pay. The difference is that **float16 is not an escape here**: a CPU has no
  16-bit float datapath of any kind, so fp16 is upconverted identically. float32 storage would
  avoid the conversion and needs 18.5 GB against 14.3 GiB of RAM.
- The port picks the dtype from the live device. `DTYPE` in `tpu.env` is the override and
  `JAX_E_COMPUTE_DTYPE` is the escape hatch.
- **`--quant-mode` matches the checkpoint, not the host.** `fp16` for the dense reference
  build, `w4a16` only for a `-w4a16-` export. Getting this wrong loads garbage rather than
  failing.
- **Watch for `pallas_interpret=True` in the startup banner.** Pallas has no CPU backend, so
  the fused W4A16 kernel auto-runs in *interpret mode* — a simulator. It produces correct
  numbers at a speed that means nothing. On the GPU siblings the same kernel is refused
  outright; here it silently runs slowly, which is the more dangerous failure.

## Order of operations

`check_host_capacity` → `check_dependencies` → `verify_cpu_backend` → `fetch_checkpoint` →
`start_jax_server` → `get_jax_logs` → `verify_model_health` → `query_model` / `get_metrics`

`fetch_checkpoint` is worth running separately: otherwise the first start spends ten minutes
in a stage with no output and there is no way to tell a slow download from a hung one.

## Diagnostics

- `get_jax_logs` tails **one logfile**, not a systemd journal and not docker — this runs as
  the invoking user with no root and no unit. It carries the device-policy banner, the staged
  load timings, one `key=value` line per request, and the READY line with the whole resolved
  configuration.
- `verify_model_health` uses `/v1/chat/completions` and reads
  `tpu_jax_degenerate_responses_total` either side of its own probe. **Do not health-check by
  testing for a non-empty response** — a token loop is reported as `status="success"`.
- `get_metrics` keeps the `tpu_jax_` series prefix on a rig with no TPU, deliberately: both
  existing benchmark reports compare on `tpu_jax_decode_tokens_per_second` **by name**, and
  the `rig` label is what separates rigs. `tpu_jax_hbm_used_bytes` is **absent** here rather
  than zero, because a CPU device has no allocator to ask; read `tpu_jax_host_rss_bytes`.
- `list_jax_servers` reports process RSS, which on this rig is the real memory story.
- The build id on `/health` still means something: with no deploy step, a mismatch says the
  running process is executing a **different copy** of the payload — the skill snapshot rather
  than the working tree, or the tree as it was before your edits.

## Guardrails

- **Binds loopback by default.** This server has no authentication of any kind and this host
  is not behind a cloud security group. Do not set `JAX_HOST=0.0.0.0` without deciding to.
- **Stopping is genuinely cheap.** Nothing was built; the warm compilations live in the XLA
  disk cache, which unlike every cloud sibling's is **not** ephemeral. Do not import the
  EC2 rigs' "weigh stop against terminate" reasoning.
- `start_jax_server` **refuses** to start when the model cannot fit in RAM plus swap, because
  starting anyway would not fail cleanly — it would thrash.

## Measurement discipline

A config flag being accepted is not evidence it did anything. Cross-check against a physical
bound, warm up at the shape you intend to measure (`max_new_tokens` is a `static_argnames`
entry, so `(bucket, max_tokens)` **is** the compiled shape), and record results in
`benchmarks/runs/<date>-<what>-cpu/`.

**A report from this rig must record the host in its own fields.** Unlike every sibling,
whose hardware slot names a specific SKU, `cpu` names whatever machine the rig is checked out
on — so two runs of this rig on two machines are not comparable and the rig name cannot carry
the difference.
