---
name: gpu-pytorch-g6-2b-management
description: Manage AWS EC2 G6 capacity (x86_64 + NVIDIA L4) and Gemma 4 E2B serving under stock PyTorch / HF transformers. Use when the user asks about provisioning, launching, listing, or terminating G6 instances, installing or debugging PyTorch on L4 / Ada / SM 8.9, G-family quotas, or the gpu-pytorch-g6-2b devops MCP agent. Triggers include "G6", "L4", "Ada", "SM 8.9", "PyTorch on GPU", "transformers serving", "g6.2xlarge".
---

# gpu-pytorch-g6-2b management

Provision and operate **EC2 G6** (x86_64 host + NVIDIA **L4** GPU) serving
`google/gemma-4-E2B-it` under **stock PyTorch + HF transformers**, through the
`gpu-pytorch-g6-2b` MCP server.

This is the **ordinary-runtime control** for this silicon: no custom model port, no XLA,
no compiled static shapes, no from-source build. `AutoModelForCausalLM` and an HF KV
cache. Its siblings on identical hardware are `gpu-jax-g6-2b` and `gpu-vllm-g6-2b`.

## Start here, every time

**Run `verify_gpu_arch` before anything else on a new instance.** It reports nvidia-smi's
view, torch's device and compute capability, the arch list, and one real matmul on the
device. A flag being accepted proves nothing; a kernel either launches or it does not.

## Order of operations

`create_g6_instance` → `get_install_progress` → `verify_gpu_arch` → `deploy_torch_server`
→ `get_torch_logs` → `verify_model_health`

Cloud-init installs the **runtime only** and then waits. The serving payload is this rig's
own source (`torch_openai_server.py`, `torch_generate.py`), so `deploy_torch_server` ships
it over SSM as a gzipped tarball — there is no published artifact to pull, and user data
could not carry it at a 16 KB limit.

**Always `make skill` before `deploy_torch_server`.** `server.py` resolves the payload
next to itself, and when the MCP server runs from `.claude/skills/…/mcp/` it ships the
*snapshot*, not your edit. The deploy reports success and the box runs stale code;
`verify_model_health` compares the served build id against the local digest and says
`STALE DEPLOY`.

## The dtype rule, and why the guard stays

**The device picks the dtype, not `tpu.env`.** `resolve_compute_dtype` reads the live
compute capability and selects **bfloat16 at SM ≥ 8.0**. The L4 is SM 8.9, so this rig
serves bf16 on a native datapath.

Keep the guard anyway. The failure it prevents is silent in exactly one direction:
**bfloat16 on a pre-Ampere GPU does not raise** — CUDA emulates it through fp32, output
stays correct, and most of decode vanishes into conversion. That is what the T4G siblings
measured at 54% of kernel time. `DTYPE` in `tpu.env` is only an override.

## What this rig does NOT have

There is **no quantization path at all**: no PLE table, no int8 LM head, no W4A16, no
`--quant-mode`. `torch_openai_server.py` defines exactly `--model`, `--host`, `--port`
and `--seq`.

`tpu.env` still carries inert keys (`QUANT_MODE`, `PLE_BITS`, `INT8_LM_HEAD`,
`PREFILL_CHUNK_SIZE`) because it was forked wholesale. **They do nothing here and must not
be re-plumbed into the serving command on the strength of existing** — argparse rejects an
unknown flag with exit code 2, so the unit would crash-loop under `Restart=on-failure`
with the reason only in `journalctl`.

**Concurrency is not an axis.** `MAX_NUM_SEQS=1`; the server decodes one sequence at a
time. Sweep context and output length.

## Comparing against the siblings — read this before quoting a number

This rig serves the **dense** checkpoint (~10.2 GB of weights). `gpu-jax-g6-2b` serves the
same model on the same chip at **6.155 GB** (`ple4 + int8_lm_head`). Decode on an L4 is
bandwidth-bound on exactly those bytes, so **a JAX-vs-PyTorch number here is not a
like-for-like runtime comparison** — roughly 40% of any gap is the weight footprint, not
the runtime. Say so whenever the two are put side by side.

Numbers that are **not** this rig's and must never be quoted as such:
- **10.66 tok/s** — `gpu-pytorch-g5g-2b`, same runtime on a **T4G**. The nearest control.
- **48.3–48.5 tok/s** — `gpu-jax-g6-2b`, same chip, different runtime *and* fewer bytes.
- **43–44 tok/s** — the vLLM sibling on **T4G**, with reduced Triton tiles.
- Anything from `~/gemma4-tips` or the `gpu-vllm-l4-*` artifact rigs. Same GPU is not the
  same measurement, and that tree misattributes both model and chip.

## Instance sizing

`g6.2xlarge` (8 vCPU, **32 GiB** host, 1 L4) is the default and the only size measured.

**G6 has twice the host RAM of G5g at the same suffix**, so never transfer a G5g verdict
onto the same G6 size name. `g6.xlarge` is 16 GiB here (not 8) and is the one size that
gets a swapfile; it has never been launched, so that path is untested code.

**`g6.16xlarge` is SINGLE-GPU despite the suffix** — the multi-GPU sizes are 12/24/48xlarge
(4, 4, 8 L4s). Nothing shards across them: the serving path is single-device, so a bigger
instance buys host RAM and vCPUs, **not** device memory. There is no `g6.metal`.

Device memory is **23034 MiB**, not the T4G's 15360.

## AMI resolution

`_resolve_ami` prefers the AWS public SSM parameter for the **x86_64 PyTorch** DLAMI, and
falls back to a name filter plus an explicit `architecture: x86_64` filter.

- **The arch is NOT in an x86_64 DLAMI's name.** Only ARM64 images announce theirs. A
  filter containing "x86_64" matches **zero** images — verified 2026-08-29.
- **`/latest/` in a parameter path is only the newest build within one PyTorch+Ubuntu
  line**, and AWS stops rebuilding old lines. Pinned to `pytorch-2.13-ubuntu-26.04`,
  which is live. Ubuntu 26.04 also ships **Python 3.14 as the system interpreter**, taking
  deadsnakes off the critical path.
- **The interpreter is probed, never hardcoded.** The DLAMI ships torch in a venv
  (`/opt/pytorch/bin/python`), not system-wide; installing transformers into
  `/usr/bin/python3.14` yields `ModuleNotFoundError: torch` *after* the install reports
  success. `TORCH_PYTHON_VERSION` only seeds candidate names.

**Never hardcode an AMI id.** torch comes from the DLAMI to keep a multi-GB download off a
spot host's critical path — on Ada that is a preference, not a requirement, since upstream
wheels carry `sm_89`. Do not re-derive the Turing rationale for it.

## Lifecycle guardrails

- Launches default to **spot**. Surface capacity errors; do not silently retry. G6 spot in
  us-east-1 has been priced at ~96–100% of on-demand, which signals tight capacity.
- Terminating is permanent but cheap to redo: no built image is lost, only a pip install
  and the model cache. One-time spot instances cannot be stopped, only terminated.
- Provisioning requires explicit **subnet, security-group, and instance-profile ids**. Do
  not create broad network or IAM policy to make a launch succeed.
- The HF token comes from Secrets Manager at boot (`save_hf_token`). **Never** put it in
  user data — instance metadata is readable by anything on the box. The bootstrap disables
  `xtrace` around the fetch, because `set -x` traces assignments with their values.
- Instance discovery is scoped to `ManagedBy=gpu-pytorch-g6-2b`. Never operate outside it.

## Diagnostics

- `verify_model_health` uses `/v1/chat/completions`. Raw `/v1/completions` skips the chat
  template and is unreliable on `-it` models. **Do not health-check by testing for a
  non-empty response** — a sibling returned `': ok: ok: ok…'`, which passes that test.
- `get_torch_logs` reads the **systemd journal**, not docker: nothing here is containerized.
  The unit is `torch-g6.service`.
- `get_install_progress` separates **cloud-init error** / **done-but-never-started** /
  **still-booting**. Cloud-init writes `install.sh` and backgrounds it, so a bootstrap that
  dies early leaves no install log — which looks identical to a slow install if the tool
  does not say otherwise. Stage timings: `grep -F '[stage]' /var/log/torch-install.log`.
- SSM truncates output at 24,000 characters. Truncation is **detected and reported**, not
  returned as if complete — a partial journal is how you conclude an error is not there.

## Measurement discipline

Quote the server's `tpu_jax_decode_tokens_per_second` gauge (`get_metrics`), not an
end-to-end rate: end-to-end carries prefill and the HTTP round trip, so it falls with
context while decode does not. The `tpu_jax_` prefix is an identifier, not a description;
series carry a `rig` label.

**Warm up at the shape you measure.** The first request at a given shape pays allocator
growth and autotune.

Cross-check against a physical bound — **300 GB/s of GDDR6 and 23034 MiB is the whole
envelope here** — not against another config. Record results in
`benchmarks/runs/<date>-<what>-g6/`; the `-g6` suffix is the hardware measured.
