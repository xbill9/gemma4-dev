# AGENTS.md — `gpu-pytorch-g5g-2b`

Guidance for coding agents working in this rig. **`CLAUDE.md` is authoritative where these
disagree**; there is no generator, so a convention change has to be applied to all three of
`CLAUDE.md`, `AGENTS.md` and `GEMINI.md` by hand.

## What this rig is

Serves **`google/gemma-4-E2B-it`** under **pure JAX** on **AWS EC2 G5g** — Graviton2
(aarch64) host, **NVIDIA T4G** GPU (Turing, SM 7.5, **15360 MiB measured**).

No PyTorch, no torch_xla, no vLLM. The engine is this repo's own port (`ports/gemma4/`) via
`torch_generate.py` behind `torch_openai_server.py`, under **systemd, not docker**. Read logs with
`journalctl`, never `docker logs`.

## Hard constraints

- **Turing has no bf16 and no fp8.** The device decides the compute dtype, not `tpu.env`:
  `jax_e_model.py` reads the live compute capability and picks `float16` below SM 8.0.
  bfloat16 *emulates* through fp32 (warning); fp8 is refused outright.
- **Never set `jax_default_matmul_precision="bfloat16"`** here. It is right on a TPU MXU and
  actively wrong on a chip with no bf16 unit. Only the TPU branch sets it.
- **The fused W4A16 Pallas kernel cannot run.** It needs 550 KiB – 1.1 MiB of shared memory
  per block against Turing's 64 KiB. It is refused at startup with the arithmetic attached.
- **`ports/gemma4/` is vendored**, shared with `tpu-jax-v5e1-2b`. `make lint` excludes it on
  purpose: ruff's UP006/UP045 would rewrite its `Dict`/`Optional` annotations and drift it
  from the sibling. Match the surrounding style; do not modernise it.

## Engineering rules

- boto3 and the standard credential chain — **never shell out to the AWS CLI** from
  `server.py`. SSM Run Command for remote admin; no inbound SSH rule, no private key.
- Every subprocess goes through `run_command(cmd: list[str])`. **Never `shell=True`.**
- MCP tools are `async def` returning markdown with emoji status prefixes (✅ ❌ 📡).
- Require explicit subnet, security-group and instance-profile ids. **Do not create broad
  network or IAM policy.**
- Scope instance discovery to `ManagedBy=gpu-pytorch-g5g-2b`.
- HF tokens live in Secrets Manager, fetched at boot into a root-only `EnvironmentFile`.
  **Never** in user data — instance metadata is readable by anything on the box.
- Launches default to spot. Surface capacity errors rather than silently retrying.
- **Never hardcode an endpoint or an AMI id.** The AMI must be arm64 *and* carry the NVIDIA
  driver; AWS ships driverless ARM64 DLAMIs that boot fine and have no GPU.

## Testing

`python3 -m unittest discover -s tests -v` — **unittest, never pytest**. 122 tests, fully
offline: no AWS, no network, no GPU. They pin the Turing dtype constraints, the AMI filter,
the host-RAM floor, the shared-memory ceiling, and that the token never reaches user data.

`make lint` runs ruff over a **hardcoded file list** plus `bash -n` on four shell scripts. A
new top-level module is silently unlinted until you add it to that list.

## Generated files — never hand-edit

`.claude/skills/**` and `skills/**` are generated copies; edit the source then `make skill`.
**Eight files** are snapshotted, including the whole serving payload, because an installed
skill still has to be able to run `deploy_torch_server`.

**`deploy_torch_server` ships the SKILL SNAPSHOT, not the working tree.** Always `make skill`
first. The deploy prints the payload root it resolved and a build id; `verify_model_health`
compares that id against the local payload and reports `STALE DEPLOY`.

## Measurement discipline

- **A config flag being accepted is not evidence it did anything.** Cross-check against an
  absolute physical bound, not another config.
- **Warm up at the shape you measure.** `max_new_tokens` is a `static_argnames` entry.
- **Do not health-check by testing for a non-empty response** — a broken deploy here returns
  fluent-looking garbage. `verify_model_health` reads the degenerate counter instead.
- Numbers from `gpu-vllm-g5g-2b` (43.1 / 44.24 tok/s) are a **different runtime** and were
  obtained with reduced Triton tiles. Never quote them as this rig's baseline.
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
