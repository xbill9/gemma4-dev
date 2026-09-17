# CLAUDE.md — gpu-vllm-mi300x-2b

Guidance for Claude Code when working inside this rig. Read `../CLAUDE.md` first for the
monorepo rules; this file covers what is different here, and where it contradicts a sibling,
**this file wins inside this directory**.

## What this rig is

One AMD Instinct MI300X serving `google/gemma-4-E2B-it` through vLLM in Docker, on a
DigitalOcean GPU droplet reached through AMD Developer Cloud (`devcloud.amd.com`) — same v2
API, same droplet ids, token from the **My AMD Team** account, not a personal one.

**This is the only AMD rig in the monorepo and the only one whose control plane is not Google
Cloud or EC2.** Every sibling's `gcloud` reflex is wrong here. The closest relative is
`~/amd-gputools`, which is a different repo with a different job: it is general MI300 tooling,
this is one serving rig.

## There is no GPU on the machine you are running on

`rocm-smi`, `amd-smi`, `rocminfo` and `hipcc` do not exist locally and never will. **A ROCm
command in a local shell is a bug, not a check.** Reach the hardware through the MCP tools
(`gpu_status`, `run_on_droplet`) or explicit `ssh`. Nothing in `server.py` may assume a local
device.

**Powering a droplet off does not stop DigitalOcean billing it** — the resources stay reserved
and the hourly rate keeps running. Only destroying it stops the meter, and this rig
deliberately has no tool that destroys one. `create` and `destroy` are absent on purpose:
both are dollar-per-hour decisions that stay a deliberate human step in the console.

## The image caveat is the most important fact in this rig

`rocm/vllm:rocm10.0.0_..._vllm_0.27.0` — the newest image AMD publishes — **cannot load Gemma
4.** It raises `AmbiguousGlobalPerLayerAttributeError` on `head_dim` during config parsing,
before the GPU is touched, because it lacks `Gemma4ModelArchConfigConvertor`. Gemma 4 uses
256-wide heads on sliding-attention layers and 512 on full-attention ones; transformers ≥ 5.15
reports that honestly and raises on a global read, and vLLM's `getattr(..., 0)` default cannot
catch it.

**Never change `VLLM_IMAGE` without running `check_image` first.** Newer is not safer here —
the working images are the nightly and the *oldest* vendor build, and the broken one sits
between them. The pip channel vLLM's own recipe names is staler than all three.

## Do not add an audio flag

E2B has a conformer audio encoder and it is unreachable: no ROCm vLLM image ships the
`vllm[audio]` extras. `librosa` and `soundfile` are absent from all three images tried, so
audio requests fail at request time whatever `--limit-mm-per-prompt` says. Setting `audio` to
anything but `0` allocates encoder memory for a path that cannot be used and makes the failure
later and more confusing. Audio needs a derived image; that is a real change, not a flag.

## Quantization: fp8 only, and it has to be `fnuz`

Measured on the part 2026-09-16 — `../HARDWARE.md` carries the full table. The short version:
**fp8 is 1.77x bf16 and is the only format on this card faster than bf16.** int8 measured
*slower* (0.69x) despite an equal spec peak, fp4 does not exist on gfx942, and GGUF is compiled
out of the image entirely.

The serving default stays bf16 — the engine reports `dtype=torch.bfloat16, quantization=None`,
matching the checkpoint's own `dtype: bfloat16`. To turn fp8 on, add `--quantization fp8` and let
vLLM derive the scales from the bf16 weights at load.

**Do not pull an fp8 checkpoint from the Hub to do that.** CDNA 3 uses `e4m3fnuz`; every fp8
checkpoint published for H100 is `e4m3fn`, and that dtype does not quietly fall back here — it
raises `HIPBLAS_STATUS_NOT_SUPPORTED`. Online quantization from bf16 sidesteps the question.
This is the same shape of trap as the image caveat above: the plausible artifact is the broken one.

Not yet measured, and worth keeping honest about: **end-to-end tokens/sec at fp8.** The 1.77x is a
GEMM ratio, and E2B spends much of decode in attention and kernel-launch overhead that fp8 does not
touch, so it will not carry over whole. `VLLM_ROCM_USE_AITER=1` is a second untested lever — the
server currently selects `TRITON_ATTN`.

## Code style

Matches the siblings, and `ruff.toml` is pinned here for the same reason `amd-gputools` pins
one — ruff's implicit defaults drift.

- Every subprocess call goes through `run_command(cmd: list[str])` using
  `asyncio.create_subprocess_exec`. **Never `shell=True`.** Remote commands are handed to the
  *remote* shell as one argument; the local side never invokes a shell.
- MCP tools are `async def` returning markdown strings with emoji status prefixes (`✅`, `❌`,
  `📡`), and every one ends in a blind `except Exception` returning `_error(exc)`. That is the
  design, not an oversight: an exception escaping a tool kills the server process for every
  later call. `BLE001` is disabled for this.
- `Optional[str]`, not `X | None`. `UP045` is disabled for this.
- Don't assume `pandas`; prefer stdlib `csv`/`json`.

## Testing

`make test` — `unittest`, never pytest. The whole `mcp` package is mocked **before** `server`
is imported, so nothing reaches the DigitalOcean API, opens SSH, or needs a token.

The fake `MCPServer` must implement `tool()` as a pass-through decorator. A bare `MagicMock`
makes `@mcp.tool()` return a MagicMock instead of the decorated coroutine, and every tool test
then fails with "'MagicMock' object can't be awaited" — which reads as a broken server and is
really a broken fake.

Anything calling `mcp.list_tools()` needs an explicit `AsyncMock` patch.

On Debian 13 the dependencies install as `python3-httpx` and `python3-dotenv` through apt;
`pip install` into the system python3 is refused by PEP 668 without
`--break-system-packages`. **Still never create a virtualenv** — the constraint is the venv,
not the install path.

## Generated files — never hand-edit

`.claude/skills/**` and `skills/**` are generated copies of the rig-root sources. Edit the
source, then `make skill`. Hand-edits are lost.

`.mcp.json` is gitignored — it is generated by `project-setup.sh`. Never commit it.

`tpu.env` is committed and deliberately **not** gitignored; it holds identifiers, never
secrets. The DigitalOcean token lives in `.env` (gitignored, mode 0600) or the environment,
as `DIGITALOCEAN_ACCESS_TOKEN`.

## Measurement

First sweep landed 2026-09-16: `benchmarks/reports/2026-09-16-vllm-sweep-mi300x.json`, 12 cells
of a 4x4 concurrency-by-context grid, 3 repeats each, worst-cell cv 8.4%. **Do not let a figure
from this rig be differenced against a TPU rig and read as a control-plane result**: nothing else
in the monorepo serves this checkpoint on AMD, so there is no twin to difference against.

**Benchmarking this server measures its prefix cache unless you stop it.** `vllm bench serve`
derives its prompts from `--seed`, which defaults to 0, and `enable_prefix_caching=True` here. Two
runs sharing a seed generate identical prompts, so the second reads the first's cache — worth
**2.25x at 8192 context** and 1.00x at 128, which is the gradient a prefill-saving cache predicts.
Cells at one context length draw from the same pool, so per-repeat seeding is not enough either.
`benchmarking_suite.py` gives every cell and repeat a seed no other run uses; `--seed-base` shifts
a whole sweep off an earlier one. The contaminated sweep is kept as
`...-seedcollision` with `prefix-cache-effect.txt` beside it, because the difference between the
two runs is the measurement. Its worst-cell cv was 51.1% against 8.4% clean.

**The KV pool is never this workload's constraint.** The engine allocates 155.04 GiB / 9,026,017
tokens and the heaviest cell in the grid wants 524,288 — 5.8%. Long-context throughput flattens on
prefill, not memory, which inverts the TPU siblings' sizing rule. The same figures put KV at
18,443.7 B/token, matching `../MODELS.md`'s geometry derivation to 0.06% and refuting the
engine-policy hypothesis its open-discrepancy block proposed; that block now records this run.

**`run_vllm_benchmark` runs the load generator in a container with no GPU device attached.** It
cannot touch the card, which is what makes it safe beside a live server, and `--entrypoint vllm` is
set unconditionally rather than branching on the image the way `_serve_argv` does.

A config flag being accepted is not evidence it did anything. `verify_capabilities` probes
each modality with a request whose correct answer is known in advance, which is why the vision
probe is a generated checkerboard rather than a fetched photo.
