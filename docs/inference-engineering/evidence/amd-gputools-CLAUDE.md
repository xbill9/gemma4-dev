# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

amd-gputools is a Python + shell toolkit for AMD Instinct MI300-class accelerators,
plus an MCP server (`server.py`) that drives the DigitalOcean droplets carrying them.

## There is no GPU on this machine

The workstation has no AMD GPU. `rocm-smi`, `amd-smi`, `rocminfo`, `hipcc` and
anything else from ROCm **do not exist locally and never will**. The hardware is a
DigitalOcean GPU droplet.

- Never run a ROCm command directly, and never write code that assumes a local
  device. A ROCm command in a local shell is a bug, not a check.
- Reach the hardware through the MCP server's tools (`mcp__amd-gputools__gpu_status`,
  `run_on_droplet`, `ssh_command`) or through explicit `ssh`.
- **Powering a droplet off does not stop DigitalOcean billing it** — the resources
  stay reserved and the hourly rate keeps running. Only destroying it stops the
  meter, and this repo deliberately has no tool that destroys one. `create_droplet`
  exists and starts that meter, which is why it orders nothing until a second call
  passes `confirm=true`.

## The droplet

Reached through AMD Developer Cloud (`devcloud.amd.com`), which is DigitalOcean
underneath — same v2 API, same droplet ids. The token comes from the **My AMD Team**
account, not a personal DigitalOcean one.

| | |
| --- | --- |
| name | `debian-gpu-mi300x1-192gb-devcloud-atl1` |
| id | `601744248` — **the fourth id this box has had; never hardcode one** |
| tag | **`gemma`** — this is what `DROPLET_TAG` must be, not `amd-gputools` |
| size | `gpu-mi300x1-192gb-devcloud` — 1× MI300X, 192 GiB VRAM, 20 vCPU, 240 GB RAM |
| region | `atl1` |
| cost | **$1.99/hour** — $48/day, $1,433 for 30 days, running now |
| OS | Debian 13 (trixie), kernel 6.12.94+deb13-amd64, Xeon Platinum 8568Y+ |

Created 2026-09-18, replacing `601418522` (itself created 2026-09-17), which was gone by then
like its predecessor. `601418522` replaced id `601142018`, which was **destroyed** — not powered
off — some time before that date. The old id returned 404 and the account held zero
droplets. Those are also the defaults `create_droplet` uses, and they live in
`amd.env` (`DROPLET_SIZE`, `DROPLET_REGION`, `DROPLET_IMAGE`, `DROPLET_SSH_KEYS`).
Re-read the id and the address rather than trusting this table — every rebuild
changes both.

### Creating one

`create_droplet` is two-step on purpose: the first call orders **nothing** and reports
the size, region, image and keys it would use, what the catalogue charges per hour and
per month, and whether the API even offers that size in that region. Call it again with
`confirm=true` to place the order. It applies `DROPLET_TAG` unconditionally — an
untagged droplet would be a machine this server pays for and cannot reach — and it
refuses by default if anything is already tagged, because a second GPU droplet doubles
the hourly bill.

**The v2 API cannot sell you this card, and that is settled, not suspected.**
`POST /v2/droplets` with the devcloud slug returns **`422: This size is unavailable`**
— measured 2026-09-17. It is not a token, tag or region typo: no MI300X is orderable
through the public API at all. `gpu-mi300x1-192gb` and `gpu-mi300x8-1536gb` are both
`available: true` with an **empty `regions` list**, and across every available region
there are exactly six orderable GPU size/region pairs, none of them MI300 (`mi325x1` in
nyc2/tor1 at $3.80/hr, `mi350x1-spot` in ric1, `mi355x1-spot` in mem1, plus the x8
variants).

**The console allocates from a fleet the API has no field for.** The devcloud create URL
carries `fleetUuid=28e7a619-ddf9-4ef8-99bc-46b38b871fe7` (which is also the "My AMD
Home" project id), and there is no fleet route on `api.digitalocean.com` — `/v2/gpus`,
`/v2/gpus/fleets`, `/v2/fleets` and every variant tried return `not_found: Your request
could not be routed`. So the asymmetry is in the **create path only**: once the droplet
exists it is an ordinary v2 droplet that `list_droplets`, `start_droplet` and the rest
drive normally.

So `create_droplet` is the preflight and the audit trail, not the way this box gets
built. Build it at:

```
https://devcloud.amd.com/gpus/new?i=95798a&region=atl1&size=gpu-mi300x1-192gb-devcloud&fleetUuid=28e7a619-ddf9-4ef8-99bc-46b38b871fe7&options=install_agent&distroImage=debian-13-x64&distro=debian
```

That prefills plan, region, image and hostname. Three things still have to be set by
hand, and the tag is the one that matters: tick the **`amd` SSH key** (the button stays
disabled until a key is selected), and expand **"Show details and additional options"**
to reach the Tags field and add **`gemma`**. An untagged droplet is invisible to every
tool here.

#### Quirks of that form, all measured 2026-09-17 while driving it

- **The Tags field is hidden by default.** It is below "Show details and additional
  options", past region, VPC, backups, volumes, networking, monitoring and startup
  scripts. Submitting without expanding that section produces an untagged droplet that
  every tool in this repo is blind to — the single most expensive mistake available on
  this page, because you pay for it and cannot see it.
- **Setting the SSH-key checkbox programmatically does not take.** Assigning `checked`
  ticks the box visually while the counter stays at `0 / 1` and the submit button stays
  disabled: React never sees the change. A real click is required, and the counter going
  to `1 / 1` is the thing to verify, not the tick.
- **Clicking the submit button by coordinate silently did nothing** — the page went
  blank and `Page.captureScreenshot` timed out — while clicking the same button by
  element reference worked immediately. Prefer the element reference.
- **The first click on "Create GPU Droplet" after adding a tag can do nothing.** Measured
  2026-09-18: a click by element reference, right after typing the tag and pressing Enter,
  left the URL unchanged, raised no error, and the account held zero droplets 20 seconds
  later. The identical second click created it at once. **Confirm the account is empty
  through the API before clicking again**, because a second click that does work orders a
  second $1.99/hour droplet.
- **The credits are time-limited and partial.** The form said $88.40 of AMD GPU credit
  **expiring 2026-10-15**, and that credit covers GPU access only; everything else on
  the account bills to the payment method.
- **The console's name rule is stricter than the API's.** It demands lowercase, 3–45
  characters, dashes only. `create_droplet` validates against the API rule instead
  (letters, digits, dots and dashes, no leading or trailing dash), so a name it accepts
  can still be one the console would have rejected.
- **Spot is offered in ATL1, MEM1, RIC1 and MKC1.** MKC1 appears nowhere in
  `GET /v2/regions`, which is another instance of the console knowing about capacity the
  public API does not.

### A freshly provisioned droplet needs one reboot — and `/dev/kfd` does not prove otherwise

Verified working 2026-09-16 on the previous droplet: `gfx942`, AMD Instinct MI300X VF,
304 CUs, 191.7 GiB VRAM, ISA `amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-`.

**`amdgpu` fails to bind the MI300X VF on a stock droplet, every time so far — and
"just reboot" is only half of it.** Measured 2026-09-17 on `601418522`, where the
failure changed shape between boots:

| boot | dmesg | cause |
| --- | --- | --- |
| first | `Doesn't get msg:1 from pf, error=-62`, stack through `amdgpu_pci_probe` | PF handshake |
| after one reboot | `failed to load amdgpu/gc_9_4_3_rlc.bin (-2)`, `early_init of IP block <gfx_v9_4_3> failed -19` | **no firmware installed** |
| after firmware + reboot | clean | works: `gfx942`, 1 agent, 191.7 GiB |

**`/lib/firmware/amdgpu` is EMPTY on the stock image — 0 files.** `-2` is ENOENT: the
blobs simply are not there. They ship in **`firmware-amd-graphics`, which lives in
`non-free-firmware`**, so on a stock droplet the apt component has to be enabled before
the fix is even installable. That is why `scaffold/remote-prepare.sh` treats the
components as a prerequisite rather than a convenience, and why it installs the firmware
from **backports** (`20260810-1~bpo13+1`) rather than trixie's `20250410-2`: gfx942 is
new enough that the eight-month gap is a real risk. After installing it, 552 blobs.

An earlier version of this file said "nothing needs installing". That was true of the
previous droplet, which already had the firmware; it is not true of a fresh one.

**What that looks like is not constant, and `/dev/kfd` is a bad test for it.** On
`601142018` the node was absent. On `601418522` it was **present, with `amdgpu` in
`lsmod`, and the card was still unusable** — `rocminfo` listed only the Xeon, `rocm-smi`
said "No AMD GPUs specified", and dmesg showed the probe dying:

```
amdgpu 0000:83:00.0: amdgpu: Doesn't get msg:1 from pf, error=-62
 amdgpu_virt_fini_data_exchange.cold  ← in amdgpu_pci_probe → amdgpu_device_fini_hw
```

`/dev/kfd` and `/dev/dri/renderD128` are created before that probe fails, so their
existence means the driver core got that far and nothing more. A first pass at this
file read "`/dev/kfd` present, `amdgpu` loaded" off `hardware_scan` and concluded no
reboot was needed; the card had not bound at all.

**The test that works is whether `rocminfo` reports a `gfx` agent.** `hardware_scan`
says so directly — it flags a card on the PCI bus with zero GPU agents as *did not
bind* — and `scaffold/remote-prepare.sh` goes one better by also counting
`/lib/firmware/amdgpu`, so its verdict distinguishes "no firmware installed" from
"firmware present, needs a reboot". Those need different fixes and look identical from
`/dev/kfd`. The benign-looking dmesg lines further down are genuinely benign;
`Doesn't get msg:1 from pf` and `gc_9_4_3_rlc.bin (-2)` are not, they are the two
failures above.

### Re-scaffolding: `scaffold_droplet` or `./scaffold-droplet.sh`

Two front ends, one behaviour. **`mcp__amd-gputools__scaffold_droplet` is the one call
that takes a bare droplet to a working GPU**: it prepares the box, reads the verdict,
reboots if the card needs it, waits for the droplet to come back and checks again.
`prepare_droplet` is only its first half — it installs everything and hands back a
droplet whose card has still not bound, which is where this went wrong the first time.
Both are idempotent; against a healthy droplet `scaffold_droplet` reboots nothing,
pulls nothing and reports the card as already bound, so it works as a health check.

It also declines to pull 35-62 GB onto a box whose GPU never came up.

The shell equivalent, for when the MCP server is not loaded:

```
./scaffold-droplet.sh              # apt, firmware, docker, reboot if needed, pull the image
./scaffold-droplet.sh --no-pull    # skip the 35-62 GB image
./scaffold-droplet.sh --probe      # also ask the image whether it can load Gemma 4
make scaffold                      # the same thing
```

**The remote half lives in `scaffold/remote-prepare.sh` and the Gemma 4 check in
`scaffold/gemma4-probe.py`. `server.py` reads those same two files** for the MCP tools
`prepare_droplet` and `vllm_image_status`, rather than keeping its own copy — they were
two copies of the same shell for about an hour, which is exactly how the copies drift.
A test enforces that the driver script does not grow its own version of the apt edit.

**The stock image carries no ROCm userspace at all.** Measured on `601418522`: the
only tool of the eleven probed that exists is `python3` (3.13.5). `rocm-smi`,
`rocminfo`, `amd-smi`, `hipcc`, `clinfo`, `docker`, `podman`, `pip3`, `git` and `tmux`
are all absent, there is no `/opt/rocm`, and `dpkg` lists no ROCm packages. The kernel
side is complete and the userspace is empty. The `rocm-smi` and `rocminfo` in
`/usr/bin` on the old droplet were therefore **installed at some point, not shipped**,
so `gpu_status` returns "command not found" on a brand-new box and that is the image,
not a fault. Docker has to be installed before `vllm/docker-serve.sh` can run.

These dmesg lines are benign on a VF and are not the problem: `failed to load
amdgpu/psp_13_0_6_cap.bin (-2)`, `Unsupported TA type: 8`, `TMZ feature not
supported`. **`Doesn't get msg:1 from pf, error=-62` is a different thing entirely** —
that one is the bind failure above, and it appears alongside a stack trace through
`amdgpu_pci_probe`.

Once installed, ROCm here is the Debian packaging (`rocm-smi`, `rocminfo` in
`/usr/bin`, ROCm 6.1.2), not AMD's own repo — there is no `/opt/rocm` and no
`amdgpu-dkms`. PyTorch is not installed either.

### Two tools here have reported confident, wrong answers

Both were found on 2026-09-17 and both are fixed, but the shape of the mistake is worth
keeping, because it is the shape this codebase keeps producing: a probe that did not run
being read as a probe that answered.

- **`gpu_status` diagnosed a machine nothing had spoken to.** Seconds after creation it
  reported "`/dev/kfd` exists, so the driver is up" about a droplet that was answering
  `ssh: connect to host … port 22: Connection refused`. The API had called it `active`,
  which says nothing about sshd; none of the probes had run; and the code treated "not
  absent" as "present". It now checks whether ssh ran at all before saying anything
  about the GPU, and the kfd probe answers three ways — present, absent, or unknown.
  **A droplet reporting `active` is not a droplet you can reach.**
- **`hardware_scan` filed `whiptail` under "ROCm packages"** on a box with no ROCm,
  because the awk filter matched a bare `hip` against the whole `dpkg -l` line. It now
  matches the package name against names that actually exist. Substring matching on
  three-letter tokens finds things that are not there.

**`rocm-smi` and `amd-smi` exit 0 when they fail.** Measured on the previous droplet:
`rocm-smi` printed "Driver not initialized" to stderr, printed nothing to stdout, and
exited 0. Never branch on their exit status — parse the output. `gpu_status` does,
after an earlier version reported a healthy "✅" with an empty table.

**The VRAM percentage key is `GPU Memory Allocated (VRAM%)`.** Not "GPU Memory Use (%)" or
any other plausible spelling — measured on ROCM-SMI 2.2.0 here 2026-09-16, while vLLM held
168.3 GiB of the 191.7 GiB. `gpu_status` printed `-` in that column for hours and the card
read as idle. The unit test used the invented key too, so it passed against a payload no
machine emits: fixtures for remote tooling get pasted from the box, never typed from memory.

The devcloud size slug is not in `GET /v2/sizes`, which lists the public
`gpu-mi300x1-192gb` at $2.59/hr instead. `list_gpu_sizes` shows the public catalogue,
not what devcloud sells.

## DigitalOcean ships its own MCP server, and it does not replace this one

`@digitalocean/mcp` (1.0.70 on npm, checked 2026-09-16) and the hosted endpoint at
`https://droplets.mcp.digitalocean.com/mcp` expose 40 droplet tools across 24 service
areas. Eight of this server's thirteen have a direct equivalent there: `droplet-list`,
`droplet-get`, `droplet-create`, `power-on-droplet`, `power-off-droplet`, `droplet-reboot`,
`droplet-action` and `size-list`.

**It is an API client, so it stops at the droplet object.** The repo tree has no `ssh`,
`exec`, `console`, `command` or `remote` tooling of any kind, which leaves `ssh_command`,
`run_on_droplet`, `gpu_status` and `hardware_scan` with no equivalent — and those are the
four that see inside the guest. `droplet-get` reports `active` on a fresh droplet whose
`amdgpu` has failed to bind and has no `/dev/kfd`, because that difference is not in the
droplet object. Its `size-list` reads the same `GET /v2/sizes` that omits the devcloud
slug, so it quotes $2.59 for a $1.99 card.

Two design differences are deliberate here and should stay that way. The official server
has `droplet-delete`; this one does not, and should not. And a tag there is a selector
for bulk actions (`power-off-droplets-tag`); here it is a boundary, so an untagged
droplet is not addressable at all — which is also why `create_droplet` applies the tag
itself rather than taking it as an argument, where `droplet-create` takes a free-form
tag list and will happily build something no tool can find again.

Do not reimplement the account, image, volume or fleet tools. If those are needed, add the
official server alongside this one.

## Commands

- `make lint` — `ruff format --check .` then `ruff check .`
- `make test` — `python3 -m unittest discover -s tests -v`
- `make check` — both; run this before committing
- `make install` — `pip install -r requirements.txt` into the system python3
- `make ssh` / `./ssh-droplet.sh [command]` — shell on the droplet, or run one command
  there. It resolves the address from the API on every call; pass `DROPLET_IP` to skip
  that. The token comes from `$DIGITALOCEAN_ACCESS_TOKEN`, then `.env`, then `~/ocean.txt`.

## Environment

Use the **system `python3`** and install into it. **Never create a virtualenv** —
`.mcp.json` launches the server with a bare `python3`, so a venv makes the server
work in your shell and fail in the client. Add new dependencies to
`requirements.txt`.

`amd.env` is the source of truth for configuration. It is committed on purpose and
must never be gitignored. **It holds no secrets.** The DigitalOcean token is
`DIGITALOCEAN_ACCESS_TOKEN` in the environment, then `.env` (gitignored, mode 0600),
then `~/ocean.txt` — the same order in `server.py` and `ssh-droplet.sh`, so a token
that works for one works for the other. A real environment variable always wins
over `amd.env`.

## MCP server conventions

`server.py` follows the sibling servers in `~/gemma4-dev`. Match them:

- **mcp 2.x, not 1.x**: `from mcp.server.mcpserver import MCPServer`. `mcp.server.fastmcp`
  does not exist in the installed SDK; 2.x renamed FastMCP to MCPServer.
- Tools are `async def ... -> str` returning **markdown with an emoji prefix** —
  `✅` success, `❌` error, `📡` in progress, `🛑` stopping. Never a dict, never TextContent.
- **Errors are returned, not raised.** Every tool ends in `except Exception as exc:
  return _error(exc)`. An exception escaping a tool kills the server process for
  every later call. A test enforces this.
- Every subprocess call goes through `run_command(cmd: list[str])` with
  `asyncio.create_subprocess_exec`. **Never `shell=True`.**
- Droplet lookups are scoped by `DROPLET_TAG`, so the server cannot touch a droplet
  nobody tagged for it. Keep new tools inside that scope.
- Never hardcode a droplet IP — resolve it per call. A rebuilt droplet gets a new one.
- The key in `.mcp.json` must equal the directory name: it prefixes every tool as
  `mcp__amd-gputools__<tool>`.

## Code style

- `Optional[str]`, not `str | None` — matches the sibling servers.
- `ruff.toml` pins the rule set deliberately. Do not delete it and do not "fix"
  `UP045` or `BLE001`: ruff's implicit defaults drift between releases and flag the
  `Optional[]` style and the blind `except Exception` that this codebase uses on
  purpose. Lint through `make lint` so the config is picked up.
- Tests are **`unittest`, never pytest**, and run offline: `tests/test_server.py`
  mocks the whole `mcp` package before importing `server`, so no test needs a token
  or a network. Keep it that way.
- Push counting, filtering and totalling into the code and have the caller quote the
  result — `list_droplets` and `_summarize_rocm_smi` compute their own tallies rather
  than printing rows for a reader to add up.

## Serving with vLLM

Two paths, both in `vllm/`, both driven from `amd.env`:

- **`vllm/docker-serve.sh`** — AMD's `rocm/vllm` image. This is the supported path and
  the one to reach for first. The image carries its own ROCm userspace; only the
  kernel `amdgpu` and `/dev/kfd` come from the host.
- **`vllm/baremetal-install.sh`** — PyTorch ROCm wheels into the system python3.

**Bare metal gets PyTorch, not vLLM, and that is not a shortcut.** There is no prebuilt
vLLM wheel for ROCm anywhere: PyPI's `vllm` wheels are CUDA builds, AMD's manylinux
index at `repo.radeon.com` carries torch and triton but no vllm, and the
per-architecture nightly index is not published (all checked 2026-09-16). Building from
source needs `hipcc` plus `rocblas`, `hipblaslt`, `miopen` and `rccl`; Debian 13 ships
`hipcc` 5.7.1 and none of those libraries, and AMD's repo does not support trixie.
vLLM therefore comes from the container. Getting it bare metal means either moving the
droplet to Ubuntu 24.04 or installing AMD's ROCm on an unsupported distro — a decision,
not a missing step.

The torch ROCm wheel bundles its own ROCm runtime libraries in `torch/lib`, which is why
bare metal works with no `/opt/rocm` at all. Install with `pip --break-system-packages`;
Debian 13 marks the system python externally-managed (PEP 668) and the thing it steers
you toward is a virtualenv, which this project does not use.

Verified 2026-09-16: `torch 2.9.1+rocm6.4` (HIP 6.4.43484) on the system python3,
**557.7 TFLOP/s** on an 8192³ fp16 matmul. `numpy` is not installed, so torch prints a
"Failed to initialize NumPy" warning — harmless for this, worth installing before real
work.

Pick the image tag by GPU architecture: `gfx94X-dcgpu` is MI300-series (this box),
`cdna` is the same family at twice the size, and `gfx110X`/`gfx120X` are RDNA and will
not run here.

**Pass the GPU groups as numeric GIDs resolved on the host.** `--group-add render`
fails outright — `Unable to find group render: no matching entries in group file` —
because the Ubuntu 24.04 image has no `render` group of its own. `video` exists in both
with no guarantee the numbers agree. On this host they are `video:44`, `render:991`.

A newer container ROCm than the host's is fine: ROCm **7.13** userspace in the image
drives the host's in-tree **6.12** `amdgpu` correctly, reporting "AMD Instinct MI300X
VF" and 304 CUs. Only the kernel driver and `/dev/kfd` come from the host.

Verified 2026-09-16, twice. First: image 44.3 GB unpacked, vLLM 0.19.1 serving
Qwen2.5-7B-Instruct **84 seconds** after container start, `max_model_len` 32768,
186.6 GB of the 191.7 GB VRAM preallocated at the default `gpu_memory_utilization`.
Then the box was moved to its actual job — `google/gemma-4-E2B-it` on
`vllm/vllm-openai-rocm:nightly-rocm100` (vLLM 0.29.1rc1.dev187, torch 2.12.0+rocm10.0.0,
transformers 5.17.0), 9,026,017 tokens of KV at `max_model_len` 32768 and 275.45x
concurrency, with text, thinking, tool calling and vision all verified live.

**With Gemma 4, newer is not safer.** `rocm/vllm:rocm10.0.0_..._vllm_0.27.0` is the
newest image AMD publishes and it **cannot load the model at all**: it lacks
`Gemma4ModelArchConfigConvertor`, so config parsing raises
`AmbiguousGlobalPerLayerAttributeError` on `head_dim` before the GPU is touched. Gemma 4
runs 256-wide heads on its sliding-attention layers and 512 on its full-attention ones;
transformers >= 5.15 reports that as a per-layer attribute and raises on a global read,
and vLLM's `getattr(..., 0)` default cannot catch it. The 0.19.1 image above works from
the other side, since transformers 5.8.1 predates per-layer attributes — so **the broken
build sits between the two that work**, and the pip channel vLLM's own recipe names is
staler than all three at 0.20.2rc1. The three-image comparison table now lives in the
companion article `~/gemma4-dev/gpu-vllm-mi300x-2b/devto-gemma4-mi300x-mcp.md`, which is
the deployment write-up; `devto-gemma4-mi300x-vllm.md` is the card-and-control-plane
article and keeps only a prose summary of the finding.

Check before pulling 35-62 GB, with `scaffold/gemma4-probe.py` — or after, via
`vllm_image_status(probe=true)`. Import `vllm.config` first, or a circular-import
`ImportError` makes a good image look broken, and map `/dev/kfd` and `/dev/dri` in, or
`get_arch_list()` returns `[]` and looks like a build with no kernels for you.

**The registry is `vllm.transformers_utils.model_arch_config_convertor`.** Worth
spelling out, because a probe that guessed four other plausible module paths reported
`gemma4_convertor: not found` against an image carrying ten `gemma4` modules and a
registered `Gemma4ModelArchConfigConvertor` — a false negative that would have sent
someone hunting for another 62 GB image. The probe now reports which modules it tried
alongside the answer, so "did not find it" cannot be mistaken for "this build lacks it".

Measured 2026-09-17 on `vllm/vllm-openai-rocm:nightly-rocm100` as pulled that day:
**35.1 GB**, vLLM reporting the odd version string `0.3.1.dev3+g0bfc7a15d`, torch
`2.12.0+rocm10.0.0`, transformers `5.17.0`, 36 convertors registered with `gemma4`
among them, and `gfx942` present in `get_arch_list()` alongside `gfx950` and `gfx1250`.
The version string is not the `0.29.1rc1.dev187` recorded earlier for this tag — nightly
tags move, so read the probe rather than assuming the build.

**The two images take different argv.** `vllm/vllm-openai-rocm` sets
`ENTRYPOINT ["vllm","serve"]`, so the model id is the first argument; AMD's images have
no entrypoint and need `vllm serve` spelled out. `vllm/docker-serve.sh` keys off the
image name.

**Audio is unreachable on every ROCm image tried.** E2B has a conformer audio encoder,
but none of these images ships the `vllm[audio]` extras — `librosa` and `soundfile` are
absent — so audio fails at request time whatever `--limit-mm-per-prompt` says. That is
why `LIMIT_MM_PER_PROMPT` sets `audio: 0`; it also skips allocating encoder memory for a
path that cannot be used. Audio needs a derived image, not a flag.

The default model is still deliberately ungated: a gated repo (Llama, and every Gemma
one before Gemma 4) turns a first run into a Hugging Face login problem instead of a
vLLM problem. `google/gemma-4-E2B-it` is Apache-2.0 and ungated, so it keeps that
property while being the thing this box is actually for. Set `HF_TOKEN` in `.env` if you
switch `VLLM_MODEL` to something gated.

## Quantization: fp8 is the only win on this card

Measured 2026-09-16 in the serving container (torch 2.12.0+rocm10.0.0, vLLM 0.29.1rc1), 8192³
matmul, 30 iterations, **while the card was also serving live traffic** — so the absolute rates
are depressed and the ratios are the result.

| dtype | TFLOP/s | % of spec peak | vs bf16 |
| --- | ---: | ---: | ---: |
| bf16 | 664.3 | 50.8% | 1.00x |
| fp16 | 662.5 | 50.7% | 1.00x |
| **fp8 `e4m3fnuz`** | **1172.5** | 44.8% | **1.77x** |
| int8 | 455.6 | 17.4% | **0.69x** |

**8192³ is not what decode runs — see the rerun at decode shapes, 2026-09-18.** Summed over Gemma 4
E2B's linear layers at M = 1–64 and replayed from a HIP graph, fp8 is **1.50x** bf16, ranging per
shape from 0.86x (fp8 slower) to 1.81x; int8 **refuses M ≤ 16** and is 0.20x at M = 64; and this
table's own 8192³ int8 figure reran at 0.40x, not 0.69x. Full write-up in
`~/gemma4-dev/gpu-vllm-mi300x-2b/benchmarks/runs/2026-09-18-gemm-decode-shapes-mi300x/REPORT.md`,
chip-level summary in `~/gemma4-dev/HARDWARE.md`.

**End to end, fp8 serving is slower than bf16 — 0.53x–0.80x output throughput** (vLLM 0.19.1,
2026-09-18), and `--quantization fp8` does not start at all on `nightly-rocm100` (both quant paths
fail on `e4m3fnuz`). Serve bf16. Details in
`~/gemma4-dev/gpu-vllm-mi300x-2b/benchmarks/runs/2026-09-18-vllm-sweep-mi300x-v0191-fp8/REPORT.md`.

- **bf16 is the baseline and fp16 is not a change.** Both go through the same CDNA 3 matrix cores
  at the same 1307.4 TFLOP/s peak. Gemma 4's config says `dtype: bfloat16` and the engine confirms
  `dtype=torch.bfloat16, quantization=None`. Moving to fp16 costs dynamic range and buys nothing.
- **int8 measured *slower than bf16*** — 0.69x, 17% of its own 2614.9 TOPS peak — even though the
  spec gives int8 and fp8 the same peak. `torch._int_mm` is not reaching tuned kernels on this
  stack. Do not reach for int8 here on the strength of the spec sheet.
- **fp8 is `e4m3fnuz`, not `e4m3fn`, and that is a trap rather than a detail.** `float8_e4m3fn` —
  the OCP flavour every NVIDIA checkpoint is published in — does not merely run slowly here, it
  raises `RuntimeError: HIPBLAS_STATUS_NOT_SUPPORTED`. vLLM agrees from its own side:
  `is_fp8_fnuz()` keys on `"gfx94"` and `fp8_dtype()` returns `torch.float8_e4m3fnuz`. **An fp8
  checkpoint built for H100 is therefore not drop-in.** Quantize online from the bf16 weights with
  `--quantization fp8` instead of hunting for a checkpoint.
- **fp4 does not exist on gfx942.** torch names the allowlist in its own error: `Block-wise scaling
  for Float8_e8m0fnu is only supported on gfx950,gfx1250`. You cannot even cast to it — `copy_()
  does not support casting Float4_e2m1fn_x2 to different types`. vLLM gates it identically:
  `supports_mx()` is `any(gfx in _GCN_ARCH for gfx in ["gfx95", "gfx1250"])`, which is **False**
  here. `mxfp4` appearing in `supported_quantization` is a generic list, not a hardware claim.
- **vLLM's mxfp4 emulation is a research instrument, not a deployment path.** With `supports_mx()`
  False the kernel is `EmulationMxfp4LinearKernel.apply_weights`: `dequant_mxfp4(...)` widens the
  weights back to bf16 *on every forward pass*, activations go through quantize-dequantize, and
  `F.linear` runs at bf16. Bf16 speed, no resident-memory saving during compute, plus dequant
  overhead and the full fp4 error. It answers "would this model survive fp4" before buying MI355X.
- **GGUF is compiled out of the image entirely.** Not gated — absent. `'gguf'` is not in
  `QUANTIZATION_METHODS` (the global registry, not the platform list),
  `vllm.model_executor.layers.quantization.gguf` is `ModuleNotFoundError`, and there are **no**
  ggml/gguf symbols in `vllm._custom_ops`. GGUF here means changing inference engine, not adding
  a flag.
- **The 4-bit that does work is weight-only.** `awq`/`gptq` are in the registry and
  `awq_dequantize` is compiled in. Storage and bandwidth only — the matmul is still bf16. With a
  ~2B model on 192 GB and KV cache at 0.2% utilisation there is no memory pressure to relieve, so
  it buys nothing here.

The attention backend in use is `TRITON_ATTN`, not AITER; `VLLM_ROCM_USE_AITER=1` is a separate
and so far untested lever.

## Not a gemma4-dev rig

This project borrows conventions from `~/gemma4-dev` but is not one of its rigs. The
four-slot `<platform>-<runtime>-<hardware>-<model>` naming scheme in its `NAMING.md`
does not apply here, and this directory does not need to conform to it.
