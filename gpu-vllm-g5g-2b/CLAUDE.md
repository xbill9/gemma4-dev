# CLAUDE.md — `gpu-vllm-g5g-2b`

Serving rig: **`google/gemma-4-E2B-it`** under **vLLM** on **AWS EC2 G5g** — a Graviton2
(aarch64) host with an **NVIDIA T4G** GPU (Turing, SM 7.5, 16 GB).

This is a full rig: `server.py`, an MCP server, a skill, a plugin manifest, and `tpu.env`.
It is **not** one of the `gpu-vllm-l4-*` artifact rigs, despite sharing the `gpu` platform
slot with them.

## Why the hardware slot is `g5g` and not `t4g`

Settled 2026-08-12; `NAMING.md` has the carve-out. Do not "correct" this back.

The GPU really is an NVIDIA **T4G**, and slot 3's normal rule is the GPU SKU, so `t4g` looks
right. Two facts outweigh it:

- **`t4g` is already an EC2 instance family** — `t4g.nano`…`t4g.2xlarge`, Graviton2
  burstable **CPU** boxes with no GPU. In an AWS context the string reads as a cheap CPU
  instance far more often than as a GPU.
- **G5g is the only Graviton+GPU family AWS ships**, and no Graviton3 or Graviton4 GPU
  instance exists. So `g5g` is not a lossy stand-in for the chip the way `ec2` or `cloudrun`
  would be — it names the Graviton2+T4G pairing exactly, and that pairing, not the GPU alone,
  is what makes this rig hard. The whole build problem is aarch64 **and** SM 7.5 together.

The chip is still called T4G everywhere it is the chip being discussed — in this file, in
`HARDWARE.md`, and in `tpu.env`. Only the slot is `g5g`.

## The one thing to know before touching anything here

**G5g needs aarch64 and SM 7.5 together, and no prebuilt CUDA artifact provides both.**

`vllm/vllm-openai:v0.27.1` ships both platforms in one manifest. Read directly from the
published image config on 2026-08-12:

| Manifest | `TORCH_CUDA_ARCH_LIST` | SM 7.5? |
| --- | --- | :---: |
| `linux/amd64` | `7.5 8.0 8.6 8.9 9.0 10.0 12.0` | **yes** |
| `linux/arm64` | `8.0 8.7 8.9 9.0 10.0 11.0 12.0` | **no** |

The single arch this rig needs is the only one the two images disagree about, and the
Dockerfile sets **no `+PTX`** (deliberately, with a comment), so nothing JIT-compiles to
cover the gap. `docs/turing-aarch64-gap.md` has the reproduction command, the second layer
of the problem (PyTorch's aarch64 wheels look the same way), and what is still unverified.

Consequences that are easy to undo by accident:

- `serving='build'` is the default and compiles vLLM on the instance with
  `--build-arg torch_cuda_arch_list=7.5`. It takes **hours** on a Graviton2. Do not
  "simplify" it back to a plain `docker run` of the published image.
- `serving='stock'` runs the published image unchanged and is **expected to fail**. It is
  apparatus for reproducing the gap on real hardware, in the same spirit as
  `gce-vllm-v5e1-2b`. It is not a fallback.
- **Run `verify_gpu_arch` first.** It settles in minutes what the build path takes hours to
  discover, by printing the device capability, `torch.cuda.get_arch_list()`, and the result
  of a real matmul. A config flag being accepted is not evidence it did anything.

## Turing is not L4 — do not copy flags from a sibling

The five `gpu-vllm-l4-*` rigs and the legacy `~/gemma4-tips-aws` tree were all written for
SM 8.9. **Turing has no bf16 and no fp8.**

| | L4 siblings (SM 8.9) | this rig (SM 7.5) |
| --- | --- | --- |
| `--dtype` | `bfloat16` | **`float16`** — bfloat16 is a hard failure here |
| `--kv-cache-dtype` | `fp8` | **`auto`** — no fp8 datapath |
| attention | FlashAttention | **`XFORMERS`** — FA needs SM 8.0+ |
| `--quantization` | `compressed-tensors` (w4a16) | unused — see below |

This rig serves the **reference bf16 checkpoint**, so its name carries no encoding slot.
`MODELS.md` puts E2B at 9.5 GiB (8.97 measured) against the T4G's 16 GB, which leaves room
for a real KV pool at 18 KiB/token. The QAT w4a16 route the L4 rigs take is not needed and
would land on Marlin kernels that want SM 8.0+ anyway.

## Instance sizing

`g5g.xlarge` is **rejected by `_validate_instance_type`**, not merely discouraged: 8 GiB of
host RAM cannot stage E2B's 9.5 GiB of weights, and failing at validation beats an OOM-kill
twenty minutes into a boot. `g5g.2xlarge` (16 GiB) is the floor and the default.

`g5g.16xlarge` and `g5g.metal` carry two T4Gs; `_tensor_parallel_size` derives TP from the
GPU count, so those get `--tensor-parallel-size 2`. Every other size gets 1.

## AMI resolution

`_resolve_ami` filters on `architecture=arm64`. This is load-bearing. The legacy tips-tree
rigs hardcode `ami-012ba162b9cd2729c`, an **x86_64** DLAMI that cannot boot on Graviton2.
Never hardcode an AMI id here; resolve it at launch.

## Engineering rules

- boto3 and the standard AWS credential provider chain — never shell out to the AWS CLI.
- SSM Run Command for remote administration; no inbound SSH rule, no private key.
- Require explicit subnet, security-group, and instance-profile ids. Do not create broad
  network or IAM policy. (The legacy sample this was scaffolded from auto-creates a security
  group open to `0.0.0.0/0` on 22 and 8080 — that was not carried over.)
- Scope instance discovery to `ManagedBy=gpu-vllm-g5g-2b`. Unlike the inf2 rig, which keeps
  a legacy tag to avoid orphaning instances, this rig is new and uses its own name.
- Hugging Face tokens live in Secrets Manager and are fetched at boot. **Never** in user
  data — instance metadata is readable by anything on the box. A test asserts this.
- Launches default to spot. Surface capacity errors rather than silently retrying.
- Termination is permanent, and here it also destroys the locally built SM 7.5 image with
  the root volume — the next launch rebuilds from source. Weigh stop against terminate.
- Never hardcode an endpoint; `get_endpoint` resolves it from the instance.
- `verify_model_health` uses `/v1/chat/completions`. Raw `/v1/completions` returns an empty
  completion on `-it` models, so an empty body there is not evidence of a broken deploy.

## Commands

Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v`. They are
fully offline — no AWS, no network, no GPU — and pin the facts above, including that
`tpu.env` and `server.py` still agree.

`make lint` runs `ruff check server.py refresh_skill.py tests` then `bash -n` on the three
shell scripts. **A new top-level module is silently unlinted until it is added to that
list.**

`make skill` regenerates the snapshots under `.claude/skills/` and `skills/`. **Only the
three `mcp/` files are generated** — `server.py`, `project-setup.sh`, `requirements.txt`.
`SKILL.md` sits in the same tree and is a hand-written **source**: `refresh_skill.py` will
not recreate it. So `rm -rf .claude/skills` destroys it permanently, which is what happened
during the t4g→g5g rename. `test_skill_is_complete_in_both_copies` now guards it.

## MCP registration lives in four places

`.mcp.json`, `.claude-plugin/plugin.json`, `.codex/config.toml`, and
`.claude/settings.local.json`'s `enabledMcpjsonServers`. All four name the server
`gpu-vllm-g5g-2b`, which prefixes every tool as `mcp__gpu-vllm-g5g-2b__…`. `.mcp.json` and
`settings.local.json` are gitignored and generated by `project-setup.sh`; the other two are
committed. Keep them agreeing — a mismatch makes `/mcp` and the tool prefix disagree about
what this rig is.

`AGENTS.md` and `GEMINI.md` cover the same ground for other tools. There is no generator:
**`CLAUDE.md` is authoritative where they disagree**, and a convention change has to be
applied to all three by hand.

There is no `make deploy` recipe on purpose: provisioning resolves an arm64 AMI at launch
time, and a Makefile would have to hardcode one.

## Measurement

**This rig has no measurements of its own yet, and none may be attributed to it.** It has
provisioned nothing. `benchmarks/` carries only the schema and README, which are **synced
copies** — `make benchmarks-sync` at the monorepo root overwrites them, so edit the root
originals, never these. `reports/` and `runs/` stay in the rig. The L4 4B
artifacts that arrived with the directory it was scaffolded from were deleted rather than
left to be counted against it by `benchmarks/rollup.py` — the same call made for
`tpu-vllm-v5p1-2b`.

When the first run happens, `benchmarks/runs/<date>-<what>-g5g/` is the naming — `<hw-short>`
equals the hardware slot, so it follows the rename — and the
first thing worth recording is the `verify_gpu_arch` output, because it is the finding this
rig was built to establish.
