# CLAUDE.md — gpu-vllm-t4-2b-w4a16

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

**vLLM on the NVIDIA Tesla T4 already attached to this Compute Engine VM, serving
Gemma 4 E2B as 4-bit weights end to end**:
`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` — Google's QAT weights repacked
to compressed-tensors W4A16, text only, with the PLE table, `embed_tokens` and an
untied `lm_head` packed int4 as well. `tpu.env` holds the full lineage.

**STATUS 2026-09-29: serving and measured**, vLLM 0.29.0. Warm compile cache,
`tpu.env` flags: model loading **2.86 GiB**, KV cache **1,099,362 tokens**, decode
**109.7 tok/s** at c=1, 8 of 8 greedy outputs token-identical to the
bf16-embedding build. `evidence/2026-09-29-embed-int4.txt`. **Concurrency sweep
2026-09-29** (`benchmarks/runs/2026-09-29-emb4-sweep-t4/`, schema report in
`benchmarks/reports/`): 8 cells, 3 repeats, worst-cell cv 2.5%, 239.6 tok/s at
512/c=8. Its bf16 and QAT comparison cells are the parent's 2026-09-18 run, so the
comparison is three variables at once — `REPORT.md` says which.

**Forked from `gpu-vllm-t4-2b` on 2026-09-29**, which had been serving 4-bit
builds under a bare `-2b` name — a claim of the bf16 reference weights under
`@NAMING.md`. That rig is back on bf16. **Slot 5 is the only slot that differs**,
so the pair is an encoding A/B on identical hardware, runtime and flags.

**The two rigs share one GPU and port 8000, and neither sees the other's
process** — each `server.py` tracks only its own `run/vllm.pid`. Stop one before
starting the other. `~/bin/vllm-t4` defaults to this rig (since 2026-09-29); set
`VLLM_T4_RIG=~/gemma4-dev/gpu-vllm-t4-2b` to drive the bf16 parent.

## `repack/` — the tools that built the checkpoint

Self-contained copies, first written in `jev-tpu-31b` (which keeps its
originals, so the published model cards' links still resolve). Run with
`PYTHONUSERBASE=/opt1/pyuser /usr/bin/python3.13`; numpy only, no torch.

| Script | Does |
| :--- | :--- |
| `repack_q4_0.py` | `-qat-q4_0-unquantized` → compressed-tensors W4A16, recovering each group's grid step |
| `text_only.py` | drops the vision/audio towers, config → `Gemma4ForCausalLM`; everything else byte-identical |
| `embed_int4.py` | packs the PLE table, and with `--embed-tokens` also `embed_tokens` + an untied int4 `lm_head`; fp16 scales by default |

Three things that bit while writing them, all recorded in `@QUANTIZATION.md`:

- **Output lands where the input's filesystem is.** Unchanged shards are
  hard-linked; `~/.cache` is a symlink to `/opt1`, so write outputs under `/opt1`
  or the root disk (~4-6 GB free) fills with copies.
- **Host RAM is 7.8 GB.** `embed_int4.py` works in row chunks for that reason;
  do not "simplify" it into one whole-table pass.
- **fp16 scales help only when the step is chosen for exactness**, not for least
  squares — see the script's `recover()` docstring.

## Slot 1 is `gpu`, not `local`, and this is the question people will re-open

Nothing here provisions anything. The GPU exists whether or not this code runs;
there is no launch, no image resolution, no capacity to find, no teardown. That is
`local`'s test, and it is **not** how `@NAMING.md` decides the slot:

> `local` is a claim about who owns the machine, not about where you are typing.
> **SSH into a cloud VM you provisioned and it keeps that VM's platform value**; a
> workstation is `local` whether or not you are sitting in front of it.

This machine is a GCE `n1-standard-2` in `us-west2-b`. So the slot is `gpu` — "a
general-purpose GPU attached to a VM, any cloud" — and this is the **first `gpu`
rig here with no provisioning half**. That combination is new; the slot value is
not. Do not rename it to `local-vllm-t4-2b`.

Nor to `gce-`: `@NAMING.md` reserves `gce` for the TPU-on-Compute-Engine case and
says "a GPU rig is `gpu` wherever it runs."

## Slot 3 is `t4`, the SKU — and the EC2 rule does not apply here

`gpu-jax-t4-2b` and `gpu-jax-l4-2b` were created under the SKU reading and renamed
to `g4dn`/`g6` the same day, 2026-08-28. **That rename does not reach this rig**,
because the rule that caused it is explicitly an EC2 rule: the instance family is
what you provision, quota and pay for on EC2, and it pins the host architecture.
Everywhere else — `local` included — slot 3 is the GPU SKU. This is Compute
Engine, so `t4`.

Do not invent an `n1` slot either. `@NAMING.md` keeps cloud names and machine
types out of slot 3 entirely.

`t4` also does not collide with anything: the collision argument that helped earn
the EC2 carve-out was about `t4g`, which AWS also sells as GPU-less Graviton2 CPU
instances.

## The A/B twins

- **`gpu-vllm-t4-2b`** — same everything except slot 5. The encoding comparison.
- **`gpu-vllm-g6-2b-w4a16`** — same encoding on an L4, and the xbill9 repacks run
  there as a `MODEL_NAME` override. Slots 1 and 3 differ, so a difference between
  the two is hardware plus control plane, never the weights.

The serving flags in `tpu.env` are the parent's, carried verbatim so that the
encoding A/B stays clean. Change them in both rigs or in neither.

## What actually blocks this rig, and it is not what it looks like

**Not VRAM.** The sibling MEASURED 9.8 GiB of weights inside 15.0 GiB of usable
HBM on this same part and still got a 329,579-token KV pool. A report that names
VRAM as the blocker on a T4 serving E2B has its arithmetic wrong.

**The install was broken, and instructively so** (fixed 2026-09-18 — see `tpu.env`
for the upgrade command and the `--no-deps` trap). `/opt1/pyuser` — the user base
for `/usr/bin/python3.13` — held a CUDA vLLM sitting on **`torch 2.11.0+cpu`**:

```
2.11.0+cpu  None  []
ImportError: libcudart.so.13: cannot open shared object file
```

Two packages present, neither usable. **Presence of files is never evidence**, which
is why `_vllm_files_present()` feeds only the disk budget and `verify_gpu_arch`
asks the interpreter.

**Three filesystems, and one `df` lies about all of them.** MEASURED 2026-09-17:

| path | total | free | holds |
| :--- | ---: | ---: | :--- |
| `/` | 33.57 GB | **4.23 GB** | pyenv 3.12's site-packages |
| `/tmp` | 3.90 GB | **3.88 GB** | pip's unpack directory |
| `/opt1` | 263.09 GB | **249.20 GB** | `~/.cache` (symlink), `/opt1/pyuser`, `/opt1/tmp` |

The first draft of this rig read `df /`, saw 4.23 GB and wrote "disk binds" into
`tpu.env`, the README and the server docstring. **It was wrong in both directions
at once**: the checkpoint has 249 GB because `~/.cache` is a symlink onto the big
volume, and `/tmp` is a third filesystem that a default `pip install` would have
failed in anyway. `check_host_capacity` now measures each target path separately —
and the tool caught the error, which is the argument for it being a tool.

**Two interpreters, and the default is the wrong one.** `python3` is pyenv 3.12.13
(site-packages on the full root disk); `/usr/bin/python3.13` with
`PYTHONUSERBASE=/opt1/pyuser` has both the packages and the room. `PYTHON_BIN`
names it, `_child_env()` carries the redirections, and
`test_no_call_site_hardcodes_python3` pins it. A probe against the wrong
interpreter gives a confident answer about something that will never serve.

**A relocated `PYTHONUSERBASE` is not a virtualenv.** The root `CLAUDE.md` forbids
venvs because the rigs deploy as a docker image with system-wide site-packages;
this is the same interpreter's user site on a different disk, and that property is
untouched.

## Host facts that differ from every measured sibling

- **No swap as provisioned, and vLLM cannot load E2B without it.** MEASURED
  2026-09-18: the default loader was OOM-killed on host RAM (EngineCore 4.0 GB anon
  RSS, host peak 7,252 of 7,800 MiB) during weight loading. A 16 GB swapfile on
  `/opt1` fixed it, peaking at 6.8 GB of swap for bf16 and 4.8 GB for QAT. **It is
  not in fstab** — after a reboot, re-create it (command in `tpu.env`) before
  starting the server.
- **7.80 GB of host RAM against the sibling's 16 GiB.** This was recorded as an
  UNMEASURED risk and it turned out to be the binding one — see the swap bullet.
  `_capacity()` still deliberately
  does not let host RAM veto the budget, and a test pins that.
- **2 vCPU against 4.** Not expected to cap throughput: the g4dn run's own notes
  reason that decode is GPU-bandwidth-bound at these rates and that a 42–48% gap
  between two spec-identical GPUs "CANNOT be attributed to the host CPU". Expect it
  to cost **startup**: 82.39 s of a 150.90 s engine init was compilation on 4 vCPU.

## The Turing work is the EC2 siblings' — except the delivery

Do not re-derive the chip facts here. `float16` because Turing has no bf16
datapath (and bf16 does **not** fail — PyTorch upconverts and vLLM logs the cast,
so the guard protects against a silent cost rather than an error). No fp8 KV. No
`VLLM_ATTENTION_BACKEND`, because vLLM does not recognize it and forces
TRITON_ATTN for Gemma 4 regardless.

**What is new is that the patch target is the host's own site-packages.** Full
write-up in `docs/turing-on-a-gce-t4.md`. Two things to carry:

- `pip install -U vllm` **silently reverts the clamp**, and unlike the EC2 rigs
  there is no derived image tag to notice it. Re-run `verify_turing_patch` after
  any vLLM change.
- The idempotency check keys on a **shared `SENTINEL`**, not on this rig's
  `MARKER`. site-packages is host-wide, so a sibling script may have patched the
  same file; keying on the rig name would insert a second clamp, halve the tiles
  again, and report success.

## Tool order, and why it is not decorative

`check_host_capacity` → `verify_gpu_arch` → `apply_turing_patch` →
`verify_turing_patch` → `start_vllm_server`. Each step's failure is cheaper than
the next one's, which is the whole ordering rule from `@RIG-ANALYSIS.md`: the
arithmetic is free, the install is minutes, an unclamped engine start is ten
minutes and blames the wrong thing.

`start_vllm_server` **fails closed** — it refuses unless the clamp is *positively
confirmed*. An earlier version refused only on the literal string `UNPATCHED`,
which let through exactly the state this host is in: verification that cannot
import vLLM at all returns neither answer, and a whitelist of known-bad answers
admits every unknown-bad one. A test pins this.

## Conventions

- Tests are `unittest`, never pytest: `python3 -m unittest discover -s tests -v`.
- Every subprocess goes through `run_command(cmd: list[str])` or `_probe()`, both
  on `asyncio.create_subprocess_exec`. **Never `shell=True`** — checked by AST,
  because `run_command`'s own docstring says the words.
- MCP tools are `async def` returning markdown with emoji prefixes (`✅`, `❌`, `📡`).
- `Optional[str]`, not `X | None`.
- Use the system `python3` for the MCP server and install into it; **never create a
  virtualenv**. The model's interpreter is `PYTHON_BIN`, which is a different thing.
- `tpu.env` is the source of truth and is committed. Never add `*.env` to
  `.gitignore`.
- Read **`MemAvailable`, never `MemFree`**.
- **Never a single `df`.** Disk is measured per target path. This host is the reason.
- `.claude-plugin/` and `.codex/` register the MCP server as `gpu-vllm-t4-2b-w4a16`
  (the directory name — `RIG_NAME` is derived from it). No `skills/` yet; if one is
  added its name must be `gpu-vllm-t4-2b-w4a16-management`, because `make
  skill-install` does `rm -rf` on its destination.

## Canonical root references

Read these before deriving their numbers here, and correct them **there**:
`@MODELS.md` (checkpoint properties, KV cost, weight footprints — including the
**open discrepancy** that says not to size KV from 18 KiB/token on the vLLM CUDA
path), `@HARDWARE.md` (accelerator properties; its "T4G" section is the chip),
`@QUANTIZATION.md`, `@NAMING.md`, `@RIG-ANALYSIS.md` (the order to consult them in).

Two root-file corrections this rig's work implies, neither made yet:

- `@HARDWARE.md` says `--dtype bfloat16` is "a hard failure" on this part and that
  the Turing-capable vLLM attention backend is `XFORMERS`. Both predate the
  2026-08-12 G5g measurement: bf16 upconverts silently, and vLLM ignores the
  backend variable and forces `TRITON_ATTN`.
- `@MODELS.md`'s KV open-discrepancy box cites the L4 run as the lone
  ~9.6 KiB/token observation. The T4 sibling's 2026-08-30 run reports the same
  329,579-token pool as the T4G baseline, which works out at the same ~9,622
  B/token — a second and third CUDA part, already in this tree.
