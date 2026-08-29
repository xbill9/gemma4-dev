# CLAUDE.md — `gpu-pytorch-g5g-2b`

Serving rig: **`google/gemma-4-E2B-it`** under **PyTorch + transformers** on **AWS EC2 G5g** —
a Graviton2 (aarch64) host with an **NVIDIA T4G** GPU (Turing, SM 7.5, 15360 MiB).

This is a full rig: `server.py`, an MCP server, a skill, a plugin manifest, and `tpu.env`.
It is **not** one of the `gpu-vllm-l4-*` artifact rigs, despite sharing the `gpu` platform slot.

The engine is `AutoModelForCausalLM` behind an OpenAI-compatible FastAPI server
(`torch_openai_server.py`), run under systemd — **not docker**, and **no `ports/` port of our
own**. Read `docs/INHERITED.md` before assuming anything here was measured here.

> **This file was the JAX sibling's copy until 2026-08-29.** It described `ports/gemma4/`, a
> Pallas W4A16 kernel, PLE quantisation and a hand-written KV ring — none of which exist in
> this rig. That copy still lives, correctly, at `../gpu-jax-g5g-2b/CLAUDE.md`; nothing was
> lost by replacing it here. Findings about the **chip** or the **model** remain in the root
> `HARDWARE.md` / `MODELS.md`, which is where to look for them.

## Why this rig exists

**Slot 2 is the only thing that moves.** `gpu-jax-g5g-2b` serves the same checkpoint on the
same silicon and spends **54% of decode in dtype conversion at 0.0% TensorCore utilisation** —
and its own 2026-08-28 experiment FALSIFIED the obvious explanation: converting the weights to
float16 changed throughput by +0.0%, because at `B=1` cuBLAS dispatches a **fp32 `gemvx`** with
no half path at all. A different framework on the same chip is the cheapest way left to ask
whether that is the chip or the framework.

**So the comparison is the deliverable, and it is only meaningful on the decode gauge.** Both
rigs emit `tpu_jax_decode_tokens_per_second`, which times decode alone. An end-to-end tok/s
carries prefill and HTTP and will not agree with it.

## The fork shipped a deployment that could not run — five fatal bugs

**FOUND AND FIXED 2026-08-29, on the first launch this rig ever attempted.** The rig was forked
from `gpu-jax-g5g-2b` on 2026-08-28, the code was rewritten before the prose, and 89 offline
tests passed throughout. **Not one of them asserted on the rendered bootstrap**, so every one of
these survived to the first real launch. They are listed because the *class* recurs: a fork
rewrites the parts you read and leaves the parts you execute.

| # | Bug | How it would have failed |
| --- | --- | --- |
| 1 | `verify_gpu()` ran `import jax` | jax is never installed here. Under `set -e` install.sh died, `INSTALL_DONE` was never written, and `get_install_progress` reported "INSTALL IN PROGRESS" **forever** |
| 2 | `ExecStart=…/jax_openai_server.py` | that file is not in the payload; the unit crash-loops |
| 3 | `_serve_argv` emitted `--quant-mode --ple-bits --int8-lm-head …` | `torch_openai_server.py` defines only `--model/--host/--port/--seq`; argparse exits 2 |
| 4 | `$PIP '{TORCH_PIP_SPEC}'` quoted | value is `transformers accelerate` — two packages; quoted, pip parses it as ONE requirement and fails |
| 5 | `torch.compile(step, backend="tpu")` | **in the payload.** There is no `tpu` backend in a CUDA build; it raised during warmup, before uvicorn bound the port |

**And a sixth that no amount of reading would have caught**, because it is a property of the AMI
rather than of the code — see the next section.

Tests now pin all of them: `test_bootstrap_has_no_jax_left_in_it`,
`test_serve_argv_only_emits_flags_the_server_defines`, and the stage-marker tests.

## The interpreter is not a free choice, and this is the rig's central hazard

**MEASURED 2026-08-29 on `i-025c02baf3c836e44`.** The DLAMI's torch lives at:

```
/opt/pytorch/bin/python   ->  torch 2.12.0+cu132
                              arch_list ['sm_75','sm_80','sm_90','sm_100','sm_110','sm_120']
```

That interpreter is **Python 3.13**, in a venv, and **not on PATH as `python3.12`**. `tpu.env`
says `TORCH_PYTHON_VERSION=3.12`, and Ubuntu 24.04's system interpreter *is* 3.12 — so the
forked bootstrap would have installed transformers into `/usr/bin/python3.12` and pointed
systemd there, producing `ModuleNotFoundError: No module named 'torch'` **after the install had
reported success.**

This is structurally different from the JAX sibling and the difference is one sentence:
**there, `pip install jax[cuda13]` put the runtime into whichever interpreter ran it, so any
python3.x would do. Here torch comes from the AMI, so the interpreter is discovered, not
chosen.** `install_runtime` probes for the first interpreter that can `import torch`, records it
in `{APP_DIR}/PYTHON_BIN`, and install, `verify_gpu` and `ExecStart` all read that one file.

**Never reintroduce a bare `command -v python3.12` on this path**, and do not "fix" the
mismatch by bumping `TORCH_PYTHON_VERSION` to 3.13 — that hardcodes a version AWS will move,
and the probe already handles it. `TORCH_PYTHON_VERSION` now only selects which interpreter apt
installs as a *fallback*.

**Corollary: this rig must never `pip install torch`.** Upstream PyPI aarch64 wheels omit
sm_75, so a pip torch would shadow the DLAMI's and serve on CPU, silently. `verify_gpu` asserts
`sm_75 in torch.cuda.get_arch_list()` and runs a real fp16 matmul, rather than trusting
`torch.cuda.is_available()`.

## The payload decodes with a KV cache, not a static buffer

The forked `torch_openai_server.py` kept ONE `[1, SEQ]` buffer, wrote each token with
`index_copy_` at a tensor position, and ran the model with **`use_cache=False`** under
`torch.compile(backend="tpu")`. That is the correct discipline on XLA, where a changing sequence
dimension recompiles. **On CUDA it is pure loss**: nothing recompiles, and every decoded token
pays a full `SEQ`-length forward. At `--seq 4096` each token would have re-read the whole
context.

It now prefills once, keeps `past_key_values`, and feeds one token per step. Two consequences
worth keeping:

- **`--seq` is a context BOUND, not a padded buffer length**, so raising it is free. There is no
  bucket ladder here and no padding, which is why the sibling's `pad_len` machinery and its
  window-eviction bug have no analogue in this rig — transformers owns the cache.
- **Prefill and decode are timed separately** and both go to `/metrics`. That split is the whole
  point: it is what makes this rig's number comparable to the sibling's.

## Metrics are named `tpu_jax_*` on a PyTorch rig, deliberately

They are wrong as description and right as an identifier. Every benchmark report in this family
compares on `tpu_jax_decode_tokens_per_second` **by name**, and `server.py`'s `_parse_prom`,
`get_metrics` and `verify_model_health` look up these exact strings. Renaming the prefix would
break continuity with the measurements this rig exists to be compared against. **The `rig` label
is what separates the runtimes.**

One mechanical constraint, easy to trip: `_parse_prom` pops `model` and folds every *remaining*
label into the sample key. So the numeric series carry **only** `model` — an extra label turns
`tpu_jax_decode_seconds_total` into a key `get_metrics` cannot find, and the cumulative-decode
line silently disappears. `rig` and `build_id` ride on `tpu_jax_precision_info`, which is parsed
separately.

`/metrics` did not exist at all before 2026-08-29, so `get_metrics` and `verify_model_health`
were both broken against this rig's payload.

## This rig has served — one run, and it answers the question it was built to ask

**`benchmarks/runs/2026-08-29-first-serve-g5g/` — MEASURED, the rig's first token.**
`g5g.2xlarge` spot in `us-east-1d`, torch 2.12.0+cu132 / Python 3.13, build `4ca8039100d7`.
**8/8 cells OK, 0 failed, 0 degenerate. Decode 10.88 tok/s median**, weights 10.209 GB.

**PyTorch is 15% SLOWER than the JAX sibling on identical silicon** (10.88 against 12.80 for
the config-fair `ple0` comparison; 13.10 is the JAX rig's quantised best). That is the opposite
of the result the rig was scaffolded expecting, and it is the useful one:

`gpu-jax-g5g-2b` spends 54% of decode in `wrapped_convert` kernels at 0.0% TensorCore, and its
own 2026-08-28 experiment already falsified the storage-dtype explanation — an all-float16 tree
moved throughput **+0.0%**, because at `B=1` cuBLAS dispatches a **fp32 `gemvx`** with no half
path. **This rig is the independent confirmation.** It loads directly as float16, has no
conversion pass to blame, and is slower anyway.

```
10.209 GB / 320 GB/s = 31.9 ms/step -> 31.3 tok/s   (a FLOOR)
measured                                 10.88 tok/s -> 35% of it
```

**Neither framework is within 3x of the bandwidth bound, and the 15% between them is far
smaller than the 3x each has to the hardware.** The deficit is not a framework artifact: it is
`B=1` decode being a matrix-*vector* product that no dtype and no runtime turns into a GEMM.
**The remaining 3x needs batching, which both rigs currently refuse** (`MAX_NUM_SEQS=1`, one
lock). Do not spend effort here on dtypes, kernels or runtimes — that question is now answered
twice, from two directions.

Two corroborations worth keeping:

- **Decode does not depend on context: 3.9% spread over a 27x range** (92 → 2,501 tokens), no
  trend. Independently reproduces the JAX rig's 3.4% over 100x, from a different framework and
  harness. **KV is not what sets decode speed** — E2B's ~18 KiB/token puts the whole 4K cache in
  the tens of MiB against 14.07 GB. **Never size this rig's context from KV arithmetic.**
- **End-to-end fell 10.12 → 6.91 tok/s across the same rows where decode was flat.** The two
  disagree by up to 36%, which is exactly why the gauge is the number and end-to-end is not.

**This rig loads 0.952 GB more than the sibling** (10.209 vs 9.257 GB) — precisely the non-text
towers the JAX loader skips. Resident, not streamed during text decode, so they cost memory and
not throughput.

## Batching is worth 7.84x and is free — MEASURED, and it is the next piece of work

**`benchmarks/runs/2026-08-29-profile-and-fixes-g5g/` — the rig's first profile.**

| B | ms/step | tok/s | peak GB |
| ---: | ---: | ---: | ---: |
| 1 | 93.21 | 10.73 | 10.271 |
| 2 | 94.43 | 21.18 | 10.308 |
| 4 | 94.92 | 42.14 | 10.382 |
| 8 | 95.05 | **84.16** | 10.529 |

**Per-step time grew 2.0% while the batch grew 8x**, for 0.258 GB. That is the direct
confirmation of what the first report could only argue from a roofline: the decode step is
dominated by costs independent of batch, so extra sequences are nearly free.

**84.16 tok/s at B=8 beats the vLLM sibling's 43-44 and the JAX sibling's 13.10 outright** — but
it is measured on the ENGINE, not through the server. `MAX_NUM_SEQS=1` and one lock mean the
served path cannot reach it. **Continuous batching is the highest-value work in this rig, and it
is now quantified rather than assumed.**

### Where the 93 ms/step goes

`torch.profiler`, 24 steps. Decode runs on `gemv` kernels (`gemv2T_kernel_val` 15.4%,
`internal::gemvx` variants 14.7%); `turing_fp16_s1688gemm_*` tensor-core kernels appear only
~106 times in 24 steps and are prefill. Three things settled:

- **SDPA is already active and attention is 1.0% of decode.** No free win there, and
  FlashAttention-2 needs Ampere. Stop considering it.
- **~5,650 kernel launches per step**, dominated by 1-3 µs elementwise kernels (`copy_` 955/step,
  `mul` 733/step, `pow` 504/step) on a chip whose launch overhead is 5-10 µs. **Launch-bound.**
  `torch.compile(mode="reduce-overhead")` + `StaticCache` is the direct fix and ranks second.
- **61.3 ms of the 93.2 ms step is not weight streaming** (the floor is 10.209 GB / 320 GB/s =
  31.9 ms), and none of it scales with B.

Note this rig's GEMV is templated on `__half`, so PyTorch is **not** uniformly promoting to fp32
the way the JAX sibling's profile showed. Do not carry that claim across.

## Two fixes the first run's code needed

### `logits_to_keep=1` — the sibling's bug, reintroduced

`forward` defaults `logits_to_keep=0`, meaning **keep all**. Prefill ran the LM head over every
prompt position and built a `[1, S, 262144]` tensor to use one row: **1.311 GB at S=2,501 and
2.147 GB at `--seq 4096`**, plus ~2 TFLOP of discarded matmul. This is `logits_at` in the JAX
sibling, reintroduced by writing the obvious `out.logits[:, -1, :]` — which slices *after* the
cost is paid.

**−12.6% per-token prefill** (0.6417 → 0.5611 ms/token), against ~17% predicted from the LM
head's FLOP share. **Compare on the range both runs share.** Run 2's slope over its own full
92–3,746 range is 0.6449 ms/token because attention is O(S²); differencing that against run 1's
shorter range hides the improvement completely.

### `device_map`, not `.to(device)` — the load was swap-bound

`from_pretrained(...).to(cuda)` builds the whole tree in HOST memory then copies it.

| | host peak | swap peak |
| --- | ---: | ---: |
| `.to(device)` | 14.0 GB | 2.8 GB |
| `device_map={"": 0}` | **10.52 GB** | **0** |

On a 16 GiB `g5g.2xlarge` the old path drove the box into the swapfile and spent minutes at
**98% iowait**. **Quote the memory, not the wall time** — the post-fix 11.3 s load also had a
warm page cache, so it is not a clean A/B; host peak and swap peak are.

The swapfile stays: it is the right safety net. This stops it being the load path.

## Coverage: the first sweep overstated it

Run 1 reported "8/8 cells" having swept only to **2,501 tokens against a `--seq` of 4096** — it
never approached the configured bound. Run 2 reaches **3,746 tokens at 10.91 tok/s**. `sweep.py`
now takes `--contexts`, so the range is named rather than inherited. Decode is flat to 6.2% over
a 41x context range.

**Not worth pursuing, now measured rather than assumed:** the attention backend (SDPA active,
1.0% of decode) and the weight storage dtype (falsified at +0.0% on the sibling; decode here
already dispatches `__half` GEMV).

## Profiling is installed on the box

`xprof` and `tensorboard` install with the serving deps and are **VERIFIED importable on
hardware** (xprof 2.23.1 `manylinux_2_35_aarch64`, tensorboard 2.21.0, both CLIs on PATH;
`torch.profiler.tensorboard_trace_handler` present). Ubuntu 24.04 is glibc 2.39, so xprof's
`manylinux_2_35` floor has headroom.

**They are shipped rather than left to an operator, deliberately.** The JAX sibling kept its
profiling deps in a `requirements-profiling.txt` that was *excluded* from the deploy payload, so
its own profiling recipe told operators to install from a path that had never existed on any
instance — xprof "installed" with `Could not open requirements file` and the extraction died on
`ModuleNotFoundError`, both in logs nobody reads.

**Their install is non-fatal on purpose.** A missing aarch64 wheel must not cost the serve, so
the branch warns loudly (`[stage] profiling-deps FAILED`) rather than dying under `set -e` —
which is precisely how `import jax` in `verify_gpu` used to kill the whole install.

`profile_decode.py` ships in the deploy payload and drives both of them. A 24-step decode with
`record_shapes=True` writes a **296 MB** tensorboard trace and `key_averages()` over it takes
minutes — narrow the window with `torch.profiler.schedule` if that matters.

## `apt-daily-upgrade` restarts the serving unit mid-load

**MEASURED 2026-08-29, and it cost two full checkpoint downloads.** A fresh Ubuntu 24.04 DLAMI
runs unattended-upgrades on boot. When it upgrades a library the service links against, it
`systemctl restart`s that service — and the restart lands squarely in the 9.5 GB download.

The failure is **invisible to every signal you would naturally check**:

```
systemctl show torch-g5g -p Result -p NRestarts -p ExecMainStatus
Result=success   NRestarts=0   ExecMainStatus=0
```

`Restart=on-failure` never fires, because nothing failed. The journal says
`Deactivated successfully`, which is what a clean stop looks like — the only tell is the PID
changing (3045 → 4784 → 9973) and `Stopping torch-g5g.service` immediately after
`Reexecuting requested from client PID N (unit apt-daily-upgrade.service)`.

**Do not diagnose this as an OOM.** It looks like one: memory peaked at 11.3 G then 14.0 G with
2.8 G of swap on a 16 GiB box, so the numbers invite the conclusion. But `dmesg` shows no OOM
kill, `systemd-oomd` has no entries, and an OOM kill is SIGKILL — it could not report
`Result=success`.

Masked by hand on this run:

```
systemctl stop apt-daily-upgrade.timer apt-daily.timer unattended-upgrades
systemctl mask apt-daily-upgrade.service apt-daily-upgrade.timer
```

**This is not yet in the bootstrap** and should be — it is the one open action item from this
run. Note the cost is worse on spot than the wall-clock suggests: each restart re-runs the
download, and the sibling has measured reclamation as early as 21 minutes.

## Engineering rules

- boto3 and the standard AWS credential provider chain — never shell out to the AWS CLI.
- SSM Run Command for remote administration; no inbound SSH rule, no private key.
- Require explicit subnet, security-group, and instance-profile ids. Do not create broad
  network or IAM policy.
- Scope instance discovery to `ManagedBy=gpu-pytorch-g5g-2b`.
- Hugging Face tokens live in Secrets Manager and are fetched at boot into a root-only
  `EnvironmentFile`. **Never** in user data — instance metadata is readable by anything on the
  box. `set +x` wraps the fetch because the script runs under `set -x` and bash traces
  assignments *with their values*. Tests assert both.
- Launches default to spot. Surface capacity errors rather than silently retrying.
- **Termination is cheap here.** There is no built image to lose with the root volume, only a
  pip install and the model cache. Do not import the vLLM sibling's stop-vs-terminate reasoning.
- Never hardcode an endpoint; `get_endpoint` resolves it from the instance.
- `verify_model_health` uses `/v1/chat/completions` and reads
  `tpu_jax_degenerate_responses_total` either side of its own probe. **Do not health-check by
  testing for a non-empty response** — the vLLM sibling was measured answering
  `': ok: ok: ok…'`, which is non-empty and worthless.

## Commands

Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v` (92 tests,
all passing as of 2026-08-29; 22 skip because they target the JAX port this rig does not carry).
They are fully offline — no AWS, no network, no GPU.

`make lint` runs `ruff check server.py refresh_skill.py torch_generate.py torch_openai_server.py
sweep.py tests`, then `bash -n` on four shell scripts. **A new top-level module is silently
unlinted until it is added to that list.**

**`deploy_torch_server` ships whichever payload root it resolves, and it prints which.**
`server.py` resolves the payload next to itself, and the MCP server runs from
`.claude/skills/…/mcp/`, so editing the working tree and deploying through the *registered* MCP
server ships the previous `make skill` output. **Always `make skill` before deploying**, and
read the `Payload root:` line in the deploy output.

**A subtler version of the same trap bit this rig on 2026-08-29:** the MCP server process loads
its snapshot at *startup*, so a `make skill` during a session does not reach an already-running
MCP server. The launch that produced this rig's first measurement was driven by importing the
rig-root `server.py` directly for that reason. If `/mcp` and your working tree disagree,
reconnect the server rather than trusting the tool descriptions.

`make skill` regenerates both snapshots. `SKILL.md` is a hand-written **source** in that tree —
`refresh_skill.py` will not recreate it, so `rm -rf .claude/skills` destroys it permanently.

There is no `make deploy` recipe on purpose: provisioning resolves an arm64 AMI at launch time,
and a Makefile would have to hardcode one.

## Benchmarking

**`sweep.py` lives at the rig root, not inside a run directory.** The JAX sibling kept its sweep
inside `benchmarks/runs/<run>/` and copy-pasted it per run, so every iteration re-derived its own
harness — three independent sources of drift between two numbers meant to be comparable.

```bash
python3 sweep.py --base http://<ip>:8000/v1 --out benchmarks/runs/<date>-<what>-g5g
```

Three things it is opinionated about, each because it has cost a measurement in this family:

- **Warm at the shape you measure**, and never average the warm-up in. Nothing is compiled here,
  but cuBLAS autotune and allocator growth are still per-shape.
- **Median, not mean**, over repeats — a spot host gives the occasional slow request.
- **Quote `tpu_jax_decode_tokens_per_second`**, not end-to-end. Both are recorded and they do
  not agree; the gauge is the one every sibling report compares on.

`benchmarks/README.md` and `serving-report.schema.json` are **synced copies** —
`make benchmarks-sync` at the monorepo root overwrites them. Edit the root originals.

Three numbers you will be tempted to reuse and must not:

- **13.10 tok/s** — `gpu-jax-g5g-2b`, same silicon, different runtime. That is the number this
  rig is *compared against*, not one it inherits.
- **43.1 / 44.24 tok/s** — the vLLM sibling, obtained with reduced Triton tiles.
- **Anything from `~/gemma4-tips`** — that tree duplicated its own artifacts and its directory
  names misattribute both model and chip.

## Agent-instruction files

`AGENTS.md` and `GEMINI.md` cover the same ground for other tools. There is no generator:
**`CLAUDE.md` is authoritative where they disagree**, and a convention change has to be applied
to all three by hand.
