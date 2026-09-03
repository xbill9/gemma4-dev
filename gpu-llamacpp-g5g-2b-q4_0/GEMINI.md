# GEMINI.md (Gemini CLI) — `gpu-llamacpp-g5g-2b-q4_0`

> Maintained by hand alongside `CLAUDE.md`, which is **authoritative where they
> disagree**. There is no generator. A convention change lands in all three by hand,
> and THE NOTE below also lives in the Ollama sibling's three copies.

Serving **`google/gemma-4-E2B-it-qat-q4_0-gguf`** — Google's QAT checkpoint in its native
**Q4_0** packing — through **`llama-server`** on **AWS EC2 G5g**: a Graviton2 (aarch64) host
with an **NVIDIA T4G** GPU (Turing, SM 7.5, 15360 MiB).

Full rig: `server.py`, an MCP server, a skill, a plugin manifest, `tpu.env`. Not one of the
`gpu-vllm-l4-*` artifact rigs, despite sharing the `gpu` platform slot.

> **STATUS: SERVED ITS FIRST TOKENS 2026-09-02** on `i-05005979aff5a0df9` — see FIRST LIGHT
> below, which corrects the AMI reasoning in this file. Scaffolded the same day from
> `gpu-pytorch-g5g-2b`.
> `benchmarks/` is deliberately empty — the sibling's four runs were deleted rather than left
> for `benchmarks/rollup.py` to count against this rig, following the `tpu-vllm-v5p1-2b`
> precedent. **Every number below is either arithmetic from the artifact's own tensor table or
> a measurement from a sibling, and says which.** None of it is this rig's result.

## Why this rig exists

**It is the only rig in the monorepo serving 4-bit weights on a GPU**, because every other
route is closed. Verified 2026-09-02 and written up in the root `QUANTIZATION.md`:

| Stack | Loads the GGUF? |
| --- | --- |
| **vLLM 0.26.0** | **No.** `grep -ril gguf` over the installed package returns 2 incidental files; no `gguf.py` in `layers/quantization/`; `QUANTIZATION_METHODS` has 31 entries and none is `gguf`. **This is not a TPU-only gap** — the CUDA build has been checked |
| **JAX** | **No.** No GGUF reader exists anywhere in the ecosystem |
| **transformers 5.12.1** | Reads it, then dequantizes to **fp32** (a 9.4 GB host transient on `per_layer_token_embd` alone) and **silently drops 35 `layer_scalar` tensors** — a converter, not a serving path |
| **llama.cpp** | Yes, natively — upstream `src/models/gemma4.cpp` plus four `mtmd` multimodal variants |

`ggmlc` does not help: it compiles PyTorch/JAX graphs **to** GGUF (an exporter), its text
coverage stops at Gemma 3, and it writes Q4_0 rather than reading it.

## The artifact is the same QAT weights the TPU rigs serve unpacked — proven, not assumed

Range-read off the Hub 2026-09-02; nothing was downloaded whole.

```
gemma-4-E2B_q4_0-it.gguf     3,349,516,256 B   sha256 fa401b55…
gemma-4-E2B-it-mmproj.gguf     986,833,664 B   sha256 021059cc…
```

GGUF v3, `general.architecture = gemma4`, **541 tensors**, 49 KV pairs, full E2B shape intact
(`shared_kv_layers=20`, `sliding_window=512`, `key_length=512` / `key_length_swa=256`).
Dtype histogram **Q4_0 ×275, F32 ×263, Q6_K ×2, F16 ×1**.

Four F32 norm tensors read out of the GGUF are **bit-identical** to the bf16 tensors in
`-qat-q4_0-unquantized`: `blk.0.attn_norm` ↔ `layers.0.input_layernorm` (9.375, 7.9375,
10.6875; mean +10.67993), `output_norm` ↔ `model.norm` (+14.20042), `blk.0.ffn_norm` ↔
`layers.0.pre_feedforward_layernorm` (+19.20642), `blk.0.layer_output_scale` ↔
`layers.0.layer_scalar` (0.02087402).

**Corollary worth carrying: Gemma 4 has NO `1 + weight` norm convention**, unlike Gemma 1/2/3.
`Gemma4RMSNorm.forward` is a plain `normed * weight`, and the bit-identity above confirms
llama.cpp writes no offset. transformers omitting `gemma4` from `TENSOR_PROCESSORS` is
therefore **correct**. Never point it at `Gemma2TensorProcessor` — that subtracts 1 from every
norm in the model, and the result looks plausible. Filed in `MODELS.md`.

## What Q4_0 buys, and the number you must not derive from it

Derived from the file's own tensor table, against the PyTorch sibling's **measured** fp16:

| | fp16 (measured) | Q4_0 (derived) |
| --- | ---: | ---: |
| Streamed per decode step | 4.514 GB | **1.407 GB** |
| Resident | 10.209 GB | **~1.4 GB** (was 3.35 GB — corrected 2026-09-03: `per_layer_token_embd`, 58% of the GGUF, is `TENSOR_READ_LAZY` and stays on the host. MEASURED 1618 MiB on `local-llamacpp-1650ti-2b-q4_0`; filed in `MODELS.md`. Free budget is ~8.8 GB of 14.07, not ~6.9 GB) |
| Bandwidth ceiling @ 277 GB/s | 61.4 tok/s | 197 tok/s |
| Measured decode | 10.88 tok/s (18% of ceiling) | — |

**Do not turn 197 into a prediction.** Decode at `B=1` on this silicon is **launch-bound**, not
bandwidth-bound — ~5,650 kernel launches per step at 1–3 µs on a chip with 5–10 µs launch
overhead. The two controlled experiments in `QUANTIZATION.md` are what a bandwidth cut actually
buys here: `ple_bits=4` removed **3.505 GB** for **0.0%**, and `int8_lm_head` removed 11.9% of
streamed bytes for **+2.3%**.

**The win is residency.** Freeing ~8.8 GB of a 14.07 GB budget is what pays for batching, and
the PyTorch sibling measured batching at **7.84x** (B=8, 84.16 tok/s) for 0.258 GB. That is the
experiment this rig exists to make cheap — but run it deliberately, with `PARALLEL_SLOTS` raised
in a run whose REPORT.md says so. Changing the default silently invalidates every comparison.

Note also that llama.cpp **splits `--ctx-size` across `--parallel` slots**, so raising the slot
count divides per-sequence context rather than adding memory.

## Three silent failures, all asserted rather than trusted

Every one of these produces a server that **starts, binds, and answers correctly**. None
produces an error. That is why each has an explicit assertion.

1. **A CPU-only build.** cmake's default on a missing toolkit is to configure *without* CUDA.
   `-DGGML_CUDA=ON` makes that a configure failure, and `verify_gpu` then greps the built
   binary's own `--list-devices` for CUDA before the unit is ever enabled. `verify_model_health`
   additionally flags a decode rate under 3 tok/s as consistent with CPU.
2. **A partial GPU offload.** `--n-gpu-layers 999` offloads everything; a partial offload is the
   same silent slowness one step later.
3. **No `nvcc`.** `install_runtime` probes and **exits 1**. It does not fall back to apt's
   `nvidia-cuda-toolkit`, which installs a toolkit unrelated to the driver on the box — trading
   a clear failure for a second silent-wrongness path.

**The AMI choice is load-bearing, and what it actually supplies was WRONG here until the first
launch — see FIRST LIGHT.** The JAX and PyTorch siblings argue for the *base* driver-only image
because they never use the DLAMI's PyTorch. This rig compiles CUDA code, so it needs `nvcc` — but
`nvcc` is **not** at `/usr/local/cuda` on the full DLAMI either. It is a pip package inside the
PyTorch venv, which is why the full image is still the right choice and why the probe globs for
it. Do not "optimise" back to the base image; it has no toolkit at all.

## FIRST LIGHT — 2026-09-02, and the AMI reasoning in this file was WRONG

**MEASURED on `i-05005979aff5a0df9`**, `g5g.2xlarge` spot in `us-east-1a`, AMI
`ami-07a66fa2acbcfea88`, driver 595.71.05.

### The nvcc guard fired, correctly, for the wrong reason

`install_runtime` exited 1 with `FATAL: no nvcc`. **The ARM64 PyTorch DLAMI carries no
`/usr/local/cuda` at all and no `nvcc` on `PATH`** — so the section above claiming "the full
DLAMI carries the toolkit" was false, and the guard is the only reason that surfaced as a dead
install rather than a CPU-only build.

**The toolkit is on the image. It ships as a pip package inside the DLAMI's PyTorch venv:**

```
/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc     13.2
                                         .../cu13/include/          cuda_runtime.h, cublas_v2.h
                                         .../cu13/lib/              libcudart.so.13, libcublas.so.13
/usr/lib/aarch64-linux-gnu/libcuda.so.1                             driver, separate
```

**This is the PyTorch sibling's central hazard one level down.** There, torch comes from the AMI
so the *interpreter* is discovered rather than chosen. Here the *toolkit* comes from the AMI, at
a path that moves with the DLAMI's Python version. `install_runtime` now globs
`/opt/*/lib/python*/site-packages/nvidia/cu*/bin/nvcc` and derives `CUDA_ROOT` from it. **Never
hardcode that path.**

**Two follow-on problems, both measured:**

1. **pip wheels ship versioned sonames only.** `libcublas.so.13` exists; plain `libcublas.so`
   does not, and cmake's `FindCUDAToolkit` wants the bare name. The bootstrap now creates those
   symlinks. Without them the configure fails to find cuBLAS with the library sitting right there.
2. **The CUDA libs are not on the loader path at runtime either**, so `LD_LIBRARY_PATH` is
   rewritten in the EnvironmentFile after the probe — the same discipline as the PyTorch sibling
   rewriting `ExecStart` once it knows which interpreter holds torch.

**apt's `nvidia-cuda-toolkit` is still not the answer**, and the launch confirmed why: its 24.04
candidate is **12.0** against a **CUDA 13** driver.

### It builds, and the binary sees the GPU

`cmake` configures with `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=75`, `LLAMA_CURL=ON`; the
build completes; and the binary's own device list reports:

```
CUDA0: NVIDIA T4G (14912 MiB, 14807 MiB free)
```

`--hf-repo`/`--hf-file` pulled the 3.35 GB GGUF, `/health` returns `{"status":"ok"}`, and
`/metrics` exposes `llamacpp:tokens_predicted_total` / `_seconds_total` exactly as the shim
assumes.

### Decode: 85.74 tok/s median — and the comparison is NOT settled

Seven repeats, 128 tokens each, warm-up discarded, measured as deltas on the rig's own counters:

```
85.34  85.56  85.69  85.74  86.30  86.45  86.69
median 85.74 tok/s   spread 1.6%
```

The Ollama sibling, **same silicon, same protocol, same day**, measured **95.48 tok/s** — i.e.
**Ollama is ~11% faster than the llama.cpp it wraps.** That is the opposite of what a wrapper
should do, and THE NOTE already names the reason it is not a paradox: **the two do not
necessarily run the same machine code.**

At least three candidate causes, none eliminated:

- **Different CUDA toolkit.** This build used the AMI's **13.2**; Ollama's `cuda_v12` bundle is
  built with **12.8**. Different codegen from the same source.
- **Different llama.cpp revision.** This is `b10760`; Ollama v0.33.2 ships `libllama.so.0.3.0`,
  which is some other upstream point.
- **Different defaults** — flash attention, KV cache type and batch sizes are not pinned to match.

**And the two figures are each server's own statistic.** Ollama's is `eval_count/eval_duration`;
this is `tokens_predicted_total/_seconds_total`. Nothing has shown they measure the same span.
**Do not publish "Ollama beats llama.cpp" off this.** The comparable number needs `sweep.py`'s
client-side stream statistic run against both, which has not been done.

What it does support: 85.74 against the Q4_0 ceiling of 197 tok/s is **44%**, in the same band
as the vLLM sibling's 52% against its fp16 ceiling — consistent with the launch-bound reading,
and with Q4_0 winning on absolute throughput because **the ceiling moved**, not the efficiency.

## What the Ollama sibling's first launch already established for this rig

**2026-09-02**, on `i-0db1233beb07f0da7`. The two rigs share their entire AWS scaffolding, so
these carry across and do not need re-establishing here:

- **The AMI resolves and boots with a GPU.** `ami-07a66fa2acbcfea88` on `g5g.2xlarge`;
  `nvidia-smi` reports **NVIDIA T4G, compute cap 7.5, 15360 MiB**. So the driverless-ARM64-DLAMI
  hazard did not fire.
- **Capacity took four AZs.** `us-east-1d`, `1b` and `1c` all returned
  `InsufficientInstanceCapacity` before `1a` granted. Loop the AZs; AWS's "choose another
  instance type" text is on-demand boilerplate.
- **Spot, SSM, tagging and the whole `create → get_install_progress → verify → health` path
  work**, and the instance was SSM-ONLINE ~15s after `running`.

**What it did NOT establish, and what this rig still has to prove on hardware:** that `nvcc` is
present on that AMI, that the cmake configure succeeds with `-DGGML_CUDA=ON`, and that
`--hf-repo`/`--hf-file` work under `LLAMA_CURL=ON`. **This rig has still never been launched.**

## It compiles, and that is not incidental

**llama.cpp publishes no prebuilt Linux aarch64 CUDA binary.** Verified against release
`b10760` (2026-09-02): the arm64 Linux assets are `-bin-ubuntu-arm64` (CPU) and
`-bin-ubuntu-vulkan-arm64`; every `-cuda-` Linux asset is x64; the only arm64 CUDA build targets
Windows.

`LLAMA_CPP_REF` is **pinned, never `master`**. llama.cpp cuts several releases a day
(b10754…b10760 all landed on 2026-09-02), and an unpinned build makes the engine a moving
variable across two runs meant to differ only in what is being measured. Bump it deliberately
and record the bump in the run's REPORT.md.

`CUDA_ARCH=75` is pinned for two reasons, the second about measurement: a narrow list builds far
faster on a Graviton2, and a broad list can leave the device on a JIT'd PTX path whose warm-up
gets recorded as a property of the engine.

## THE NOTE: this rig and `gpu-ollama-g5g-2b-q4_0` share an engine

**This note also lives in the Ollama sibling's `CLAUDE.md`**, following the `~/gemma4-tips`
precedent — people read one rig rather than the tree. The two copies are the same facts with
the perspective swapped (each names itself as "this rig"), so **diff them when you change one**:
they are not byte-identical and a mechanical copy will read backwards. Verified 2026-09-02.

1. **Same engine.** Ollama ships `lib/ollama/libllama.so`, `libllama-server-impl.so` and
   `libmtmd.so`; its `llama/` directory holds only `compat` and `server` — it links llama.cpp,
   it does not fork it. The Gemma 4 graph both execute is upstream `src/models/gemma4.cpp`.
   **Slot 2 names the front end, not the decoder.**
2. **Not the same file.** Ollama re-containers the GGUF. Its `gemma4:e2b-it-qat` manifest holds
   a model blob of **3,349,514,112 B** against Google's **3,349,516,256** (−2,144) and a
   projector of **986,833,312** against **986,833,664** (−352); both digests differ. Tensors are
   presumed identical, the container provably is not. **So `-q4_0` is a weaker claim on that
   rig**, and its `MODEL_NAME` records the Ollama tag rather than a Hub id.
3. **Not the same template.** This rig uses the GGUF-embedded Google canonical template
   (published 2026-07-09, read out of the header). Ollama uses its own
   `model/renderers/gemma4.go`. Template differences show up as **quality** differences —
   never attribute them to the runtime.
4. **Not necessarily the same machine code.** Ollama auto-selects `cuda_v12` (native sm_75
   CUBIN) or `cuda_v13` (**PTX for every arch, native SASS for none** — it JITs at load) by
   driver version; `OLLAMA_LLM_LIBRARY` overrides. This rig is whatever `CUDA_ARCH` says.
   **Pin both rigs to the same choice or the comparison measures codegen, not runtime.**

And one thing that cannot be resolved from outside: Ollama also carries an experimental Go
engine at `x/models/gemma4/`, mirroring llama.cpp's `gemma4.cpp` / `gemma4-assistant.cpp`
file-for-file. Which one served a given request is not observable. **A number from the Ollama
rig names a front end.**

The practical asymmetry, and the reason both rigs exist: **Ollama hands you a working aarch64
CUDA binary with native sm_75 SASS; llama.cpp makes you compile one.** That is a real
operational difference, and having both is how it gets measured instead of argued about.

## There is no deploy step, and no payload

Structural, not a missing feature. The JAX and PyTorch siblings ship their own
`*_openai_server.py` over SSM because "our Gemma 4 port" is not a published artifact. **Here the
server IS the artifact** — built from a pinned upstream ref, fetching its own checkpoint through
`--hf-repo`/`--hf-file`. Cloud-init takes the instance from empty to serving in one round.

Two consequences worth stating because they retire hazards documented at length elsewhere:

- **The stale-payload trap cannot happen here.** The PyTorch sibling's `deploy_torch_server`
  ships whichever payload root it resolves, so deploying through the *registered* MCP server
  ships the previous `make skill` output. With no payload, there is nothing to ship stale.
- **The interpreter hazard has no analogue.** "The interpreter is discovered, not chosen" is the
  PyTorch rig's central hazard; here the artifact is a binary at a path the bootstrap chose, so
  install, verify and `ExecStart` cannot resolve to three different things.

`_PAYLOAD_FILES`, `_payload_root`, `_payload_digest` and `_payload_tar_b64` were **deleted**
rather than left returning empty — a payload digest of nothing would still have been reported as
a build id, and it would have been wrong. `_build_id()` returns `<ref>-sm<arch>` instead,
because the same ref built for a different arch is a different binary.

## Metrics are `llamacpp:*`, and this is the one house convention the rig cannot follow

Every sibling emits `tpu_jax_decode_tokens_per_second` — wrong as description, right as an
identifier, because every report in the family compares on that exact string. **This rig cannot**:
llama-server's exposition is fixed at `llamacpp:*` and there is no server of ours in the path to
rename it.

So the translation lives in `_decode_rate`, and reports quote the translated number:

```
llamacpp:tokens_predicted_total          / _seconds_total   -> decode   (THIS)
llamacpp:prompt_tokens_total             / _seconds_total   -> prefill  (never add it in)
```

`--metrics` is **off by default in llama.cpp**, so `_serve_argv` always passes it; without it
`/metrics` 404s and the gauge does not exist. Do not "fix" the naming by renaming llama.cpp's
metrics in a scraper — the raw names are what an operator sees in `curl /metrics`, and a shim
that disagrees with the wire is worse than one that admits the mapping.

**`verify_model_health` computes degeneracy locally**, because llama-server has no counter for
it. Weaker than the siblings' server-side verdict and labelled as such — but dropping the check
was not an option: the vLLM sibling on this exact silicon was measured answering
`': ok: ok: ok…'`, which is non-empty, 200, and worthless. **Never health-check by testing for a
non-empty response.**

## Benchmarking

`sweep.py` lives at the rig root, not inside a run directory — the JAX sibling copy-pasted its
sweep per run and got three independent sources of drift between numbers meant to be comparable.

```bash
python3 sweep.py --base http://<ip>:8000/v1 --out benchmarks/runs/<date>-<what>-g5g
```

**`--decode-source auto` resolves to `stream` here, and that is correct rather than degraded.**
`usage.decode_tokens_per_second` is a field our own servers invent; llama-server does not emit
it, so the harness falls through to the client-side inter-token statistic — which is exactly the
common statistic the 2026-08-31 rework exists to provide. Do not teach the harness to scrape
`/metrics`; that reintroduces a per-rig statistic under a shared name. Read `get_metrics`
alongside a sweep as a sanity check, never as the sweep's own number.

`benchmarks/README.md` and `serving-report.schema.json` are **synced copies** —
`make benchmarks-sync` at the monorepo root overwrites them. Edit the root originals.

Numbers you will be tempted to reuse and must not:

- **10.88 tok/s** — `gpu-pytorch-g5g-2b`, fp16, same silicon. The figure this rig is *compared
  against*, not one it inherits.
- **12.80 / 13.10 tok/s** — `gpu-jax-g5g-2b`, `ple0` and quantised best respectively.
- **43.1 / 44.24 tok/s** — the vLLM sibling. **Neither is a benchmark.** 43.1 is a single-sample
  smoke test with a 19-token prompt; 44.24 has no benchmark artifact anywhere in the tree. The
  number to compare against is the 2026-08-14 concurrency sweep in
  `gpu-vllm-g5g-2b/benchmarks/runs/2026-08-14-rust-frontend-g5g/`: c=1 TPOT 31.44 ms
  (~31.8 tok/s), c=8 168.33 tok/s.
- **Anything from `~/gemma4-tips`** — that tree duplicated its own artifacts and its directory
  names misattribute both model and chip.

## Engineering rules

Unchanged from the G5g siblings, and they are not negotiable per-rig:

- boto3 and the standard AWS credential provider chain — never shell out to the AWS CLI.
- SSM Run Command for remote administration; no inbound SSH rule, no private key.
- Require explicit subnet, security-group and instance-profile ids. Do not create broad network
  or IAM policy.
- Scope instance discovery to `ManagedBy=gpu-llamacpp-g5g-2b-q4_0`.
- HF tokens live in Secrets Manager and are fetched at boot into a root-only `EnvironmentFile`.
  **Never** in user data — instance metadata is readable by anything on the box. `set +x` wraps
  the fetch because the script runs under `set -x` and bash traces assignments *with their
  values*. Tests assert both. (This checkpoint is ungated; the control stays anyway.)
- Launches default to spot. **Termination is more expensive here than on the PyTorch sibling** —
  that rig loses a pip install and a model cache; this one loses a compile as well.
- Never hardcode an endpoint; `get_endpoint` resolves it from the instance.

## Commands

Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v`. Fully offline
— no AWS, no network, no GPU, no compiler.

**`tests/test_server.py` exists to assert on the RENDERED BOOTSTRAP.** When `gpu-pytorch-g5g-2b`
was forked it shipped **five fatal bugs** that all survived to the first launch — `import jax` on
a rig with no jax, an `ExecStart` naming a file not in the payload, three serve flags the server
did not define, a quoted pip spec holding two packages, and `torch.compile(backend="tpu")` on
CUDA. **89 offline tests passed throughout, because not one asserted on the bootstrap.** A fork
rewrites the parts you read and leaves the parts you execute. Those tests are the file's point.

`make lint` runs `ruff check server.py refresh_skill.py sweep.py make_report.py tests`, then
`bash -n` on four shell scripts. **A new top-level module is silently unlinted until it is added
to that list.**

`make skill` regenerates both snapshots (three files, not the siblings' six). `SKILL.md` is a
hand-written **source** in that tree — `refresh_skill.py` will not recreate it, so
`rm -rf .claude/skills` destroys it permanently.

There is no `make deploy` recipe and nothing for one to do.

## Agent-instruction files

`AGENTS.md` and `GEMINI.md` cover the same ground for other tools. There is no generator:
**`CLAUDE.md` is authoritative where they disagree**, and a convention change has to be applied
to all three by hand — plus, for THE NOTE above, to the Ollama sibling's three as well. Six
copies. Keep the note in as few places as the rules allow.
