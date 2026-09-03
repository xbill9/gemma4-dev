# CLAUDE.md — `gpu-ollama-g5g-2b-q4_0`

Serving **`gemma4:e2b-it-qat`** — Ollama's repackaging of Google's QAT **Q4_0** GGUF — through
the **Ollama daemon** on **AWS EC2 G5g**: a Graviton2 (aarch64) host with an **NVIDIA T4G** GPU
(Turing, SM 7.5, 15360 MiB).

Full rig: `server.py`, an MCP server, a skill, a plugin manifest, `tpu.env`.

> **STATUS: SERVED ITS FIRST TOKENS 2026-09-02** on `i-0db1233beb07f0da7` — see FIRST LIGHT
> below. `benchmarks/` is still empty: a seven-repeat smoke test is **not a benchmark run**, and
> the one number it produced is a server-side gauge that cannot be differenced against the
> siblings' figures. Every other number here is a property of the bundle, arithmetic from the
> artifact, or a sibling's measurement, and says which.

## FIRST LIGHT — 2026-09-02, and it found a bug the offline tests could not

**MEASURED on `i-0db1233beb07f0da7`**, `g5g.2xlarge` spot in `us-east-1a`, AMI
`ami-07a66fa2acbcfea88`, Ollama v0.33.2 / `cuda_v12`. Capacity took **four AZs** — 1d, 1b and
1c all returned `InsufficientInstanceCapacity` before 1a granted.

**The bug: `Error: $HOME is not defined`.** systemd gives a unit a minimal environment with no
`$HOME`, and `ollama serve` exits 1 on every start without it. `Restart=on-failure` crash-looped
the unit **13 times in ~2 minutes** while the bootstrap's daemon-ready poll sat on a dead port,
expired after 120s, and **carried on anyway** — so the visible failure would have been a
confusing `ollama pull` error two stages later.

Both are fixed and both are pinned by tests (`test_home_is_set_in_the_environment_file`,
`test_the_daemon_ready_wait_is_fatal`). Note what the class was: **`HOME` is the only variable
this rig sets that is not an `OLLAMA_*` name**, so the "validate against the daemon's own
variable list" test could not have caught it — that test checks for variables Ollama ignores,
not for one it requires.

### What the launch confirmed

- **The artifact is what the registry manifest said.** The pull fetched blobs `3646b4c147cd`
  (3.3 GB) and `58c187648007` (986 MB) — exactly the digests read off the manifest before any
  hardware existed. `ollama list` reports 4.3 GB, `parameter_size: 4.6B`,
  `quantization_level: Q4_0`, `family: gemma4`.
- **`nvidia-smi`: NVIDIA T4G, compute cap 7.5, 15360 MiB.** Both `cuda_v12` and `cuda_v13` are
  present in the bundle and the pin resolved.
- **The VRAM assertion works in both directions**: it correctly reported "no model loaded"
  before the first generate, then `size_vram == size` after.

### Decode: 95.48 tok/s median — and read the next paragraph before quoting it

Seven repeats, warm-up discarded, 128 tokens each:

```
95.24  95.45  95.47  95.48  95.84  95.96  96.12
median 95.48 tok/s   spread 0.9%
```

**This is `eval_count / eval_duration` — a SERVER-SIDE decode gauge that excludes prefill and
HTTP entirely.** The vLLM sibling's c=1 figure (~31.8 tok/s) is derived from `vllm bench serve`
TPOT, which *includes* streaming overhead. **Those are different statistics and this repo's own
rule forbids differencing them.** The comparable number needs `sweep.py`'s client-side stream
statistic, which has not been run. Do not publish "3x vLLM" off this.

What it does support, against this rig's own roofline: 95.48 against the Q4_0 ceiling of
197 tok/s is **48%** — close to the 52% the vLLM sibling reaches against its *fp16* ceiling of
61.4. Consistent with the launch-bound reading rather than against it: every runtime sits near
or below half its roofline, and Q4_0 wins on absolute throughput because **the ceiling itself
moved**, not because the efficiency did.

### An open question, stated rather than guessed

`/api/ps` reports `size_vram = 1,649,923,849` (1.65 GB) for a 3.35 GB model file, with
`size_vram == size` so Ollama considers it fully offloaded. `nvidia-smi` shows **2,791 MiB**
resident in total, the balance being CUDA context, KV and activations.

1.65 GB is suspiciously close to the **1.407 GB of non-PLE weights** this rig's own arithmetic
calls "streamed" — which would mean Ollama keeps the 1.927 GB PLE table off the device, exactly
as the gather-not-a-matmul analysis predicts. **That is inference, not measurement.** It has not
been confirmed and must not be written up as fact until someone checks where the PLE tensors
actually live.

## Why this rig exists

It and `gpu-llamacpp-g5g-2b-q4_0` **share an engine**. That is not a reason to have only one of
them — it is the reason to have both, because the differences that remain are exactly the ones a
front end imposes, and they are measurable.

**The practical asymmetry, and the honest headline: Ollama hands you a working aarch64 CUDA
binary with native sm_75 SASS. llama.cpp makes you compile one.** Verified 2026-09-02 —
llama.cpp release `b10760` publishes no Linux aarch64 CUDA asset at all (arm64 Linux is CPU and
Vulkan; every `-cuda-` Linux asset is x64), while Ollama's generic `ollama-linux-arm64.tar.zst`
is 1543 MB and carries `cuda_v12/` and `cuda_v13/` trees with `libggml-cuda.so`, cuBLAS and
cudart. So this rig has **no build step at all**, and the sibling's bootstrap is mostly a
compile.

## The CUDA variant is the single most important setting here

**MEASURED 2026-09-02** by walking the fatbin headers in both `libggml-cuda.so` and
decompressing one entry of each kind to identify them (`kind=1` decompresses to `.version 8.7 /
.target sm_50` PTX text; `kind=2` to an ELF with `e_machine=190`, EM_CUDA):

| Bundle | Native CUBIN | PTX |
| --- | --- | --- |
| `cuda_v12` | **sm_60, 61, 70, 75, 80, 86, 89, 90, 100, 120** | sm_50 … 120 |
| `cuda_v13` | **none at all** | sm_75, 80, 86, 87, 89, 90, 100, 103, 110, 120, 121 |

142 fatbin containers each, 0 malformed walks.

**Ollama picks between them by driver version**, and this rig's DLAMI carries a CUDA 13 driver —
so left alone it selects `cuda_v13`, which has **PTX for every architecture and native SASS for
none**, and JITs every kernel at load. That works; sm_75 PTX is present. But the warm-up then
belongs to the deployment and would be recorded as a property of the engine.

`OLLAMA_LLM_LIBRARY=cuda_v12` pins it. **The llama.cpp sibling builds native sm_75 too, so
pinning is what makes the pair comparable** — unset it and the A/B silently measures codegen
instead of runtime.

The bootstrap asserts the pinned variant is actually present in the bundle before anything
depends on it: `OLLAMA_LLM_LIBRARY` bypasses autodetection, it does **not** create a library
that is not there, and a bad pin makes the daemon fall back — possibly to CPU — with no error.

## Two more defaults that would quietly damage a measurement

Both from `envconfig/config.go`, read 2026-09-02:

- **`OLLAMA_CONTEXT_LENGTH` defaults to 0**, which means *"4k/32k/256k based on VRAM"*. Two
  instance sizes would silently get two different contexts. Pinned to 4096, matching the
  sibling's `--ctx-size`.
- **`OLLAMA_KEEP_ALIVE` defaults to 5 minutes.** A sweep that pauses longer than that reloads
  the model on its next request and records the reload as latency. Set to `-1` (forever).
  **No sibling has this knob** — llama-server, vLLM and our own JAX and PyTorch servers all hold
  weights for the life of the process. It is purely a daemon property, which is exactly the kind
  of thing this rig exists to surface.

`restart_ollama` drops the loaded model, so the first request after it pays a full load. Warm up.

## `ollama serve` takes no arguments, and that inverts the sibling's hazard

Every sibling has `_serve_argv`; this rig has **`_serve_env`**. The unit's `ExecStart` is bare
and all configuration is environment.

That is not cosmetic. **An unknown flag makes argparse exit 2 and the unit crash-loop** — loud,
and exactly the PyTorch fork's first-launch failure. **An unknown environment variable is
silently ignored by the daemon**, which then serves happily at its own defaults. A typo'd
`OLLAMA_CONTEXT_LEN` would produce a working rig at a VRAM-derived context and nothing anywhere
would say so.

So `_serve_env` is validated in tests against the daemon's own variable list, lifted from
`envconfig/config.go`. That test matters more here than the sibling's flag test, not less.

## The failure this rig ships silently is a CPU-resident model

**Ollama chooses its own offload and cannot be told to fail.** The sibling can demand
`--n-gpu-layers 999` and assert on its built binary's device list; there is no equivalent here —
`N_GPU_LAYERS` deliberately does not exist in this rig, and `OLLAMA_GPU_OVERHEAD` is a nudge
rather than a guarantee.

So the only honest check is **where the bytes ended up after a load**. `/api/ps` reports `size`
and `size_vram` per loaded model:

- `size_vram == 0` → the model is on the CPU. It serves correctly, several times slower, and
  logs nothing.
- `size_vram < size` → partial offload. Also working, also slow, also silent.

The bootstrap runs a real generate, then asserts on `size_vram` before writing `INSTALL_DONE`.
`verify_gpu_arch` re-checks it on demand, and `verify_model_health` flags a decode rate under
3 tok/s as consistent with CPU. `nvidia-smi` succeeding proves the driver works and nothing else.

## There is no `/metrics`, so this rig PROBES where every sibling SCRAPES

Verified against `server/routes.go`: Ollama registers `/`, `/api/version`, `/api/status`,
`/api/ps`, `/api/generate`, `/api/chat`, `/api/tags`, `/api/show`, the `/v1/*` OpenAI shims and
more — and **no Prometheus route of any kind**.

| | Instrument | What it reports |
| --- | --- | --- |
| llama.cpp, JAX, PyTorch siblings | scrape a counter | cumulative over every request since start |
| **this rig** | **one probe** | **one request, right now** |

That is a weaker instrument and is labelled as such everywhere it appears. The decode gauge
comes from `/api/generate`:

```
eval_count / eval_duration                <- decode   (THIS)
prompt_eval_count / prompt_eval_duration  <- prefill  (never add it in)
total_duration                            <- carries both, plus load and HTTP
```

**Every duration is a Go `time.Duration`, i.e. NANOSECONDS.** Dividing by `1e6` instead of `1e9`
gives a rate 1000x too low, which reads as a catastrophically broken rig rather than a unit bug.
Tests pin the unit, because nothing else would catch it.

**Note the health endpoint too: it is `GET /`**, returning the plain text "Ollama is running".
There is no `/health`, and `/` answers 200 as soon as the daemon binds — *before any model is
loaded* — so it is necessary and nowhere near sufficient. `verify_model_health` computes
degeneracy locally, because Ollama has no counter for it. **Never health-check by testing for a
non-empty response**: the vLLM sibling on this exact silicon was measured answering
`': ok: ok: ok…'`, which is non-empty, 200, and worthless.

## THE NOTE: this rig and `gpu-llamacpp-g5g-2b-q4_0` share an engine

**This note also lives in the llama.cpp sibling's `CLAUDE.md`**, following the `~/gemma4-tips`
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
   presumed identical, the container provably is not. **So `-q4_0` is a weaker claim on THIS
   rig**, and its `MODEL_NAME` records the Ollama tag rather than a Hub id.
3. **Not the same template.** This rig uses Ollama's own `model/renderers/gemma4.go`.
   The llama.cpp sibling uses the GGUF-embedded Google canonical template (published
   2026-07-09, read out of the header). Template differences show up as **quality** differences —
   never attribute them to the runtime.
4. **Not necessarily the same machine code.** Ollama auto-selects `cuda_v12` (native sm_75
   CUBIN) or `cuda_v13` (**PTX for every arch, native SASS for none** — it JITs at load) by
   driver version. **This rig pins `cuda_v12`**; the sibling builds native sm_75.
   **Pin both rigs to the same choice or the comparison measures codegen, not runtime.**

And one thing that cannot be resolved from outside: Ollama also carries an experimental Go
engine at `x/models/gemma4/`, mirroring llama.cpp's `gemma4.cpp` / `gemma4-assistant.cpp`
file-for-file. Which one served a given request is not observable. **A number from the Ollama
rig names a front end.**

The practical asymmetry, and the reason both rigs exist: **Ollama hands you a working aarch64
CUDA binary with native sm_75 SASS; llama.cpp makes you compile one.** That is a real
operational difference, and having both is how it gets measured instead of argued about.

## What Q4_0 buys, and the number you must not derive from it

Identical to the sibling's, because it is a property of the weights rather than the front end.
Derived from the GGUF's tensor table against the PyTorch sibling's **measured** fp16:

| | fp16 (measured) | Q4_0 (derived) |
| --- | ---: | ---: |
| Streamed per decode step | 4.514 GB | **1.407 GB** |
| Resident | 10.209 GB | **3.35 GB** |
| Bandwidth ceiling @ 277 GB/s | 61.4 tok/s | 197 tok/s |

**Do not turn 197 into a prediction.** Decode at `B=1` on this silicon is **launch-bound**, not
bandwidth-bound. `QUANTIZATION.md` has the two controlled experiments: `ple_bits=4` removed
3.505 GB for **0.0%**, `int8_lm_head` removed 11.9% of streamed bytes for **+2.3%**.

**The win is residency**, and residency is what pays for batching — 7.84x measured on the
PyTorch sibling. Here that means raising `PARALLEL_SLOTS` (`OLLAMA_NUM_PARALLEL`), and only in a
run whose REPORT.md says so.

## Benchmarking

```bash
python3 sweep.py --base http://<ip>:8000/v1 --out benchmarks/runs/<date>-<what>-g5g
```

`--decode-source auto` resolves to `stream` here, as on the llama.cpp sibling and for the same
reason: `usage.decode_tokens_per_second` is a field our own servers invent, and Ollama's OpenAI
shim does not emit it. The client-side inter-token statistic is the cross-rig one. Read
`get_metrics` alongside a sweep as a sanity check, never as the sweep's own number — and
remember it is a probe, so it cannot report what happened *during* the sweep.

Numbers you will be tempted to reuse and must not: **10.88** (PyTorch sibling, fp16),
**12.80 / 13.10** (JAX sibling), **43.1 / 44.24** (neither is a benchmark — 43.1 is a
single-sample smoke test, 44.24 has no artifact anywhere in the tree). The vLLM figure to
compare against is `gpu-vllm-g5g-2b/benchmarks/runs/2026-08-14-rust-frontend-g5g/`: c=1 TPOT
31.44 ms (~31.8 tok/s), c=8 168.33 tok/s. And **anything from `~/gemma4-tips`** is
misattributed.

## Engineering rules

- boto3 and the standard AWS credential provider chain — never shell out to the AWS CLI.
- SSM Run Command for remote administration; no inbound SSH rule, no private key.
- Require explicit subnet, security-group and instance-profile ids.
- Scope instance discovery to `ManagedBy=gpu-ollama-g5g-2b-q4_0`.
- HF tokens live in Secrets Manager, fetched at boot into a root-only `EnvironmentFile`, wrapped
  in `set +x`. **Inert on this rig** — Ollama pulls from its own registry, not Hugging Face —
  but kept, because it costs nothing and survives a repoint at a Modelfile importing a Hub GGUF.
- Launches default to spot. **Termination is cheaper here than on the llama.cpp sibling**: that
  rig loses a compile, this one loses a 1.5 GB download and a 4.3 GB pull.
- Never hardcode an endpoint; `get_endpoint` resolves it from the instance.

## Commands

Tests are **`unittest`, never pytest**: `python3 -m unittest discover -s tests -v` (53 tests,
all passing as of 2026-09-02). Fully offline — no AWS, no network, no GPU.

`tests/test_server.py` asserts on the **rendered bootstrap**, for the reason the sibling's does:
the PyTorch fork shipped five fatal bugs that all survived to first launch while 89 offline tests
passed, because not one asserted on the bootstrap.

`make lint` runs `ruff check server.py refresh_skill.py sweep.py make_report.py tests`, then
`bash -n` on four shell scripts. A new top-level module is silently unlinted until added.

`make skill` regenerates both snapshots. `SKILL.md` is a hand-written **source** — `rm -rf
.claude/skills` destroys it permanently. There is no `make deploy` and nothing for one to do.

## Agent-instruction files

`AGENTS.md` and `GEMINI.md` cover the same ground for other tools. **`CLAUDE.md` is authoritative
where they disagree.** No generator: a convention change lands in all three by hand, and THE NOTE
above also lives in the llama.cpp sibling's three copies. Six copies total.
