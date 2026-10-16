# CLAUDE.md — local-llamacpp-3050oem-2b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

`llama-server` from `ggml-org/llama.cpp`, driven directly, serving the exact
Q4_0 GGUF of Gemma 4 E2B-it (`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`, v2,
SHA-256 `419db9a6…`) off one **RTX 3050 OEM** in the desktop whose hostname is
`gpu`. One process, one GGUF file named on the command line. No control plane,
nothing provisioned, Ctrl-C is a complete teardown.

**Forked 2026-10-13 from `local-llamacpp-1650ti-2b-q4_0`.** Slot 3 is the only
slot that differs: same runtime, same GGUF by hash, same serving flags. That makes
the pair a card-against-card comparison on identical weights, but **not a
controlled A/B**: different hosts (i7-5820K desktop here, i7-10750H laptop
there), different llama.cpp commits (`77c4540ce` here, `f95b0d9` there),
different days. Quote a ratio between them as indicative only.

**Slot 3 is `3050oem`**, per `NAMING.md`'s rule for consumer GPUs: series digits
plus the suffix the driver reports. The OEM part is a different configuration
from the retail RTX 3050 8 GB, so a bare `3050` would name a card this is not.

The parent's control machinery came along. `attest.py` reads the serving process
out of `/proc` and still earns its place: it catches a CPU-only binary or another
rig's `llama-server` on port 8080. There is **no CPU twin on this host yet**;
`local-llamacpp-cpu-2b-q4_0` is pinned to the parent's laptop.

## Host — MEASURED 2026-10-13

| | |
| :--- | :--- |
| GPU | NVIDIA GeForce **RTX 3050 OEM**, PCI `10de:2508` (GA106), **Ampere, compute capability 8.6, tensor cores**, 8192 MiB, 120 W limit, SM max 2100 MHz |
| Driver | 615.78.08 |
| CPU | Intel i7-5820K (Haswell-E), 6C/12T, AVX2, no AVX-512 |
| Board / RAM | ASUS X99-A, 46 GiB, 12 GiB swap |
| OS | Debian sid, kernel 7.2.10, gcc 16.2.0, CUDA 13.4 (V13.4.92) |

## Build

`make build`: `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86`, `GGML_NATIVE=ON`,
Release, `FORCE_MMQ` off, into `~/llama.cpp/build/`. Two things about this host:

- **`nvcc` is not on PATH.** The toolkit is at `/usr/local/cuda`; run
  `export PATH=/usr/local/cuda/bin:$PATH` before `make build`. The target says so
  if you forget.
- **Rebuilding the drafter from Google's bf16** (`gguf_exact.py` in
  `benchmarks/runs/2026-10-14-drafter-q4exact-3050oem/`) needs `python3-torch`,
  `python3-safetensors`, `python3-sentencepiece`, `python3-transformers` (apt).
  `make draft` only downloads.
- **System python has no pip by default.** `cmake`, `python3-pip`,
  `python3-huggingface-hub`, `python3-dotenv`, `python3-httpx` and `ruff` came
  from apt; `mcp==2.2.0` from `pip install --break-system-packages`.

The 2026-10-13 build ran at `-j10` (the `BUILD_JOBS` default here) and took
10m24. Never pass a bare `-j`: it starts every CUDA translation unit at once,
which is what OOM'd the parent's 15 GiB laptop.

`make download` fetches the GGUF to `MODEL_PATH` and checks `MODEL_SHA256`.

## Measurements

`benchmarks/runs/2026-10-13-first-light-3050oem/` — llama-bench pp512/tg128,
r=5, each cell gated on the GPU at ≤40 °C.

| | pp512 | tg128 |
| --- | ---: | ---: |
| serving config (`-fa 1`, f16 KV) | 3859.96 | **116.14** |
| `-fa 0` | 3408.19 | 108.46 |
| q8_0 KV | 3708.48 | 107.57 |

- Flash attention: 1.133x prefill, 1.071x decode. On.
- q8_0 KV: 0.961x prefill, 0.926x decode. Off, as on the parent.
- Threads 4 against 6: within 0.5%. Decode runs on the card.
- The repeated cell lands within 0.6% of the first; the card held 1890–1905 MHz
  and never passed 44 °C.
- **1510 MiB of 8192 VRAM** serving at `-c 8192`. `make info`: 1.20 GiB of
  weights must be resident, because `per_layer_token_embd` (1.321 GB, 51% of the
  file) is `TENSOR_READ_LAZY` and stays in the mmap.

Against the parent's tg128 81.59–81.92 and pp512 344–345 on the same file:
about **1.42x decode and 11x prefill**, indicative only (see above).

**Concurrency, `benchmarks/runs/2026-10-13-concurrency-3050oem/`** —
`sweep.py`, c=1–64, 5 repeats, one server at `-c 65536 --parallel 64`
(1024 tokens per slot, 2612 MiB):

| | c=1 | c=16 | c=64 |
| --- | ---: | ---: | ---: |
| 512 in / 128 out, aggregate tok/s | 93.28 | 316.07 | 358.05 (3.84x) |
| 128 in / 512 out, aggregate tok/s | 104.74 | 505.42 | 618.12 (5.90x) |

- Prefill still serves one request at a time: TTFT about doubles per doubling
  of c. The card prefills fast enough that decode batching shows through anyway;
  the 1650 Ti's 512/128 output stalled at 1.47x c=1.
- 512/128 spreads are wide at c=8–32 (15–26%, low tail). 128/512 stays under 5%.
- Clock pinned at 1890 MHz, 68 °C peak; prompt cache 0 tokens.

**`per_layer_token_embd` is ON THE CARD since 2026-10-13**
(`benchmarks/runs/2026-10-13-ple-offload-3050oem/`): `-ot` via
`OVERRIDE_TENSORS` in `tpu.env`. Prefill 1.036x at pp512 and 1.024x at pp2048,
decode 1.001x, for 1260 MiB (serving total 2770 MiB of 8192). The 1650 Ti
measured no gain from the same move. `-ub 2048` adds nothing. The first-light
and concurrency runs above were made with the table on the host.

**Context is 131,072 since 2026-10-13**, the trained maximum, because this rig
serves one user (`benchmarks/runs/2026-10-13-long-context-3050oem/`): 3610 MiB
in all, short chats 0.997x. Generation with 8k / 32k / 65k / 120k tokens already
in the conversation: 107 / 91 / 75 / 58 tok/s.

**Speculative decoding is ON since 2026-10-13**
(`benchmarks/runs/2026-10-13-spec-mtp-3050oem/`): Google's E2B QAT assistant
drafter, converted to Q8_0 GGUF by `make draft`, run as `--spec-type draft-mtp
--spec-draft-n-max 2`. Generation 1.39x overall greedy (1.20x story .. 1.85x
code), 1.36x sampled; 3/4/6 drafted tokens are worse and 6 is a net loss. Costs
230 MiB and 3.6% of prompt speed. **Greedy output differs from serving without
it** (each arm deterministic; they part at a near-tie word), so any benchmark
or reference comparison that needs byte-identical greedy text must send `cache_prompt: false`
(the prompt cache changes greedy text too, `benchmarks/runs/2026-10-14-drafter-divergence-3050oem/`) and empty
`SPEC_DRAFT_MODEL` first. The first-light, concurrency and context runs above
were measured without the drafter.

**The exact Q4_0 drafter replaced the Q8_0 conversion on 2026-10-14**
(`benchmarks/runs/2026-10-14-drafter-q4exact-3050oem/`): 59 MB against 98, closer to
Google's weights, same speed (151.1 against 150.1 tok/s), same acceptance, GSM8K and
BFCL level with each other and with no drafter at `--parallel 8`, 40 MiB less VRAM.
`make draft` now downloads it from Hugging Face instead of converting.

**The drafter costs nothing in answer quality as the rig serves** (`-c 131072 --parallel 1`,
one request at a time, `benchmarks/runs/2026-10-14-accuracy-fullctx-3050oem/`): GSM8K 0.8969
with it against 0.9014 without, BFCL 0.9250 against 0.9225, every difference within range of
zero and of the `--parallel 8` runs; GSM8K finishes 1.76x faster with it. **Turning the prompt
cache off changes no score** (`benchmarks/runs/2026-10-14-accuracy-cache-3050oem/`), so
accuracy runs at the server default stand. **With several users at once the drafter
pays clearly only for one request at a time** on mixed chat prompts
(`benchmarks/runs/2026-10-15-concurrency-mixed-3050oem/`): 0.92–1.17x at 2–8 requests, 0.79–0.87x
at 32–64, time to first token 6–48% lower with it. Thinking on lifts it to 1.04–1.31x at 2–8
(`benchmarks/runs/2026-10-15-concurrency-thinking-3050oem/`); 1 drafted token loses less at 32–64
(0.84–0.92x) and matches 2 for one request (`benchmarks/runs/2026-10-15-concurrency-draft1-3050oem/`).
**KV cache layout for multi-request runs:** per-slot caches (the default, used by the concurrency
runs above) halved decode speed at random on short requests; a shared cache (`-kvu`) avoids that
but slows as the total context held grows: 0.55–0.70x per-slot at 64 requests of up to 640 tokens
(`benchmarks/runs/2026-10-15-concurrency-drafter-kvu-3050oem/`), 1.2–1.6x on requests under 300.
**With 1,000–3,500-token prompts the drafter pays up to 16 requests**
(`benchmarks/runs/2026-10-15-concurrency-longmixed-3050oem/`, `-c 131072 --parallel 32`): 1.26–1.34x for
one request, 1.02–1.14x at 2–16, 0.96–1.04x at 32, acceptance 0.56–0.62. Per-slot caches are the faster
layout there (shared 0.49–0.53x at 32 requests, time to first token up to 2x) and showed no random slowdown.

**Answer quality against E4B**: see `local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-10-13-quality-e2b-vs-e4b-3050oem/`.
This build scored suite 0.683 (E2B bf16 on TPU: 0.683), GSM8K 0.898 (bf16: 0.910),
BFCL 0.925 (bf16: 0.927). E4B is +4.8 suite and +3.5 GSM8K points, level on BFCL.

**Not measured:** a `sweep.py` context sweep, `GGML_CUDA_FORCE_MMQ`, and a CPU-only arm here.

## Inherited and still true here

- **Never pass `--no-mmap`.** Even with `per_layer_token_embd` on the card,
  `token_embd`'s 216 MiB input copy is read from the mmap. A test asserts it.
- **Never lower `N_GPU_LAYERS` "to be safe."** Full offload uses under a fifth of
  the card.
- Gemma 4 reasons, and llama.cpp routes the thinking block to
  `reasoning_content`; `content` stays empty until it closes. `REASONING=off` is
  the server default, `query_model` defaults to `max_tokens` 1024, and a test
  keeps it ≥512. A request can still turn thinking on with
  `chat_template_kwargs`, which `sweep.py` does.
- Raw `/v1/completions` returns an empty completion on `-it` models; use
  `/v1/chat/completions`.
- `sweep.py`: `decode_tps` counts inter-chunk gaps, not tokens; check the prompt
  cache was defeated via `/metrics` (`llamacpp:prompt_tokens_cached_total`);
  llama.cpp splits `--ctx-size` across `--parallel` slots, so a context sweep
  and a concurrency sweep need different servers. Never feed `/metrics` into
  the decode column.
- `start_model_server` spawns with `subprocess.Popen`, because asyncio's
  subprocess transport kills a live child when it is torn down.
- Port 8080 is shared with every `local-llamacpp-*` rig. Only one serves at a
  time on a host.

## Conventions

- Tests are `unittest`, never pytest: `python3 -m unittest discover -s tests -v`.
- Every subprocess call goes through `run_command(cmd: list[str])` using
  `asyncio.create_subprocess_exec`. **Never `shell=True`.**
- MCP tools are `async def` returning markdown strings with emoji status
  prefixes (`✅`, `❌`, `📡`).
- `Optional[str]`, not `X | None`.
- Use the system `python3` and install into it. **Never create a virtualenv.**
- `tpu.env` is the source of truth and is committed. Never add `*.env` to
  `.gitignore`.

## Canonical root references

`@MODELS.md`, `@HARDWARE.md`, `@QUANTIZATION.md`, `@NAMING.md`,
`@RIG-ANALYSIS.md`. Read them before deriving their numbers here, and correct them
there.
