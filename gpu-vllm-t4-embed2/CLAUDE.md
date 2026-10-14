# CLAUDE.md — gpu-vllm-t4-embed2

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

**EmbeddingGemma 2 (`google/embeddinggemma-2`) under vLLM's pooling runner, on the
Tesla T4 already attached to this Compute Engine VM.** It is the **first embedding
model in this tree**: `/v1/embeddings`, no chat completions, no KV cache, no
tokens generated. Forked 2026-10-07 from `gpu-vllm-t4-2b`, with which it shares the
GPU, the host and port 8000. Only one of the T4 rigs can serve at a time.

**STATUS 2026-10-07: serving and measured once.** Read
`benchmarks/runs/2026-10-07-first-serve-t4/REPORT.md` before quoting a number.
Headline: **1−cos ≤ 7.0e-9 against sentence-transformers** at float32 with text
inputs. Throughput is about 3.9k input tok/s at ~500-token inputs and 2.2k at ~2k, and
**one stream saturates the GPU**, so concurrency adds latency but no throughput. Images
serve and rank correctly, but image parity is unmeasured.

**Slot 4 is `embed2`**, because the checkpoint has no Gemma 4 size. One file loads as
271M parameters (text) up to 744M (with the 167.8M vision and 305.6M audio towers). See
`@NAMING.md` slot 4. **It is not an A/B twin of anything.** No other rig here serves
this checkpoint on this GPU, so never difference a number from it against a
generative T4 rig.

## Three things that break sibling assumptions

**1. float32, and the reason is not Turing's.** Google says float16 is incompatible
with this model. Turing has no bf16 datapath, and vLLM casts bf16 to fp16 on SM 7.5
*without failing*. So `--dtype bfloat16` here would silently serve the forbidden dtype.
float32 doubles the weights (1.03 GiB text only, 1.63 GiB with vision), which is
irrelevant against 15 GiB.

**2. A different Triton kernel breaks, and the siblings' patch does not touch it.**
Encoder-only attention runs through
`triton_prefill_attention.context_attention_fwd`, not the unified decoder kernel that
`gpu-vllm-t4-2b` clamps. Here is what it needs at float32 on 64 KiB per block:

| head_dim | stock | BLOCK 16 fp32 | BLOCK 16, q/k/v fp16 |
| ---: | :--- | :--- | :--- |
| 256 (sliding) | 69,632 B, fails | fits, err 5.7e-6 | — |
| 512 (full) | fails | 66,560 B, **still fails** | fits, rel. err 4.6e-4 |

16 is `tl.dot`'s floor, so no fp32 tile holds a 512-wide layer on this part.
`patch_triton_encoder_turing.py` clamps BLOCK to 16 and runs only the 512-wide calls
in fp16 with fp32 accumulation. q, k and v are RMS-normed immediately before
attention (`q_norm`, `k_norm`, weightless `v_norm`), and the end-to-end parity shows
no measurable cost. **FlexAttention is not the alternative**: it failed on a
CUDA-graph bug in its mask builder, then on the same 64 KiB at the 16 floor (REPORT
§"What it took").

**3. Its own `PYTHONUSERBASE`, `/opt1/pyuser-embed2`.** `/opt1/pyuser` holds the
generative rigs' vLLM 0.29.0 with their decoder clamp applied. This model needs vLLM
support that is in no release (merged 2026-10-06 as `02b83919aa`) and transformers
≥ 5.19.0. Upgrading the shared user base would break both siblings and silently
revert their patch. A relocated user base is not a virtualenv (root `CLAUDE.md`).

## Installing: two traps

- **pip cannot resolve it in reasonable time.** Its resolver spent 20+ minutes
  backtracking through every `filelock` and `httpx2` release before it was killed.
  `uv pip compile` resolved the same request in seconds. Install from the lock
  (`benchmarks/runs/2026-10-07-first-serve-t4/evidence/requirements-lock.txt`)
  with `pip --no-deps`. `check_host_capacity` prints the command.
- **The nightly index lists only its newest wheel**, so a `vllm==<nightly>` pin
  stops resolving within hours. The lock and `VLLM_PIP_SPEC` pin the per-commit
  wheel URL.

## Serving facts

- `--runner pooling`. There is no KV cache, because encoder-only layers have no KV
  spec, and the engine showed 1,979 MiB in use with vision on.
  `GPU_MEMORY_UTILIZATION`, `KV_CACHE_DTYPE` and `MAX_NUM_SEQS` are deliberately
  absent.
- `--hf-overrides '{"matryoshka_dimensions":[768,512,256,128]}'`. `config.json`
  declares none, so without it vLLM rejects every `dimensions` request. The explicit
  list makes an untrained size a 400.
- `--limit-mm-per-prompt {"image":1,"audio":0,"video":0}`. Audio needs the
  `vllm[audio]` extras, which are not installed. Set `image: 0` for text only: 96 s
  to healthy against 177–202 s, and about 0.6 GiB less.
- **vLLM does not add the task prefixes.** `/v1/embeddings` embeds exactly what it
  is sent. `embed_text`, `rank_documents` and `verify_embeddings` prepend
  `task: search result | query: ` and `title: none | text: `. The prefixes live in
  `server.py`, not `tpu.env`, because the Makefile's `include` + `export` would put
  literal quotes into the environment.
- **`make serve` works here.** `start_vllm_server` uses `subprocess.Popen` in a new
  session. The sibling's `asyncio.create_subprocess_exec` child dies when the
  one-shot interpreter exits. A test pins this.
- **Host RAM is the term to watch.** 6.3 of 7.4 GB was in use while serving with
  vision, and a sentence-transformers reference loaded alongside was OOM-killed. Run
  parity in two phases (`scripts/parity.py`, `PHASE=vllm` then `PHASE=ref`).

## Tool order

`check_host_capacity` → `verify_gpu_arch` → `apply_encoder_patch` →
`verify_encoder_patch` → `start_vllm_server` → `verify_embeddings`.
`verify_gpu_arch` also asks whether the installed vLLM registers
`EmbeddingGemma2Model` and whether transformers ships `embedding_gemma2`. That is
the check probing `/opt1/pyuser` by mistake would fail. `start_vllm_server` fails
closed: it starts only once the clamp is *positively* confirmed.

## Conventions

Carried from `gpu-vllm-t4-2b` and unchanged:

- unittest (`python3 -m unittest discover -s tests -v`).
- `run_command`/`_probe` for short-lived subprocesses; the engine is the one `Popen`.
- Never `shell=True`.
- `Optional[str]`.
- `tpu.env` is committed and is the source of truth.
- Measure disk per path, never with a single `df`.
- `MemAvailable`, never `MemFree`.

Registration lives in four places (`.mcp.json` is gitignored;
`.claude-plugin/plugin.json`, `.codex/config.toml`,
`.claude/settings.local.json`), and a test checks that the committed three agree. Each
tool gated in `.codex/config.toml` must exist, and a test checks that too.

## Canonical root references

`@MODELS.md` has the EmbeddingGemma 2 checkpoint facts. `@HARDWARE.md` has the T4
(its "T4G" section is the chip). Read `@NAMING.md` (slot 4, `embed2`) and
`@RIG-ANALYSIS.md` as well. Correct facts there, not here.
