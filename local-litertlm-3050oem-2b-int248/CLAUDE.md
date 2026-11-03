# CLAUDE.md — local-litertlm-3050oem-2b-int248

Gemma 4 E2B served by **LiteRT-LM** (Google's on-device runtime) on the RTX 3050 OEM in the
desktop `gpu`, from Google's own `.litertlm` build. The first `litertlm` rig. No MCP server, no
skill, no plugin: `tpu.env`, a Makefile, `render_config.py`, measurements. Same card as the
`local-llamacpp-3050oem-*` and `local-pytorch-3050oem-*` rigs; they do not run beside each other.

## Read first

- **`tpu.env` decides.** The directory name is documentation (root `NAMING.md`).
- **The GPU path is WebGPU (Dawn) on Vulkan, not CUDA.** `--verbose` logs
  `Selected adapter: NVIDIA GeForce RTX 3050 OEM, arch=ampere, vendor=nvidia, backend=Vulkan`.
  It needs `libvulkan1` and `/etc/vulkan/icd.d/nvidia_icd.json`. `CUDA_VISIBLE_DEVICES` does not
  select the card. The intel/freedreno `renderD128: Permission denied` lines at start-up are Mesa
  ICDs probing and are harmless.
- **Install into the system python3's user site**, never a virtualenv:
  `make install` (`pip install --user --break-system-packages -r requirements.txt`). Debian has no
  `python3-litert-lm`. litert-lm needs **protobuf ≥ 6.31.1**; Debian's 3.21.12 fails at import with
  `cannot import name 'runtime_version'`, so a newer protobuf in `~/.local` shadows it.
- **Per-rig state lives in `LITERT_LM_DIR`** (`~/.cache/local-litertlm-3050oem-2b-int248`), not
  `~/.litert-lm`. `make import` hard-links the model there; the engine writes its compiled GPU
  caches beside it (778 MB of weights cache for this model).
- **`litert-lm serve` takes no per-model flags.** Backend, cache and speculative decoding come
  from a `config.json` keyed by model id; `make serve` renders `run/config.json` from `tpu.env`
  with `render_config.py`. Override on the command line: `make serve SPECULATIVE_DECODING=off`.

## The model file

`litert-community/gemma-4-E2B-it-litert-lm`, file `gemma-4-E2B-it.litertlm` (2,588,147,712 B,
SHA-256 in `tpu.env`). It holds separate graphs for prefill/decode, the embedding table, the
per-layer embedding table, vision, audio and a 44 MB **MTP drafter**. The same repo's
`gemma-4-E2B-it-gpu.litertlm` is a different artifact: text only, no drafter, and stored
differently (int4 linears, int8 embeddings, a 4-bit per-layer table, no int2), so it would be a
different rig. `litert-lm describe` on it reports `Speculative Decoding: NO`.

**Slot 5 `int248` is measured, not quoted**
(`benchmarks/runs/2026-11-02-first-light-3050oem/encoding.txt`): int2 for the wide MLPs of the 20
KV-shared layers and the shared 262144 × 1536 embedding/output table; int4 for every attention
projection, the MLPs of the first 15 layers and the 35 per-layer embedding tables; int8 for the
per-layer embedding gates and projections. **Every quantized tensor has one scale per output row**
(symmetric, zero points 0). Google calls it the "Gemma-4 mobile quantization scheme". It is not
the QAT Q4_0 grid the xbill9 repacks keep: those have one scale per group of 32, which this
format cannot hold, so their values cannot be written into this file. The sister rig
`local-litertlm-3050oem-2b-q4w4a16emb4` builds its own `.litertlm` with int4 blocks of 32 and writes the
repack's exact codes into that.

## Server quirks

- **`max_tokens` is ignored**; the server reads only `max_completion_tokens`. A request with
  `max_tokens: 256` runs to the end of the reply (1,100+ tokens on the A/B prompts).
- **Constrained decoding is on for every chat request, and it costs tool calls.** With tools its grammar
  forces arguments into the schema's declared order; the model learned alphabetical order from the
  template and drops optional arguments. BFCL 0.825 → 0.890 here, and the exact QAT build 0.8625 → 0.9125
  (level with llama.cpp), with it off via `serve_noconstraint.py`. Off, about 1 reply in 400 is a
  malformed call that the server answers with HTTP 500
  (`benchmarks/runs/2026-11-03-tools-constraint-3050oem`).
- No `timings` in responses. Decode speed is measured on the stream (see `spec_ab.py`); the
  drafter's accepted runs arrive as multi-token chunks.
- The first request loads the engine (about 4 s with the GPU cache present, 10 s without).
- `litert-lm benchmark` decode with the drafter on is not a chat speed: acceptance depends on the
  text, and the benchmark's text is synthetic.

## Shell gotcha

`pkill -f "litert-lm serve"` run from a shell matches that shell's own command line and kills
it. Match the python process: `pgrep -f '^/usr/bin/python3 .*litert-lm serve'`.

## Commands

`make install | download | import | serve | status | query | bench | test | lint`. Tests are
`unittest` (`make test`), offline.
