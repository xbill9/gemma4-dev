# CLAUDE.md — local-mistralrs-3050oem-2b-q4_0

Gemma 4 E2B served by **mistral.rs v0.9.5** (`mistralrs serve`, Rust/Candle, built here with CUDA and
flash-attn for sm_86) on the RTX 3050 OEM in the desktop `gpu`, from the same exact QAT Q4_0 GGUF as
`local-llamacpp-3050oem-2b-q4_0` (same file path), so the pair is an engine A/B on identical bytes.
Created 2026-11-03 for the 3050 serving-engine comparison. No MCP server, no skill, no plugin: `tpu.env`,
`serve.py`, a Makefile, measurements. Same card as the other `local-*-3050oem-*` rigs; they do not run
beside each other.

## Read first

- **`tpu.env` decides**, and `serve.py` turns it into the `mistralrs serve` command (`make command`).
- **Build:** `make install` = `cargo install --locked --path mistralrs-cli --features cuda,flash-attn` with
  `CUDA_COMPUTE_CAP=86` into `~/.local/mistralrs-v0.9.5/` (106 minutes on the i7-5820K, 2026-11-03).
  `cudnn` is left out: cuDNN exists on this host only inside a pip wheel. Features are comma-separated in
  `tpu.env`; Make keeps quotes from an included file.
- **It loads a projector for every gemma4 GGUF** (`--mmproj`, Google's QAT mmproj, sha in `tpu.env`), and
  after load holds 4,104 MiB on the card: weights, the per-layer table and the image and audio encoders.
  llama.cpp keeps the per-layer table in CPU RAM (2,809 MiB on the card).
- **Size the cache with `--kv-cache`, not `--gpu-memory-utilization`.** With a utilization goal the engine
  reserves encoder memory for every concurrent request (about 300 MiB each; 64 requests want 19 GB, 10 or
  more do not load) and then refuses every request with `503 service_unavailable` ("Memory pressure
  prevented forward admission"), even at 1 request. A fixed cache serves: 4 requests with 8,192 tokens;
  with the drafter, 4 requests with 4,096. 12,288 does not load. `--paged-attn off` also serves, but the
  drafter needs paged attention.
- **`/health` answers before requests are admitted**: wait for a first reply, not for `/health`.
- **Port 8020**, model id `default`. Prefix cache off (`PREFIX_CACHE_N=0`).
- **Drafter:** `--mtp-model` takes Google's QAT assistant as bf16 safetensors (`MTP_MODEL`, empty = none).

## Commands

`make src | install | command | serve | status | query | test | lint`. Tests are `unittest`, offline.
