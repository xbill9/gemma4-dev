# CLAUDE.md — tokenlab-sglang-amd-31b

Gemma 4 31B (`Gemma-4-31B-IT`) through **AMD Token Lab**, a hosted OpenAI-compatible API
(`https://api.tokenlab.amd.com/v1`, key from `notebooks.amd.com/tokenlab`). Read the monorepo
`CLAUDE.md` first; this file covers what differs.

## What this rig is, and is not

- **Someone else's server.** Nothing here provisions, starts or stops anything. There is no
  MCP server, no skill, no plugin; `make install` is a no-op. The rig is `tpu.env`, the key,
  a local chat page and a terminal client.
- **Slot 2 is `sglang`, measured, although the rig was asked for as vLLM.** `/v1/models`
  describes Gemma-4-31B-IT as a "Dynamic sglang-router service" and every reply carries
  SGLang's `matched_stop` field. Other Token Lab models are "vllm-router" services. If the
  description changes, re-check before trusting the name.
- **Slot 3 is `amd` because the GPU is not disclosed** ("AMD Instinct and Radeon"). Never
  difference a number from here against `gpu-vllm-mi300x-31b` and read it as a hardware or
  engine result: different runtime, unknown hardware, and the service is shared.
- **Timings include other people's load.** The model has a shared limit of 128 requests at
  once (HTTP 429 `model_concurrency_rate_limit_exceeded`, `Retry-After: 1`), and a stream can
  sit on `: ping` keep-alives before its first token. Record time to headers and time to the
  first token separately (`probe.py` does).

## Files

- `tpu.env` — endpoint, model, retry settings, UI port. Committed.
- `.env` — `TOKENLAB_API_KEY`, mode 0600, gitignored. A real environment variable wins.
  The key has a 30-day access period and a points budget; 429 when points run out, 403
  when the period ends.
- `tokenlab.py` — config loading and `open_api()`, which attaches the key and retries
  429/502/503/504 before the reply starts, honouring `Retry-After`.
- `chat_ui.py` — the browser chat page, forked from
  `~/tpu-inference/tpu-vllm-v5e1-2b/chat_ui.py`. Binds 127.0.0.1; the key stays in the proxy.
  The TPU page's `/metrics` engine and load panels are replaced by the model card from
  `/v1/models` and this session's counts (requests, retries, tokens, `usage.cost` points).
  Adds a thinking toggle, top-p instead of top-k (top-k is not a supported parameter here),
  image attachment (the model takes images), and a warning when a stream ends without `[DONE]`.
- `ask.py` — terminal client in the style of `~/bin/ask-1650ti` / `ask-t4`: REPL with
  history, one-shot, pipe stdin as material, `-T` thinking, `-I` image, `-r` raw JSON.
  `make bin` links it as `~/bin/ask-tokenlab`.
- `benchmarks/runs/2026-11-13-first-light-tokenlab/` — `probe.py` and its `probe.jsonl`.

## Stream format

SSE with extras an over-strict parser trips on: `: ping` comment lines while queued,
`id: N` lines after each event, and `event: done` before `data: [DONE]`. Thinking arrives as
`delta.reasoning_content`. Thinking is switched with `chat_template_kwargs.enable_thinking`.

## Commands

`make ui`, `make ask`, `make models`, `make query`, `make probe`, `make test` (offline
`unittest`, nothing touches the network), `make lint`.
