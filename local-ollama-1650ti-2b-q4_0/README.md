# local-ollama-1650ti-2b-q4_0

Serving **`gemma4:e2b-it-qat`** — Ollama's repackaging of Google's QAT **Q4_0**
GGUF — through the **Ollama daemon** on the **GTX 1650 Ti (Max-Q)** in the
machine under the desk.

No cloud, no control plane, no provisioning. The card is there whether or not
any code runs, `127.0.0.1:8000` is known before the process starts, and Ctrl-C
is a complete teardown.

## Why it exists

It is the **A/B partner of `local-llamacpp-1650ti-2b-q4_0`**. The two serve the
same weights on the same card through the same engine — Ollama links llama.cpp
as a library, and the Gemma 4 graph both execute is upstream
`src/models/gemma4.cpp`. Everything that differs is a **choice the daemon makes
and llama.cpp leaves to you**, which is exactly what the pair is built to
measure.

Four such choices, all measured 2026-09-04:

| | `local-llamacpp-…` | `local-ollama-…` |
| :--- | :--- | :--- |
| Artifact | Google's file, hashable | re-containered blob, template stripped |
| Projector (986 MB) | not loaded | loaded unconditionally by the **stock tag** |
| Resident VRAM @ 8192 ctx | 1618 MiB | 2762 MiB stock → **1612 MiB as served** |
| Chat template | the GGUF's own | Ollama's Go renderer |
| Offload | `-ngl 99`, yours to set | daemon's VRAM estimate, detect-only |
| Flash attention | `-fa 1`, worth +4.8% | `--flash-attn auto`, already on |

## Quick start

```bash
make install          # system python3; never a virtualenv
make pull             # stock tag, 4.3 GB into ~/.ollama/models
make text-only        # drop the projector layer — worth 1150 MiB (see below)
make serve            # foreground daemon on 127.0.0.1:8000
make residency        # is it fully on the GPU, and what does the driver say
make query            # one chat request, thinking enabled
```

## What this rig serves, and why it is not the stock tag

`MODEL_NAME` is **`gemma4:e2b-it-qat-text`**, not `gemma4:e2b-it-qat`.

The stock tag declares `vision` and `audio`, so the daemon passes `--mmproj`
unconditionally and a vision+audio encoder is resident for a pure-text workload:
**1154 MiB of a 4096 MiB card**, which `ollama ps` does not count. The variant is
`ollama show --modelfile` of the stock tag with the second `FROM` removed and
nothing else changed — same model blob **by digest**, same `RENDERER gemma4` /
`PARSER gemma4`, same three sampling parameters. It shares the blob, so it costs
no extra disk.

| | resident @ 8192 ctx | capabilities |
| :--- | ---: | :--- |
| `gemma4:e2b-it-qat` | 2762 MiB | completion vision audio tools thinking |
| **`gemma4:e2b-it-qat-text`** | **1612 MiB** | completion tools thinking |
| llama.cpp sibling, same weights | 1618 MiB | — |

It is not only the encoder. With the projector loaded the daemon also asked
llama-server for **2048 tokens/slot when told 1024**, so 32 slots did not fit at
all — 26/36 layers offloaded and 512 MiB of KV on the host, which halved
single-stream decode. Projector-free, the same request is honoured at 1024/slot
and **32 slots fit entirely in VRAM at 2106 MiB**.

The stock tag stays on disk as `MODEL_STOCK_TAG`: it is what
`gpu-ollama-g5g-2b-q4_0` serves, so it is the reference for comparing against
that rig.

`make status` in another shell tells you whether it is up.

## Measurements

`benchmarks/runs/` holds this rig's own runs and nothing else — no sibling's
numbers travelled with the scaffolding. `benchmarks/INDEX.md` indexes them and
is generated; regenerate from the monorepo root with `make benchmarks-rollup`.

## Three things that will cost you a run

- **`ollama ps` under-reports.** Its `SIZE` excludes the projector: 1.7 GB
  against the driver's 2762 MiB on the stock tag. Use `make residency`, which
  reads both and interprets the gap against the model's declared capabilities.
- **A small `num_predict` returns nothing at all.** Gemma 4 thinks first, and
  Ollama's default discards the thinking block; at 128 tokens the block has not
  closed, so `content` and `thinking` are both empty with
  `done_reason: length`. 128 tokens were generated. None came back.
- **The prompt cache outlives your client.** `cache_prompt: false` does not
  survive Ollama's OpenAI translation, and a fixed RNG seed makes a second
  process regenerate the first one's prompts. `sweep.py` defaults to
  `--prompt-mode shuffled` here for that reason; see its comments.

## Layout

```
server.py    MCP server — daemon lifecycle, residency, inference. No provisioning tools.
modelfiles/  text-only.Modelfile — the projector-free variant, built by `make text-only`
sweep.py     benchmark harness (a COPY of the llama.cpp sibling's; rigs are siblings)
tpu.env      source of truth, committed
tests/       offline unittest suite — python3 -m unittest discover -s tests -v
benchmarks/  this rig's runs and reports, plus the synced root schema and README
```

Read `CLAUDE.md` before changing anything.
