# gpu-ollama-g5g-2b-q4_0

An MCP devops agent for **EC2 G5g** (AWS Graviton2 host + NVIDIA **T4G**, Turing SM 7.5),
serving **`gemma4:e2b-it-qat`** — Ollama's repackaging of Google's QAT Q4_0 GGUF — through the
**Ollama daemon**.

> **This rig has served nothing.** Created 2026-09-02. `benchmarks/` is empty on purpose, and
> every number in this repo's prose is a property of the bundle, arithmetic from the artifact,
> or a sibling's measurement — each says which. See `CLAUDE.md`.

## Its sibling runs the same engine

`gpu-llamacpp-g5g-2b-q4_0` serves the same weights on the same silicon, and **Ollama links
llama.cpp** — the Gemma 4 graph both execute is upstream `src/models/gemma4.cpp`. Slot 2 names
the front end, not the decoder.

They are two rigs because four differences are visible to a benchmark (THE NOTE in `CLAUDE.md`
has the byte counts). The headline one:

> **Ollama ships a working aarch64 CUDA binary with native sm_75 SASS. llama.cpp publishes
> none, so the sibling compiles.** This rig has no build step at all.

## Quick start

```bash
pip install -r requirements.txt   # the MCP server's own deps, on your machine
make test                         # 53 offline tests: no AWS, no network, no GPU
make lint
make skill
```

Then, through the MCP server:

```
create_g5g_instance  ->  get_install_progress  ->  verify_gpu_arch
                     ->  verify_model_health   ->  get_metrics
```

No deploy step: cloud-init downloads the bundle, starts the daemon, pulls the tag, and asserts
the model reached VRAM before declaring success.

## The one thing that goes wrong silently

**A CPU-resident model.** Ollama chooses its own offload and cannot be told to fail — there is
no `--n-gpu-layers` equivalent. It serves correctly, several times slower, and logs nothing.

The only honest check is `size_vram` from `/api/ps`: **0 means CPU**, and less than `size` means
a partial offload. `verify_gpu_arch` reads it; run it *first* whenever a number looks low.

## Three pinned settings

| Setting | Ollama default | Why pinned |
| --- | --- | --- |
| `OLLAMA_LLM_LIBRARY` | auto, by driver | `cuda_v13` is **PTX-only for every arch** and JITs at load; `cuda_v12` carries native sm_75 CUBIN. Pinning is what makes the sibling pair comparable |
| `OLLAMA_CONTEXT_LENGTH` | `0` = 4k/32k/256k **by VRAM** | two instance sizes would silently get two contexts |
| `OLLAMA_KEEP_ALIVE` | `5m` | a sweep pausing longer reloads the model and records it as latency |

`ollama serve` takes **no arguments** — all of this is environment, and the daemon *ignores* an
unknown variable rather than rejecting it. A typo yields a working rig at the wrong settings,
which is why `_serve_env` is tested against the daemon's own variable list.

## Reading a number off it

There is **no `/metrics`** — Ollama registers no Prometheus route. `get_metrics` **probes**: one
generate, then `eval_count` / `eval_duration` off the response. Those are Go `time.Duration`s,
i.e. **nanoseconds**. A probe measures one request, not a run.

```bash
python3 sweep.py --base http://<ip>:8000/v1 --out benchmarks/runs/<date>-<what>-g5g
```

Health is `GET /` ("Ollama is running"); there is no `/health`, and it answers 200 before any
model is loaded.

## Configuration

`tpu.env` is the source of truth and **every key in it is read by `server.py`** — a test asserts
this. The directory name is documentation, not config.

## Layout

```
server.py        the MCP server: EC2 lifecycle, bootstrap rendering, SRE tools
tpu.env          authoritative configuration
sweep.py         benchmark harness (shared shape with every sibling)
make_report.py   sweep output -> serving-report.schema.json
tests/           offline; test_server.py asserts on the RENDERED BOOTSTRAP
```

No serving payload and no build. The engine is a published release tarball.
