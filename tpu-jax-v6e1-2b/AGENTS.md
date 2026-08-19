# 🤖 Gemini Workspace Context: TPU Management Skill & tpu-jax-v6e1-2b MCP Agent

This workspace context file helps **Gemini Code Assistant** (and other developer tools) quickly understand the layout, goals, and integration methods of the **tpu-jax-v6e1-2b** project.

---

## 🎯 Project Overview & Role

This repository packages a Claude Code skill (`tpu-jax-v6e1-2b-management`) and a **Model Context Protocol (MCP) server** (`tpu-jax-v6e1-2b`) that together act as an AI DevOps/SRE agent for Google Cloud TPUs. Two main purposes:

1. **Infrastructure Operations:** Finding, probing, provisioning, and destroying TPU capacity **through Compute Engine only** (`gcloud compute instances create --machine-type=ct6e-standard-1t`), and running Gemma 4 JAX or vLLM workloads on the resulting VMs (v6e, v5p). This rig carries **no queued-resource path** — the Cloud TPU API is no longer under active development and TPU7x onward is Compute Engine or GKE only. v5e has no Compute Engine path at all and belongs on a `tpu-*` rig.
2. **Log & SRE Diagnostics:** Utilizing the self-hosted Gemma 4 model to analyze system/cloud logs and generate remediation suggestions.

---

## 📂 Quick Navigation

Key entrypoints in the codebase:

- **MCP server source:** [server.py](server.py) — the authoritative `tpu-jax-v6e1-2b` FastMCP agent (full tool catalog in `SKILL.md` / the `get_help` tool)
- **Skill definition:** [.claude/skills/tpu-jax-v6e1-2b-management/SKILL.md](.claude/skills/tpu-jax-v6e1-2b-management/SKILL.md) — lifecycle, tool catalog, required vLLM flags, field notes
- **Installer:** [project-setup.sh](project-setup.sh) — one-command skill install + MCP registration
- **Root Makefile:** [Makefile](Makefile) — `skill` / `skill-install` / `skill-package` / `init` targets
- **Snapshot refresher:** [refresh_skill.py](refresh_skill.py) — regenerates the bundled skill copies from the root sources
- **Plugin marketplace manifests:** [.claude-plugin/](.claude-plugin/) — makes the repo installable via the Claude Code plugin system
- **Reference guide:** `.claude/skills/tpu-jax-v6e1-2b-management/references/tpu-guide.md` — TPU getting started guide: zones, quotas, troubleshooting

---

## 🛠 Development Workflow & Makefile Tasks

The repo-root files (`server.py`, `project-setup.sh`, `tpu.md`) are authoritative; the skill directories and zip are generated snapshots. After editing a source:

```bash
make skill         # Regenerate skill snapshots + plugin copy
make skill-install # ...and install to ~/.claude/skills
make skill-package # ...and rebuild dist/tpu-jax-v6e1-2b-management-skill.zip
```

---

## 🔗 Integration with Gemini CLI via LiteLLM Proxy

You can redirect standard Gemini CLI commands to run against the self-hosted Gemma 4 model served from a TPU VM deployed by this agent. This lets developers use their own self-hosted inference engine under the hood.

### 1. Install LiteLLM Proxy

```bash
pip install 'litellm[proxy]'
```

### 2. Configure LiteLLM

Create a `litellm_config.yaml` targeting the TPU vLLM endpoint (get the IP with the agent's `get_vllm_endpoint` / `get_tpu_vm_endpoint` tools):

```yaml
model_list:
  - model_name: "gemma4-tpu"
    litellm_params:
      model: "openai/google/gemma-4-31B-it"
      api_base: "http://YOUR_TPU_IP_ADDRESS:8000/v1"
      api_key: "none"
    router_settings:
      model_group_alias:
        "gemini-2.0-flash": "gemma4-tpu"
        "gemini-2.0-flash-lite": "gemma4-tpu"
        "gemini-1.5-flash": "gemma4-tpu"
        "gemini-1.5-pro": "gemma4-tpu"
```

Adjust `model` to match the served model (`MODEL_NAME` env var of the agent), e.g. `openai/google/gemma-4-12B-it` or `openai/google/gemma-4-E4B-it`.

### 3. Run Proxy & Point Gemini CLI at It

Run the proxy locally:

```bash
litellm --config litellm_config.yaml --port 4000
```

The `model_group_alias` mapping above is what does the real work: any request the
Gemini CLI makes for a `gemini-*` model is routed to the self-hosted `gemma4-tpu`
endpoint. Then point the CLI at the proxy:

```bash
export GOOGLE_GEMINI_BASE_URL="http://localhost:4000"
export GEMINI_API_KEY="local-proxy-token"
export GEMINI_MODEL="google/gemma-4-31B-it"   # match the served model
```

> **Note:** environment-variable names vary between Gemini CLI releases — if the CLI
> ignores `GOOGLE_GEMINI_BASE_URL`, check `gemini --help` / its settings file for the
> current base-URL override; the LiteLLM config itself needs no changes.

---

## 🔧 Technical Standards for vLLM & Gemma 4 Tool Calling

When managing TPU deployments or customizing vLLM serving, ensure the following vLLM serving parameters are applied for stable Gemma 4 tool integration:

- **Optimization flags:** `--tensor-parallel-size 8` (TPU v6e-8), `--disable_chunked_mm_input`, `--max-model-len 16384`.
- **Tool Parsing:** `--enable-auto-tool-choice`, `--tool-call-parser gemma4`, and `--reasoning-parser gemma4` to enable native function calling compatibility.
- **Multimodal configuration:** `--limit-mm-per-prompt '{"image":4,"audio":1}'` and `--max_num_batched_tokens 4096`.
- **Universal SRE Help:** The agent exposes a standardized `get_help` tool providing details on active configuration environment variables and all exposed tools.

## 📊 Analysis Standards

- **Dependency Portability:** Avoid assuming third-party analysis libraries like `pandas` are installed in the workspace environment. Prefer standard libraries (e.g., `csv`, `json`) for data parsing and aggregation scripts.

---

## ⚡ Gemma 4 E2B QAT JAX Engine & TPU v6e-1 Benchmarks

This workspace includes a high-performance, raw JAX inference engine for **Gemma 4 E2B QAT** (`ports/gemma4/jax_e_model.py` and `jax_openai_server.py`), specifically tailored for Cloud TPU v6e single-chip hardware (`ct6e-standard-1t`, 32 GB HBM3).

### 🚀 Hardware-Specific Optimizations
1. **128-Aligned Static Bucket Padding** (`pad_to_tpu_v6e_bucket`):
   - Aligns sequence lengths and batch dimensions to TPU v6e Matrix Unit (MXU) $128 \times 128$ systolic array boundaries ($N \pmod{128} = 0$), preventing XLA graph recompilation and maximizing hardware FLOP efficiency.
2. **Vectorized On-Chip Top-K Sampling** (`onchip_sample_tpu_v6e_jax`):
   - Leverages Gemma 4 E2B's tile-aligned $262,144$ vocabulary dimension ($2,048 \times 128$), running sampling 100% on TPU cores with zero CPU host transfers.
3. **Persistent XLA Compilation Disk Cache**:
   - `jax.config.update("jax_compilation_cache_dir", "~/.cache/jax_compilation_cache")` persists compiled HLO across restarts, skipping ~17s of XLA compilation on a warm restart.
4. **OpenAI-Compatible SSE Token Streaming**:
   - Real-time `text/event-stream` token generation supported via `jax_openai_server.py`.

### 📈 Performance Status: RE-MEASURED 2026-07-29

The 2026-07-21 sweep (peak 6,496.8 tok/s) was withdrawn — `jax_e_benchmark_sweep.py`
timed prefill un-jitted, divided a scan total that contained the prefill by the step
count, and ran a decode loop with no KV cache and no attention mask. Use
`ports/gemma4/jax_e_benchmark_sweep_v2.py`, which times jitted prefill and
steady-state cached decode separately.

Re-measured on a v6e-1. Full data in
`benchmarks/runs/2026-07-29-kv-quant-v6e1/REPORT.md`:

| | measured |
|---|---|
| decode budget | **663,552–712,704** resident KV tokens (bf16) · **1.25–1.54M** (int8) |
| prefill admission | **B × chunk ≤ 8,192** prompt tokens per pass |
| KV | **18.00 KiB/token**, verified against the checkpoint |
| int8 KV | 1.88–1.98× capacity, ~1.18× faster, quality-neutral |
| int4 PLE | 53% smaller model, quality- and throughput-neutral, +13% capacity |
| best config | `ple-4 / kv-int8 / donated` — 1.92× faster at 53% the size |

**Read the corrections in that report before quoting any number from it.** Three
findings were withdrawn during the session: a "flat 524,288 KV tokens, 0.0% spread"
invariant that was an artifact of a power-of-two sampling ladder; "int8 KV is
1.2–1.8× faster", which was mostly the cost of a cache being copied every step; and
a roofline calculation that counted the 4.70 GB PLE table as streamed when it is a
gather.

**This engine is not at the hardware limit.** Calibrated against a trivial streaming
reduction that reaches 1,417 GB/s (86% of the published 1,640), the decode step's KV
read runs at ~56% of that with buffer donation enabled and 19% without. Absolute
throughput here describes this implementation, not the chip.

```bash
python3 ports/gemma4/jax_e_benchmark_sweep_v2.py \
  --batch-sizes 1,2,4,8,16,32,64 --contexts 8,128,512,2048 --json-out results.json
```

### 📚 JAX References

- **[JAX advanced guides](https://docs.jax.dev/en/latest/advanced_guides.html)** —
  the index worth reading before optimizing anything here. Its *Performance
  optimizations* section documents **buffer donation**, which turned out to be the
  single largest inefficiency in this engine: without it `dynamic_update_slice`
  rebuilds the whole KV cache to write one token, costing 1.62× on a bf16 cache. We
  found that by measuring against a calibrated bandwidth ceiling; it was in the docs
  the entire time. *Performance benchmarking and profiling* covers the trace tooling
  used in `benchmarks/queued/kernel_gap_suite.py`, and *JAX Memories and Host
  Offloading* is directly relevant to the HBM accounting in the report above.
- **Type promotion**: float8 dtypes are deliberately excluded from JAX's promotion
  lattice, which is why a cached fp8 key meeting a bf16 query raises rather than
  promoting — while int8 *is* in the lattice and silently contracts against raw
  integers. See `quantize_kv` in `ports/gemma4/jax_e_model.py`.
- **[How to think in JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html)**
  — start here. Two of its core lessons produced the largest bugs in this engine:
  - **JAX arrays are immutable**, so an "in-place" update is really a new array.
    `dynamic_update_slice` writing ONE token into the KV cache therefore rebuilt
    the whole cache every decode step, which cost 1.62x until buffer donation was
    enabled. The immutability is the language; donation is the escape hatch.
  - **Static vs traced values.** A Python `int` stored in a params pytree becomes a
    tracer under `jit`, and `int()` on a tracer raises. `gather_ple` derives its
    group count from an array SHAPE rather than a stored integer for this reason.
  Also covers why `print` inside `jit` shows a tracer, and why dynamic shapes
  (boolean indexing) will not compile — which is why every cache and mask in this
  engine is statically shaped and padded to a bucket.
- **[JAX debugging](https://docs.jax.dev/en/latest/debugging.html)** — the tools for
  the failure mode this codebase keeps producing: things that run, report success,
  and compute the wrong thing.
  - `jax_debug_nans` catches NaN at the point of creation instead of many layers
    downstream. The engine currently asserts `jnp.all(jnp.isfinite(...))` by hand
    in a few places; this is the supported way.
  - `jax.experimental.checkify` adds jit-compatible runtime assertions — bounds,
    NaN, user predicates — which is what a quantized KV cache wants: a scale of
    zero or an unwritten slot should raise, not silently produce garbage.
  - `jax.debug.print` / `jax.debug.breakpoint` print from *inside* jit. Worth
    knowing before hand-rolling probe wrappers, and note that wrapping a Pallas
    kernel body breaks `pallas_call`'s inspection of it — probe the public entry
    point instead.
  - `jax_disable_jit` turns a traced program back into ordinary Python. The fastest
    route through a `TracerBoolConversionError`, e.g. calling `int()` on a value
    that came from a params pytree — see the group-size handling in `gather_ple`,
    which derives its shape statically for exactly this reason.
- **[Gemma 4 QAT checkpoints](https://ai.google.dev/gemma/docs/core#qat)** — the
  authority on which checkpoint variant to load, and the reason the int4 weights this
  engine unpacks are not a post-hoc approximation: QAT simulates quantization *during*
  training, so the model learns to compensate for the precision loss. The suffixes are
  not interchangeable:
  - `-w4a16-ct` — 4-bit weights, 16-bit activations, aimed at cloud serving. This is
    the default `MODEL_ID` in `jax_openai_server.py` and what `w4a16_impl` decodes.
  - `-q4_0-unquantized` — half-precision QAT weights, intended for custom compilation
    and speculative decoding. Faster on this engine (10.1 vs 8.1 tok/s, `deploy.md`),
    at ~4× the HBM.
  - `-gguf` (llama.cpp-style 4-bit) and `-mobile-ct` (`wNa8o8`, 2-bit decode layers)
    target other runtimes entirely — neither belongs on TPU.
  QAT variants exist across E2B, E4B, 12B, 26B A4B, and 31B, but the mobile ones only
  for E2B/E4B. Check the list here before assuming a variant exists for a given size.

### 🔑 Verified Takeaways
- **QAT checkpoints load.** The pure-JAX path resolves the vLLM TPU loader failure
  (`#3225`: `k_norm` demanded for KV-shared layers 15–34 that the checkpoint legitimately
  omits) and unpacks W4A16 int4 weights with zero PyTorch in the path.
- **The decoder is verified**, not merely fast: cached decode matches a full-sequence
  re-forward within float32 roundoff.
- **vLLM baseline (measured, valid):** 213 tok/s single-stream (16 ms TTFT), ~2,140 tok/s
  at concurrency 64, ~8.5 min time-to-healthy — see
  `benchmarks/reports/2026-07-21-gemma4-e2b-v6e1.json`.

