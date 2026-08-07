# TorchTPU vs vLLM — Gemma 4 E2B on TPU v6e-1

**Run:** 2026-07-25 · `torchtpu-vm` (ct6e-standard-1t, flex-start, europe-west4-a, project `comglitn`)
**Model:** `google/gemma-4-E2B-it`, bfloat16 (stock, unquantized)
**Comparison baseline:** vLLM serving report [`benchmarks/reports/2026-07-21-gemma4-e2b-v6e1.json`](../../reports/2026-07-21-gemma4-e2b-v6e1.json) — same model, same chip, same zone.

## Headline results

| Configuration | Single-stream decode | Per-token latency | Notes |
| :--- | ---: | ---: | :--- |
| **TorchTPU eager** (HF `generate()`) | **3.4 tok/s** | ~295 ms/tok | op-by-op dispatch; functional but unusable for latency |
| **TorchTPU compiled** (static-shape loop) | **227.3 tok/s** | **4.4 ms/step** | `torch.compile(backend="tpu")`, SEQ=256 buffer, **no KV cache** (full re-forward/step) |
| **vLLM** (engine, concurrency 1) | 209 tok/s | 4.7 ms TPOT, TTFT 16 ms | full serving stack over HTTP, fp8 KV cache |
| **vLLM** (engine, concurrency 100) | 2,215 output tok/s aggregate | 36.8 ms TPOT | saturation — nothing in the TorchTPU setup competes here |

**Takeaways**

1. `torch.compile(backend="tpu")` is worth **67×** over eager (3.4 → 227.3 tok/s). Eager TorchTPU is for correctness checking only.
2. At batch 1, a compiled static-shape torch loop **matches/beats the vLLM engine** (227 vs 209 tok/s; 4.4 vs 4.7 ms/token) — remarkable given the loop re-forwards the entire 256-token buffer every step with **no KV cache** while vLLM decodes incrementally. Compute is cheap on the MXU; overhead dominates at batch 1.
3. vLLM remains ~10× ahead at concurrency (2,215 aggregate tok/s at c=100). Continuous batching + paged KV cache are unmatched for shared endpoints.
4. Cost per 1M output tokens at $1.35/chip-hr: TorchTPU compiled single-stream **$1.65** vs vLLM single-stream $1.79 vs vLLM saturated **$0.17**. Multi-user serving belongs on vLLM; single-stream/experimental work is cost-equivalent either way.

## Model scaling on one v6e-1 (added later on 2026-07-25)

Same protocol per model — eager HF `generate()` vs the compiled static-shape loop
(SEQ=256, no KV cache, greedy). Logs: `11-bench-scale.log`, `12-bench-e4b.log`.

| Model | Params (loaded) | Eager | Compiled | ms/step | Compile speedup |
| :--- | ---: | ---: | ---: | ---: | ---: |
| gemma-4-E2B-it | 5.5 B | 3.4 tok/s | **227.3 tok/s** | 4.4 | 67× |
| gemma-4-E4B-it | ~9 B | 2.8 tok/s | **114.1 tok/s** | 8.8 | 41× |
| gemma-4-12B-it | 13.0 B | 2.2 tok/s | **45.2 tok/s** | 22.1 | 21× |

Observations: compiled throughput halves as parameter count doubles — the
compiled loop is cleanly compute-bound on the MXU, while eager barely moves
(dispatch-bound at ~3 tok/s regardless of model size). 12B bf16 (~26 GB
weights) fits and runs on the 32 GB chip with the 256-token buffer.
Note: there is no `google/gemma-4-4B-it` — the 4B-class model is `gemma-4-E4B-it`
(404 recorded in `11-bench-scale.log`).

## Software under test

| Component | Version |
| :--- | :--- |
| torch_tpu | `0.1.1.dev20260725090141` (that day's nightly) |
| torch | `2.11.0+cpu` (pinned — see findings) |
| transformers | 5.14.1 · libtpu 0.0.44 · Python 3.12 (no venv, dedicated interpreter) |
| vLLM baseline | `0.23.1rc1.dev1076+g5c342876a`, `vllm/vllm-tpu:nightly`, tpu-inference (JAX) backend |

## Timeline (from logs/)

| Step | Duration | Log |
| :--- | ---: | :--- |
| Flex-start capacity grant + VM RUNNING | < 1 min | `01-create.log` |
| Startup script (python3.12 OK; pip fails — see findings) | ~2 min | `02-serial-boot.log` |
| Manual wheel install with user token (torch_tpu + transformers) | ~2 min | `03-pip-install.log` |
| torch 2.13→2.11 downgrade + smoke test (incl. first matmul compile) | ~7 s test | `05-torch-downgrade-smoke.log` |
| Model load: cold (incl. HF download) / warm | 61.1 s / ~7 s | `06-generate-eager.log` |
| Eager `generate()` benchmark | 3.4 tok/s | `06b-generate-eager-summary.txt` |
| Compiled loop: warmup incl. model compile | 19.3 s | `10-decode-compiled.log` |
| Compiled loop: 64 timed decode steps | 0.28 s | `10-decode-compiled.log` |

Time from `gcloud create` to first compiled benchmark number: **~25 min** (including debugging).
vLLM baseline time-to-healthy on the same image class: 510 s.

## Findings (things that bit us, in order)

1. **The VM's default compute SA cannot read the torch_tpu wheel registry** (`ml-oss-artifacts-transient` is Google-owned; no IAM grant possible). The startup script's pip step fails with "No matching distribution found for torch_tpu". Workaround used: finish the install over IAP SSH with a user-identity token piped via stdin. Long-term: mirror the wheels into a project-owned Artifact Registry.
2. **torch_tpu's `torch` dependency is unpinned** → pip installed torch 2.13.0+cpu and the extension failed to load (`undefined symbol: …autograd…deleteNode…`). Fix: `torch==2.11.0+cpu` from the same index. Now the default in `TORCH_TPU_PIP_SPEC`.
3. **HF `generate()` cannot be `torch.compile`d on TorchTPU today** — generate passes a scalar CPU tensor into the compiled graph; TorchTPU compiled mode requires all args on-device (`tensor is expected to be on tpu, got cpu`). Upstream's own HF example runs generate eager-only. The compiled number here comes from a hand-rolled static-shape greedy loop (`torchtpu_decode_bench.py`) that follows all the TorchTPU rules: fixed [1,256] buffer, all tensors on `tpu`, no graph breaks, single compile.
4. **jinja2 papercut:** image ships jinja2 3.0.3; `apply_chat_template` needs ≥3.1.
5. **Fused SDPA kernel constraint (tuning headroom):** TorchTPU's fused attention kernel logged that it requires sequence lengths divisible by **512** (ours: 256 → unfused fallback). The 227 tok/s number therefore leaves kernel-level performance on the table; a 512-buffer run may be faster still despite doubling per-step FLOPs.

## Comparability caveats

- TorchTPU numbers are in-process wall-clock over 64 greedy steps after warmup; vLLM numbers come from `vllm bench serve` over HTTP with its full scheduler. The vLLM figures include serving overhead the torch loop doesn't pay.
- The torch loop has no KV cache and a 256-token window; it is a *fixed-footprint single-stream* design, not a general server. Longer contexts shift the balance toward vLLM's incremental decode.
- TTFT: vLLM 16 ms warm. The torch loop's equivalent is one step (~4.4 ms) once compiled, but the first request pays ~19 s of compile — amortize or pre-warm.
- Single run per configuration; the baseline report measured run-to-run cv ≤ 0.3 % for greedy/static-shape workloads on this stack.

## Verdict

For **serving many users**, vLLM on TPU stays the only sensible choice (10× aggregate throughput, $0.17/M tokens saturated). For **single-stream, fixed-footprint, or research workloads** — including the QAT/custom-quantization experiments that vLLM's loader currently blocks — TorchTPU with a compiled static-shape loop is already competitive with the serving engine at batch 1, on its very first nightly we tried. The stock-2B milestone is met; the QAT experiment is unblocked (same loop, swap the checkpoint).

## Artifacts

- `torchtpu_decode_bench.py` — the compiled benchmark loop (reusable for QAT variants)
- `startup-script-rendered.sh` — exact VM startup script used
- `logs/01…10` — every command's full output, serial console included
- Repo `torchtpu_generate.py` — eager HF runner (works as-is on any workload=pytorch VM)
