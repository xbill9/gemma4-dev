# jev-tpu-31b

Exploration: a Jev-style label read of Gemma 4 31B on TPU v6e. Sibling of `../jev-tpu`, which read E2B, E4B, 12B and 26B-A4B on one v6e chip and could not serve any 31B. The read, data and scoring code here are copied from `../jev-tpu`; `nimble_suite/run_suite.py` differs by one recorded field (`PREREGISTRATION.md`, deviations).

## Status, 2026-09-25: every Gemma 4 size serves on one v6e chip

With three patches to `tpu_inference`, Google's QAT W4A16 checkpoints serve in vLLM on one v6e chip at `vllm/vllm-tpu@sha256:19a1a052…`, 31B included. All three are upstream pull requests, none merged yet:

| patch | pull request | needed for |
|---|---|---|
| KV-shared layers own no K/V parameters | vllm-project/tpu-inference#3299 (rebased) | E2B, E4B QAT exports |
| JAX-path compressed-tensors W4A16 on `gmm_v2` | vllm-project/tpu-inference#3653 | every QAT W4A16 export |
| `Gemma4UnifiedForConditionalGeneration` (12B) text-only on JAX | vllm-project/tpu-inference#3654 | 12B |

Label read on the 3,880-record public suite, W4A16 against bf16 on the same records (`results/2026-09-25-w4a16-QUANT.md`, from `quant_compare.py`): E2B −2.8 points (95% range −4.0 to −1.6), E4B −1.6 (−2.5 to −0.9), 12B −1.0 (−1.7 to −0.3); 31B scores 77.6% and has no bf16 reference on one chip. Output tokens per second, 16 concurrent requests of 256 tokens: W4A16 runs 1.14x (E2B), 1.21x (E4B) and 1.41x (12B) faster than bf16, and 31B runs at 500. `PREREGISTRATION.md` defines the read and records its deviations; logs are in `results/*-evidence/`.

The section below is the state before the patches, measured on the same image, and still describes the stock image.

## The issue: no 31B checkpoint both fits one v6e chip and loads in vLLM on TPU

Measured 2026-09-24 on one v6e chip, `vllm/vllm-tpu@sha256:19a1a052…` (vLLM `0.29.1rc1.dev468+g0b7f11a1e`); logs in `../jev-tpu/results/2026-09-24-v6e1-evidence/`. Usable HBM on one chip is about 28.74 GiB (`total_hbm_limit_cap_gb` in the vLLM boot log; `../HARDWARE.md`).

| 31B checkpoint | Size on disk | Fits one v6e chip | Loads in vLLM on TPU |
|---|---|---|---|
| `google/gemma-4-31B-it` | 62 GB bf16 | no | — |
| `RedHatAI/gemma-4-31B-it-FP8-dynamic`, `-FP8-block` | 30.98 GiB | no: over the cap before any KV cache or working memory | yes: fp8 w8a8 is the one quantized scheme on the JAX path |
| `google/gemma-4-31B-it-qat-w4a16-ct` | 21.67 GiB | yes | **no**: `NotImplementedError: compressed-tensors scheme for layer 'model.language_model.layers.0.self_attn.q_proj' is not yet supported in the JAX path`, 120 s into boot |
| `cyankiwi/gemma-4-31B-it-AWQ-4bit` (stored as compressed-tensors w4a16) | 19.47 GiB | yes | **no**: the same error |
| real AWQ, GPTQ, AutoRound int4 (`QuantTrio/gemma-4-31B-it-AWQ`, `Intel/gemma-4-31B-it-int4-AutoRound`) | 18–21 GiB | yes | no: torch path only, and Gemma 4 is JAX-path only (`../QUANTIZATION.md`) |
| NVFP4 builds | about 19 GiB | yes | no: torch path only |
| `-qat-q4_0-gguf` | — | — | no: GGUF is not a vLLM format |

The format that loads is too large, and every format small enough does not load. qwix (in-memory int8/int4 quantization at load) would have to read the bf16 weights first; its `use_abstract_model` mode, which avoids that, failed on E2B (`../QUANTIZATION.md`, "qwix is the only way in"). The w4a16 gap is a placeholder in `tpu_inference`'s compressed-tensors dispatcher: `# TODO: w4a8 / wNa16 schemes need their own JAX methods (not yet ported)`.

v6e comes in 1, 4 and 8 chips; there is no v6e-2.

## Routes to explore

**A. The pure-JAX engine on one v6e chip.** `~/tpu-jax-31b` (`jax_engine.py`, `jax_openai_server.py`) served `google/gemma-4-31B-it-qat-w4a16-ct` on a v6e-1 at 19.30 GB resident with 14.25 GB headroom (`../tpu-jax-v6e1-31b-w4a16/benchmarks/runs/2026-07-31-gemma4-31b-v6e1/REPORT.md`). Its server returns no logprobs, but the engine computes the final-position logits before sampling (`jax_engine.py:458`, `onchip_sample_tpu_v6e_jax(logits, …)`). The work is one endpoint that returns the log-softmax at the label token ids, which gives exact label probabilities rather than the top 32. Cost: one v6e-1 for about an hour. The result is about 31B through a different serving stack, so it sits beside the vLLM arms rather than in the same table.

**B. vLLM on a v6e-4.** `google/gemma-4-31B-it` at bf16, or the RedHatAI fp8 build, with `--tensor-parallel-size 4`. Same stack and flags as `../jev-tpu`. Cost: about $5.40 an hour flex-start or $11.90 on demand in europe-west4 (`../TPU.md`), about an hour of runtime plus a longer boot.

**C. Wait for wNa16 on the JAX path.** When `tpu_inference` ports w4a16, `google/gemma-4-31B-it-qat-w4a16-ct` (21.67 GiB) fits one chip and runs through `../jev-tpu` unchanged. Check with one 3-minute boot: the failure above appears 120 s in.

**D. qwix `use_abstract_model`.** Deprecated and broken on E2B as of 2026-08-07; re-test on a small model before spending a 31B boot on it.

## Before running anything

Write `PREREGISTRATION.md` here first, as in `../jev-tpu`. For route A the read changes (exact log-softmax at the label ids, not vLLM's top 32), so the plumbing check is E2B or E4B on the same engine against `../jev-tpu`'s per-record outputs before the 31B arm.
