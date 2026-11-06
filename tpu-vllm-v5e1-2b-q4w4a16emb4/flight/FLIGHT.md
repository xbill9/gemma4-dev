# The v5e-1 ladder

The G6 ladder (`gpu-vllm-g6-2b-q4w4a16emb4/benchmarks/runs/2026-11-0{5,6}-*-g6`) on one Cloud TPU v5e chip:
every engine that serves Gemma 4 E2B on a v5e-1, the same requests and statistics. The CUDA engines of the G6
ladder (llama.cpp, SGLang, mistral.rs, LiteRT-LM) have no TPU build; the TPU set is vLLM (`tpu_inference`), the
pure-JAX engine, its Rust front end, and TorchTPU in Python and in Rust. Started 2026-11-06.

| Arm | Engine | Rig | Weights |
| --- | --- | --- | --- |
| `vllm` | vLLM, this rig's patched image | `tpu-vllm-v5e1-2b-q4w4a16emb4` | `xbill9/...-w4a16-ct-text-emb4` (the G6 ladder's) |
| `vllm-text`, `vllm-text-dN` | the same | the same | `xbill9/...-w4a16-ct-text`; `-dN` adds `google/gemma-4-E2B-it-assistant`, N drafted tokens |
| `jax` | pure-JAX engine | `~/jax-dev/tpu-jax-v5e1-2b-q4w4a16` | `xbill9/...-w4a16-ct-text`, `--quant-mode w4a16` |
| `jax-bf16` | pure-JAX engine | `~/jax-dev/tpu-jax-v5e1-2b` | `google/gemma-4-E2B-it`, dense |
| `jaxrust` | Rust model over PJRT | `~/jax-dev/tpu-jaxrust-v5e1-2b` | `google/gemma-4-E2B-it` (bf16 only) |
| `torchtpu` | TorchTPU 0.2.0, Python | `~/torchtpu-dev/tpu-pytorch-v5e1-2b-w4a16` | `google/gemma-4-E2B-it-qat-w4a16-ct` |
| `torchtpu-rust` | TorchTPU 0.2.0, Rust (PyO3) | `~/torchtpu-dev/tpu-pytorchrust-v5e1-2b-q4w4a16emb4` | `-text` linears with the `-emb4` int4 tables |

Why the drafter arms use `-text`: vLLM's MTP drafter copies the target's bf16 `embed_tokens` and cannot pair with
the int4-table build; Google's QAT drafter was slower than none on this chip at every load (`QUANTIZATION.md`,
2026-09-29), so the drafter is the bf16-release `-it-assistant`.

## Where it runs

Three flex-start v5e-1s in `us-west4-a`, 24-hour run limit, one engine on a chip at a time:

- `tpu-vllm-v5e1-2b-q4w4a16emb4` (this rig's queued resource, its own startup script: pulls and patches the pinned
  image, serves vLLM): every vLLM and JAX arm. `setup-jax.sh` adds Python 3.13, `jax[tpu]==0.12.0.1` (the
  2026-11-03 v5e-1 runs' version), Rust nightly, the `jaxrust_rs` module and every checkpoint in `/opt/hf`.
- `ladder-v5e1-torchtpu`: the `torchtpu` arm. `ladder-v5e1-torchtpurs`: the `torchtpu-rust` arm. Both run the
  TorchTPU Rust rig's rendered startup script (Python 3.14, torch 2.14.1, torch_tpu 0.2.0, the crate built on the
  VM, the `torchtpu-rs` unit). TorchTPU decodes at roughly 12 to 50 tok/s on this chip, so its GSM8K runs take
  hours; separate chips keep them off the vLLM and JAX timeline.

Every arm on a chip of the same type and zone; a ratio between two engines on different chips is a ratio between
two v5e-1s.

## Jobs (`ladder.py`)

- `one:<arm>` for every arm: the G6 `engines.py` requests (six prompts, 256 tokens, greedy, thinking off,
  e2e and streamed, 3 repeats; then 4 at once).
- `sweep:<arm>` for the vLLM arms: `sweep.py` (the G6 copy), 512/128 and 128/512, 1-64 requests, 5 repeats.
  The JAX and TorchTPU servers take one request at a time; their 4-at-once row shows it.
- `acc:<arm>` for every arm: GSM8K (1,319, 2,048 tokens), one request at a time; BFCL simple (400) for the vLLM
  arms only. The JAX and TorchTPU servers render the chat template without the request's `tools` and return no
  `tool_calls`, so BFCL there would score a missing server feature, not the model.

Not measured: TPU memory in use (no `nvidia-smi` equivalent wired in).
