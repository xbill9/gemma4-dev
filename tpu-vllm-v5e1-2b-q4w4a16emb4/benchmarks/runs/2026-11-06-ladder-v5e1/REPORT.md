# v5e-1 ladder: the engines that serve Gemma 4 E2B on one TPU v5e chip (2026-11-06/07)

Three flex-start `v5litepod-1` VMs in `us-west4-a`, one engine on a chip at a time, measured with the G6
ladder's requests and statistics (`flight/FLIGHT.md`). Every figure below is computed by `analyze.py` from the
run directories here and printed in `SUMMARY.txt`; this file quotes it.

| Directory | Queued resource | Jobs |
| --- | --- | --- |
| `tpu-vllm-v5e1-2b-q4w4a16emb4/` | A | `vllm`, `vllm-text`, `vllm-text-d4` (failed), `jax` moved to C |
| `ladder-v5e1-torchtpu/` | B | `torchtpu-bf16` (`one` only), `jax-bf16` |
| `ladder-v5e1-torchtpurs/` | C | `torchtpu-rust` (`one` only), `jaxrust`, `jax` |

Software: vLLM `0.29.1rc1.dev468+g0b7f11a1e` on the rig's patched image (`vllm-tpu-q4w4a16emb4:patched`, image
`sha256:097f23f4…`); jax / jaxlib `0.12.0.1`, libtpu `0.0.50.1`, rustc nightly `249c1e4e3` (`*/versions.txt`).
Scripts as run: `ladder.py`, `sweep.py`, `setup-jax.sh`, `finish.sh`, `attest.py`.

#### One request at a time

Six prompts, 256 tokens, greedy, thinking off, 3 repeats; geometric mean over prompts of each prompt's median.

| Arm | e2e tok/s | decode tok/s | TTFT ms | 4 at once tok/s | x vllm |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vllm` (`-text-emb4`) | 143.9 | 145.1 | 15.4 | 477.2 | 1.000 |
| `vllm-text` | 134.2 | 135.2 | 16.2 | 449.8 | 0.932 |
| `jax` (W4A16) | 104.1 | – | 15.9 | 122.1 | 0.723 |
| `jax-bf16` | 134.3 | 131.7 | 11.3 | 150.1 | 0.933 |
| `jaxrust` (bf16) | 145.2 | 145.5 | 9.4 | 145.4 | 1.009 |
| `torchtpu-bf16` | 6.6 | – | 153.6 | 6.7 | 0.046 |
| `torchtpu-rust` | 13.4 | 13.4 | 81.3 | 13.6 | 0.093 |

- 🟢 The Rust model over PJRT is level with vLLM for one request (1.009x) and starts in 22 s against vLLM's
  1,860–2,055 s.
- ⚠️ The JAX and TorchTPU servers serve one request at a time, so 4 at once is 1.00–1.17x one request; vLLM
  gives 3.32x.
- ⚠️ The `jax` and `torchtpu-bf16` streams counted 0 tokens on every row, so they have no decode figure; their
  end-to-end figures are complete.
- All arms repeat their greedy replies on 6 of 6 prompts.

#### Many requests at once (vLLM)

Output tok/s, 5 repeats, unique prompts.

| Arm | Shape | 1 | 8 | 16 | 32 | 64 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `vllm` | 512 in / 128 out | 140.6 | 682.2 | 937.9 | 1,156.8 | 1,288.4 |
| `vllm-text` | 512 in / 128 out | 131.3 | 652.5 | 906.8 | 1,125.9 | 1,266.0 |
| `vllm` | 128 in / 512 out | 144.5 | 778.4 | 1,112.8 | 1,462.5 | 1,689.1 |
| `vllm-text` | 128 in / 512 out | 134.4 | 737.3 | 1,084.7 | 1,424.9 | 1,660.1 |

- 🟢 The int4 vocabulary tables are worth 1.07x for one request, falling to 1.02x at 64.
- ⚠️ Against vLLM on the G6 L4 with the same checkpoint (`gpu-vllm-g6-2b-q4w4a16emb4/.../2026-11-06-suite-g6`),
  one v5e chip is 1.06x / 1.04x for one request and 0.81x / 0.46x at 64 (512/128, 128/512).

#### Answers

GSM8K (1,319, 2,048 tokens) for every arm, BFCL simple (400) for the vLLM arms; one request at a time.

| Arm | GSM8K | − vllm, paired [95%] | GSM8K wall s | BFCL |
| --- | ---: | --- | ---: | ---: |
| `vllm` | 0.9007 | – | 3,630 | 0.920 |
| `vllm-text` | 0.8992 | −0.0015 [−0.0099, +0.0068] | 3,948 | 0.920 |
| `jax` | 0.8969 | −0.0038 [−0.0114, +0.0038] | 5,183 | – |
| `jax-bf16` | 0.9128 | +0.0121 [−0.0008, +0.0250] | 3,946 | – |
| `jaxrust` | 0.9113 | +0.0106 [−0.0008, +0.0227] | 3,594 | – |

- 🟢 Every arm is level with `vllm` on GSM8K (each paired range includes zero); the two bf16 builds lead by
  about one point, short of a clear lead.
- 🟢 `vllm` and `vllm-text` give the same BFCL answer on all 400 questions.

#### `vllm-text-d4` does not start on one v5e chip

vLLM with `google/gemma-4-E2B-it-assistant` drafting 4 tokens ran out of TPU memory in warmup, about 32
minutes after launch, at both settings tried:

| `--gpu-memory-utilization` | weights + drafter GiB | KV cache tokens | wanted | free |
| --- | ---: | ---: | ---: | ---: |
| 0.70 | 6.25 | 277,792 | 512 MiB | 429.5 MiB |
| 0.60 | 6.25 | 186,048 | 512 MiB | 450.2 MiB |

Lowering the setting shrank the KV cache by 91,744 tokens and left about the same memory free, so the failing
allocation likely comes out of the budget the setting sizes, after the KV cache is placed (inferred). A smaller `--max-num-seqs`
or a fixed KV size is untested. Logs: `tpu-vllm-v5e1-2b-q4w4a16emb4/one-vllm-text-d4{-oom-gmu070,}/server.log`.

#### Scope

One v5e chip per arm, three chips of the same type and zone, so a ratio between engines on different VMs is a
ratio between two v5e-1s. TorchTPU arms were dropped after their `one` jobs (the Rust build's GSM8K ran at about 29 s a question);
`vllm-text-d1` was not run. TPU memory in use while serving was not measured.
