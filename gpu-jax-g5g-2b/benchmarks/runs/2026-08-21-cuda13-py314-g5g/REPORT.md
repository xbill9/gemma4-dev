# 2026-08-21 — CUDA 13 + Python 3.14 on EC2 G5g (Graviton2 + NVIDIA T4G)

**Purpose: prove the `jax[cuda13]` / Python 3.14 bump boots and serves, and measure whether
it moved throughput.** It boots, it serves, and it does not move throughput.

Same AMI, same instance size, same checkpoint and same serving flags as
`2026-08-19-first-serve-g5g`, so the runtime is the only variable.

## Result

| | 2026-08-19 (CUDA 12 / py3.12) | **2026-08-21 (CUDA 13 / py3.14)** | delta |
| --- | ---: | ---: | ---: |
| Decode gauge (warm) | 12.5 tok/s | **12.4 tok/s** | −0.8% |
| End-to-end (warm) | 12.00 tok/s | **11.78 tok/s** | see caveat |
| Weight load | 158.8 s | **160.4 s** | +1.0% |
| GPU memory in use | 13,573 MiB | **13,581 MiB** | +8 MiB |
| Runtime install | ~5 min | **~4 min** | — |

**The bump is performance-neutral. Cite it as currency, not speed.**

**Caveat on the end-to-end row.** It is not a clean comparison and should not be quoted as a
regression. The two runs generated different completion lengths from the same prompt — 64
tokens on 2026-08-19, **53** here, both `finish_reason: stop` — and `usage.prompt_tokens`
came back as **20** here against the 17 the earlier report records for the identical prompt
string. End-to-end wall includes prefill and HTTP, so a different token count moves it.
The server's own `tpu_jax_decode_tokens_per_second` gauge is the like-for-like number, and it
reads 12.4 against 12.5.

## Environment

| | |
| --- | --- |
| Instance | `g5g.2xlarge` spot, **`us-east-1d`**, `i-08639f402a3c3e76b`, $0.395/hr |
| AMI | `ami-077792d0bb6a000b8` — Deep Learning ARM64 AMI OSS Nvidia Driver GPU PyTorch 2.7 (Ubuntu 22.04) 20260501 — **identical to the baseline run** |
| GPU | NVIDIA **T4G**, compute capability **7.5**, **15,360 MiB**, driver 580.126.09, `CUDA Version: 13.0` |
| Python | **3.14.6** (deadsnakes `3.14.6-1+jammy1`) |
| JAX | jax 0.11.1, jaxlib 0.11.1, **jax-cuda13-plugin** 0.11.1, **jax-cuda13-pjrt** 0.11.1 |
| CUDA | from pip: **nvidia-cublas 13.6.1.10**, nvidia-cuda-runtime 13.3.29, nvidia-cuda-nvcc 13.3.73, nvidia-cudnn-cu13 9.24.0.43, nvidia-nccl-cu13 2.31.2. No toolkit, no `nvcc` binary from apt. |
| Other | transformers 5.15.1, fastapi 0.141.1, uvicorn 0.52.4, numpy 2.5.2, scipy 1.18.0, safetensors 0.8.0, jinja2 3.1.6 |
| Serving flags | `--kv-cache-dtype auto --quant-mode fp16 --max-model-len 8192` |

### Every dry-run prediction held

`tpu.env` predicted the resolved stack on 2026-08-20 from `pip install --dry-run`. On real
hardware it matched **exactly** — `nvidia-cublas 13.6.1.10`, `nvidia-cuda-runtime 13.3.29`,
`nvidia-cuda-nvcc 13.3.73`. The unsuffixed CUDA 13 naming held, and the documented exceptions
(`nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13`) kept their suffix.
`safetensors 0.8.0` installed its `cp310-abi3` wheel under 3.14, confirming the
forward-compatibility note.

## Throughput, measured

Five end-to-end runs from a laptop over the public endpoint, same prompt as the baseline
("In two sentences, what is a Graviton processor?"), `max_tokens=64`, `temperature=0`, after
two warm-up requests.

| run | wall (s) | completion tokens | tok/s |
| ---: | ---: | ---: | ---: |
| 1 | 4.50 | 53 | 11.78 |
| 2 | 4.50 | 53 | 11.77 |
| 3 | 4.50 | 53 | 11.77 |
| 4 | 4.50 | 53 | 11.79 |
| 5 | 4.50 | 53 | 11.79 |

Spread 0.024 tok/s. `tpu_jax_decode_tokens_per_second` read **12.4** over the same requests.

**Cold is still not warm.** The first request off a fresh engine took **18.06 s** for 53
tokens against 4.50 s warm — a 4.0x end-to-end ratio. The baseline's 56x figure is TTFT
specifically, which was not isolated here; this is the whole-request ratio and is not the
same measurement.

## Memory

| | Value |
| --- | ---: |
| Capacity | 15,360 MiB |
| In use while serving | **13,581 MiB** |
| Weights loaded | 9.26 GB reported by the engine, in 160.4 s |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 0.90 |

## Issues

| Issue | Status |
| --- | --- |
| **`apt-get update` wedged for 12 minutes at the first bootstrap step.** `us-east-1.ec2.ports.ubuntu.com` returned **503** over IPv4 and resolved to **AAAA records only**, on a host with no IPv6 address and no IPv6 default route. Not lock contention: apt held the lists lock itself and nothing was blocking it. Canonical `ports.ubuntu.com` answered 200 in 0.45 s. Worked around by hand on this instance; `install.sh` now sets `Acquire::http::Timeout`/`Acquire::Retries` and falls back to the canonical mirror. | fixed |
| **The stall was invisible.** `get_install_progress` reported "INSTALL IN PROGRESS" for the whole 12 minutes with the log mtime frozen — exactly the hiding failure mode `DPkg::Lock::Timeout` was added to prevent, via a path that option does not cover. | fixed |
| **Spot capacity refused in `us-east-1c` and `us-east-1b`** before `us-east-1d` accepted. Second run in a row where AZ-walking was required. | workaround |
| **Loader logs `Loading W4A16 QAT weights into JAX` while serving fp16.** Cosmetic but actively misleading — argv passes `--quant-mode fp16` and `jax_engine` resolves `fp16` correctly. Stale log string in the load path. | open |

## What was not measured

| Cell | Status | Why |
| --- | --- | --- |
| Concurrency > 1 | infeasible | Engine is `max_num_seqs=1`. |
| Context sweep | not run | Single-prompt validation, as with the baseline. |
| TTFT / prefill isolated | not run | Only whole-request wall was timed, so the baseline's 131 ms warm prefill was not re-measured. |
| Two-GPU sizes | infeasible | Engine is single-device. |
| `g5g.xlarge` + swapfile | not run | 16 GiB host needs no swap. |

## Reproduce

```bash
# Launch; expect to walk AZs. 1c and 1b both refused spot capacity.
#   create_g5g_instance -> get_install_progress -> verify_gpu_arch
#   deploy_jax_server   -> get_jax_logs        -> verify_model_health
# WARM UP before recording anything.
```

Measured 2026-08-21 on `g5g.2xlarge` spot, `us-east-1d`, instance `i-08639f402a3c3e76b`.
