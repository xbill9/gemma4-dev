# gpu-pytorch-g5g-2b

Serve **`google/gemma-4-E2B-it`** with **PyTorch + transformers** on **AWS EC2 G5g** — an AWS
Graviton2 (64-bit Arm) host paired with an **NVIDIA T4G** (Turing, SM 7.5).

> **Status: this rig has served, once.** Forked from
> [`gpu-jax-g5g-2b`](../gpu-jax-g5g-2b/) on 2026-08-28; first serve **2026-08-29** on a
> `g5g.2xlarge` spot instance — **10.88 tok/s decode, 0 degenerate**
> ([`benchmarks/runs/2026-08-29-first-serve-g5g/`](benchmarks/runs/2026-08-29-first-serve-g5g/)),
> then profiled and fixed the same day
> ([`.../2026-08-29-profile-and-fixes-g5g/`](benchmarks/runs/2026-08-29-profile-and-fixes-g5g/)).
> 92 tests pass offline.
>
> **The profile's headline: batching is worth 7.84x and is very nearly free** — per-step time
> grew 2.0% while the batch grew 8x, reaching **84.16 tok/s at B=8** for 0.258 GB. That beats
> both siblings outright, but only on the engine: `MAX_NUM_SEQS=1` means the served path cannot
> reach it yet. Continuous batching is the top-ranked work.
>
> That first launch found **six fatal bugs in the deploy path**, none of which the offline tests
> could see, because none of them asserted on the rendered bootstrap — `verify_gpu` imported
> `jax`, the systemd unit pointed at `jax_openai_server.py`, `_serve_argv` emitted the JAX port's
> flags, the pip spec was quoted as one requirement, the payload called
> `torch.compile(backend="tpu")`, and the DLAMI's torch turned out to live in a **Python 3.13
> venv** that `TORCH_PYTHON_VERSION=3.12` could never have found. See `CLAUDE.md`.
>
> Figures below that are **not** from that run were measured on the **JAX sibling**, not here.

## Why this exists

**Slot 2 is the only thing that moves**, so this is a clean runtime A/B on identical silicon.

`gpu-jax-g5g-2b` serves at 13.10 tok/s and spends **54.0% of decode in dtype conversion at 0.0%
TensorCore utilisation** — a finding nobody has explained, and one that survived converting the
checkpoint to float16. A completely different framework on the same chip is the cheapest way to
ask whether that is *the chip* or *the framework*.

| | `gpu-jax-g5g-2b` | **this rig** |
| --- | --- | --- |
| runtime | pure JAX + XLA | **PyTorch + transformers** |
| model code | this repo's own port, ~2,030 lines | **`AutoModelForCausalLM`** |
| CUDA comes from | `pip` (`jax[cuda13]`) | **the AMI** |
| AMI | base OSS driver, Ubuntu 26.04 | **PyTorch 2.12, Ubuntu 24.04** |
| host / GPU | Graviton2 · T4G, SM 7.5 | *same* |
| compute dtype | `float16` (device-selected) | `float16` (device-selected) |

## The thing that makes this rig cheap: torch ships in the AMI

`docs/turing-aarch64-gap.md` measured the AWS ARM64 DLAMI's torch and found **Turing already in
its arch list**:

```
torch 2.12.0+cu132  arch_list ['sm_75','sm_80','sm_90','sm_100','sm_110','sm_120']
```

So there is **no pip CUDA install, no build and no toolkit** — torch is already there, already
built for this GPU. The rig only installs `transformers accelerate` on top.

**Do not `pip install torch` here.** Upstream PyPI aarch64 wheels omit `sm_75`; you would get a
silent CPU fallback or a from-source build. `test_torch_is_not_installed_by_pip` asserts the pip
spec stays clean, and the AMI test asserts we stay on the PyTorch line rather than reverting to
the base driver image the JAX rig uses.

**The PyTorch DLAMI line is alive**, contrary to what the JAX rig's notes imply. That rig's
warning is about the frozen `pytorch-2.7-ubuntu-22.04` line specifically; AWS has since shipped
2.8 → **2.12** on Ubuntu 24.04, rebuilt 2026-07-24.

## Turing still decides the dtype

**bfloat16 on this chip does not raise — it emulates through fp32.** That is how the JAX sibling
lost 86.8% of decode to a one-line dtype mistake, and PyTorch fails the same way: CUDA accepts
bf16 on Turing and quietly routes it through fp32.

So `resolve_compute_dtype()` reads the live compute capability and picks `float16` below SM 8.0,
logging what it resolved:

```
torch device policy: name=NVIDIA T4G compute_capability=7.5 pre_ampere=True compute_dtype=float16
```

The guard lives in **both** `torch_openai_server.py` and `torch_generate.py`, because either can
be run alone.

## What is NOT inherited, and why

- **No `ports/gemma4/`.** The JAX rig's hand-written model is the thing PyTorch replaces; here
  `transformers` owns the model definition. Gemma 4's heterogeneous head dims (sliding 256,
  global 512) are transformers' problem now — **and whether its attention path handles them on
  Turing is unverified.**
- **No XLA compilation cache.** The S3 cache and its systemd timer are JAX/XLA-specific. torch
  has its own inductor cache with different semantics; nothing has been wired, and the JAX
  reasoning must not be carried across.
- **No W4A16 Pallas guard.** That kernel is a JAX/Pallas artefact. This rig serves the dense
  reference checkpoint at float16, so the name carries no encoding slot — same conclusion, a
  different route.
- **Not `torch_xla`.** It publishes x86_64 wheels only, so there is nothing to install on
  Graviton2, and its CUDA backend is deprecated with the nightly builds already removed. That is
  a PyTorch/XLA packaging decision rather than an XLA limitation — `jax-cuda13-pjrt` ships an
  aarch64 CUDA build today, which is exactly what the JAX sibling runs on. The full workings,
  including why plain CUDA is also the better experiment, are in
  [`docs/INHERITED.md`](docs/INHERITED.md).

## Quickstart

```bash
pip install -r requirements.txt
python3 -m unittest discover -s tests -v   # 89 tests, fully offline
```

Then through the MCP tools (`mcp__gpu-pytorch-g5g-2b__…`):

```
create_g5g_instance → get_install_progress → verify_gpu_arch → deploy_torch_server
                    → get_torch_logs → verify_model_health → query_model → get_metrics
```

**Always `make skill` before `deploy_torch_server`** — the deploy ships the skill snapshot, not
the working tree.

## The first things to check on a real run

1. **The device-policy line.** If it says `bfloat16`, detection failed and every number after it
   is meaningless.
2. **Whether transformers' attention handles Gemma 4 on SM 7.5.** This is the open risk. vLLM
   forces a Triton kernel here that wants 98,304 bytes of shared memory against Turing's 65,536;
   transformers' SDPA path should avoid that, but nothing here has proven it.
3. **Decode against the JAX sibling's 13.10 tok/s** — the whole reason the rig exists.

## License

Apache-2.0 — see [`../LICENSE`](../LICENSE).
