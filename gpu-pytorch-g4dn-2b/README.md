# gpu-pytorch-g4dn-2b

Serve **`google/gemma-4-E2B-it`** with **PyTorch + transformers** on **AWS EC2 G4dn** — an
x86_64 Intel host paired with an **NVIDIA T4** (Turing, SM 7.5, **15360 MiB measured**, 14.07 GiB
usable).

> **Status: this rig has served nothing.** Forked from
> [`gpu-pytorch-g5g-2b`](../gpu-pytorch-g5g-2b/) on 2026-08-29 and retargeted from Graviton2 to
> x86_64. 81 tests pass offline. No instance has been launched, no weights loaded, no token
> generated, and `benchmarks/` is deliberately empty. **Every number quoted below was measured
> on another rig** — [`docs/INHERITED.md`](docs/INHERITED.md) is the list of what carries.

## Why this exists

Four rigs form a 2×2 over {runtime} × {host}. The GPU is the same Turing generation in both
columns — T4G on G5g, T4 on G4dn, both SM 7.5, both **15360 MiB** — so the **column is the
host** and the **row is the runtime**:

| | G5g (Graviton2, aarch64) | G4dn (x86_64) |
| --- | --- | --- |
| pure JAX | [`gpu-jax-g5g-2b`](../gpu-jax-g5g-2b/) ✅ | [`gpu-jax-g4dn-2b`](../gpu-jax-g4dn-2b/) ✅ |
| PyTorch | [`gpu-pytorch-g5g-2b`](../gpu-pytorch-g5g-2b/) | **this rig** |

`gpu-jax-g5g-2b` serves at 13.10 tok/s and spends **54.0% of decode in dtype conversion at 0.0%
TensorCore utilisation**. That is *not* a storage-dtype problem — converting the whole parameter
tree to float16 moved throughput **+0.0%**, because at `B=1` cuBLAS dispatches an all-fp32 GEMV
that has no half path at all.

**The row is now answered.** `gpu-jax-g4dn-2b` first served on **2026-08-29** and landed on top
of its G5g sibling: **13.1 tok/s against 13.10**, weight bytes equal to the byte, and an xprof
profile reproducing 54.4% conversion / 32.8% fp32 GEMV / 0.0% TensorCore with roofline peaks
identical to three decimals. The host contributes **nothing measurable** to decode, so the 86.9%
tax is a **Turing** property rather than a Graviton2 one.

**The column is what this rig is for: is that tax the chip, or is it XLA?** `gpu-jax-g4dn-2b` is
the same host, chip, checkpoint and dtype policy, which makes this the **cleanest runtime A/B in
the tree** — cleaner than the G5g pair, whose vLLM side used hand-reduced Triton tiles.

**The baseline to beat or match** (median of 3, warmed at the measured shape, 64 output tokens):

| input tokens | 41 | 521 | 2,057 |
| --- | ---: | ---: | ---: |
| `tpu_jax_decode_tokens_per_second` | 13.1 | 13.2 | 13.1 |
| end-to-end `output_tok_per_s` | 12.671 | 11.654 | 8.748 |

Compare on the **gauge**: it is flat in context, while end-to-end falls because it carries
prefill and HTTP.

## What changed from the G5g PyTorch rig

Only the host — but the host reaches further than it looks.

| | `gpu-pytorch-g5g-2b` | **this rig** |
| --- | --- | --- |
| host | Graviton2, aarch64 | **x86_64 Intel** |
| GPU | T4G, SM 7.5, 15360 MiB | **T4**, SM 7.5, 15360 MiB (*identical* — measured) |
| default size | `g5g.2xlarge` (8 vCPU, 16 GiB) | **`g4dn.xlarge`** (4 vCPU, 16 GiB) |
| AMI line | PyTorch 2.12, Ubuntu 24.04 | **PyTorch 2.13, Ubuntu 26.04** |
| system interpreter | 3.12 (+ deadsnakes) | **3.14** (ships with 26.04) |
| runtime | PyTorch + transformers | *same* |
| compute dtype | `float16` (device-selected) | *same* |

Two of those are traps rather than upgrades, and both are written up in
[`docs/INHERITED.md`](docs/INHERITED.md):

- **The `describe-images` fallback had to be rewritten, not carried.** AWS names the two
  architectures' images in *different word order* — `Deep Learning ARM64 AMI OSS Nvidia Driver
  GPU PyTorch …` against `Deep Learning OSS Nvidia Driver AMI GPU PyTorch …` — so the sibling's
  pattern matches **zero** x86_64 images. Verified 2026-08-29. It would have failed only when
  SSM was also unavailable, which is exactly when the fallback is load-bearing.
- **`TORCH_PYTHON_VERSION` moves with the AMI.** deadsnakes publishes `python3.14` for jammy and
  noble only, so a 26.04 image needs 3.14 and an older one must not ask for it.

## Torch ships in the AMI, and that is the whole design

The bootstrap installs **into** the DLAMI's own PyTorch environment rather than beside it, which
is the exact inverse of the JAX rigs: there the image supplies only a driver because pip supplies
CUDA. Here the image supplies torch, so the AMI is a **PyTorch** DLAMI rather than the base one,
and the rig installs `transformers accelerate` on top and nothing else.

`install.sh` **probes** for the interpreter that can already `import torch` and writes the
resolved path to `PYTHON_BIN`, because the venv path moves between DLAMI releases. Installing
into `/usr/bin/python3` and pointing the unit there yields `ModuleNotFoundError: No module named
'torch'` *after* the install reports success.

**A box with no torch is a wrong AMI, not a missing `pip install torch`.** A base driver-only
DLAMI boots perfectly well and fails at the torch-interpreter stage.

> ⚠️ **The G5g rig's reason for this does not carry.** It states that upstream PyPI wheels omit
> `sm_75`; that was measured for **aarch64**. Upstream x86_64 CUDA wheels have carried Turing for
> years. Taking torch from the AMI is still right — vendor build on a vendor driver — but *that*
> is not the argument here, and `verify_gpu_arch` is what settles it on any given image.

## Turing still decides the dtype

**bfloat16 on this chip does not raise — it emulates through fp32.** Correct numbers, quiet
slowdown, which is worse than an error.

`resolve_compute_dtype()` reads the live compute capability and picks `float16` below SM 8.0,
then logs what it resolved:

```
torch device policy: name=Tesla T4 compute_capability=7.5 pre_ampere=True compute_dtype=float16
```

The guard lives in **both** `torch_openai_server.py` and `torch_generate.py`, because either can
be run alone, and `tests/test_engine.py` asserts the two agree and that the boundary is SM 8.0
rather than a chip name.

## Instance sizing

`g4dn.xlarge` is the default and **every g4dn size is supported** — unlike the G5g sibling, which
rejects its 8 GiB size, this family starts at 16 GiB.

Two properties of this family read as typos and are not:

- **The size suffix does not give the GPU count.** `g4dn.16xlarge` carries **one** T4;
  `g4dn.12xlarge` carries **four**.
- **vCPU is RAM/4, not RAM/2.** A `g4dn.xlarge` has 16 GiB and **4** vCPUs. The G-family quota is
  counted in vCPUs, so a figure derived the G5g way is wrong where the number is used.

Only `g4dn.xlarge` gets a swapfile. That is the size the rig launches, so unlike the sibling —
where the block stayed latent behind a size nobody launched — it is exercised from the first
launch. **A bigger instance buys host RAM and vCPUs, never device memory:** the engine is
single-device, so on `12xlarge` three T4s idle and on `metal`, seven.

## What is NOT here, and why

- **No `ports/gemma4/`.** The JAX rigs' hand-written model port is the thing `transformers`
  replaces — and with it the KV ring, the padding-eviction bug in it, the PLE quantiser, and the
  fused W4A16 Pallas guard.
- **No compilation cache.** Nothing on this path compiles. The fork carried
  `JAX_COMPILATION_CACHE_DIR`, `JAX_CACHE_S3_URI` and a systemd timer syncing an empty directory
  to S3 every ten minutes, reporting success forever. **Removed**, with a test asserting it stays
  removed. If `torch.compile` is adopted the knob is `TORCHINDUCTOR_CACHE_DIR`.
- **No W4A16 path.** Same conclusion as the JAX rigs — dense reference checkpoint, so the name
  carries no encoding slot — by a **different route**: there is no Pallas here, and
  `AutoModelForCausalLM` has nowhere to put w4a16 weights without bitsandbytes or torchao.
- **Not `torch_xla`.** Note the sibling's first reason is **false on this platform**:
  `torch_xla` 2.9.0 does publish `manylinux_2_28_x86_64`. What still stands is that PyTorch/XLA's
  CUDA backend is deprecated with the nightly builds removed, and that plain CUDA is the better
  experiment anyway — `torch_xla` would make this JAX→XLA vs PyTorch→XLA, two frontends over one
  compiler, where plain PyTorch makes it XLA vs not-XLA.

## Quickstart

```bash
pip install -r requirements.txt
python3 -m unittest discover -s tests -v   # 81 tests, fully offline
```

Then through the MCP tools (`mcp__gpu-pytorch-g4dn-2b__…`):

```
create_g4dn_instance → get_install_progress → verify_gpu_arch → deploy_torch_server
                     → get_torch_logs → verify_model_health → query_model → get_metrics
```

**Always `make skill` before `deploy_torch_server`** — the deploy ships the skill snapshot, not
the working tree.

## The first things to check on a real run

1. **`verify_gpu_arch`, before anything else.** It prints `torch.cuda.get_arch_list()` next to
   the measured capability and runs a real fp16 matmul. On this rig it is also the only thing
   that settles what the image's torch build actually covers.
2. **The device-policy line.** If it says `bfloat16`, detection failed and every number after it
   is meaningless.
3. **Whether transformers' attention handles Gemma 4 on SM 7.5.** The open risk. vLLM is forced
   onto a Triton kernel here that wants 98,304 bytes of shared memory against Turing's 65,536;
   transformers' SDPA path should avoid that, but nothing has proven it.
4. **Decode, against the JAX rigs** — the reason the 2×2 exists. Quote the server's
   `tpu_jax_decode_tokens_per_second` gauge, not an end-to-end rate, and warm up at the shape you
   measure.

## License

Apache-2.0 — see [`../LICENSE`](../LICENSE).
