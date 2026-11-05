# SGLang 0.5.23 on the Tesla T4 — first light, 2026-11-05

Tesla T4 (sm_75, 15,360 MiB, driver 620.70.06), GCE n1-standard-2 (2 vCPU, 7.4 GiB RAM, 16 GB swap).
SGLang 0.5.23 cp313 wheel `--no-deps` in `/opt1/pyuser-engines` beside vLLM 0.33.0, torch 2.14.0+cu130,
transformers 5.20.2, flashinfer-python 0.7.2.post1, sglang-kernel 0.4.10. The two 3050 rig patches applied.
A mistral.rs build was using both vCPUs throughout, so the startup times below are not representative.

Logs here have the `/health` polling and progress-bar lines removed.

## 1. The W4A16 checkpoint does not load on sm_75 (`first-start.log`)

`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` @ `db715f83`, the 3050 rig's checkpoint:

```text
RuntimeError: ('Quantization scheme is not supported for ', 'the current GPU. Min capability: 80. ', 'Current capability: 75.')
```

`CompressedTensorsWNA16.get_min_capability()` returns 80. Lowering it would not help: the Marlin kernel it
calls is JIT-compiled from `sglang/kernels/jit/csrc/gemm/marlin/`, and every kernel body there is inside
`#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800` → empty stub. (vLLM 0.33's Marlin runs on this card.)

So this rig serves `google/gemma-4-E2B-it-qat-q4_0-unquantized` @ `6befbaca`: the same QAT Q4_0-grid
values stored as bf16, served float16 (SGLang logs "Compute capability below sm80. Use float16").

## 2. Triton attention overflows Turing's shared memory (`probe-bf16.log`, `probe-bf16b.log`)

Probed with the cached `google/gemma-4-E2B-it` @ `3e22461f`. Decode CUDA-graph capture:

```text
triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 98304, Hardware limit: 65536.
```

in `_decode_grouped_att_m_fwd` (`decode_attention.py`). With that clamped, the first prefill:

```text
triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 65536.
```

in `extend_attention_fwd`. Gemma 4's full-attention layers are 512 wide. `patches/turing-smem.diff` sets, on
compute capability < 8 with head width ≥ 512: decode `BLOCK = 16`, `num_stages = 1` (the same move SGLang
already makes for MI3xx at width ≥ 576), and extend `BLOCK_M, BLOCK_N = (16, 16)` (was (32, 32)).

## 3. Serving (`probe-qatunq.log`)

QAT unquantized checkpoint, `--mem-fraction-static 0.88`, 64 running requests, both clamps:

- weights 10.00 GB on the card (multimodal checkpoint, `Gemma4ForConditionalGeneration`), load 1,142.73 s
  (one 9.6 GB safetensors file on 7.4 GiB of RAM, swapping, with the cargo build running)
- full-attention KV 104,831 tokens, sliding-window KV 45,121 tokens, float16
- decode CUDA graphs captured for 1–64 requests in 115.09 s
- 14,151 MiB in use on the card
- "Name the planets in one line." → `Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.`
- a 717-token prompt (prefill through the clamped extend kernel) → a correct one-sentence summary

FlashInfer's sm_75 JIT (used by `sgl_kernel` norm ops on this card) needs `CUDA_HOME=/opt1/cuda-13.4` and
`CPATH=/opt1/cuda-extra-include` (symlinks to the curand headers only; see `tpu.env`).

Not measured yet: speed, accuracy, the drafter.
