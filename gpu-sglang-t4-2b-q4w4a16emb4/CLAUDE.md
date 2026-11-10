# CLAUDE.md — gpu-sglang-t4-2b-q4w4a16emb4

SGLang 0.5.23 on the T4, serving the 3050 rig's checkpoint,
`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` @ `db715f83` (W4A16 int4 linears and int4 vocabulary
tables, text only), 3.25 GB on the card. Forked 2026-11-10 from `gpu-sglang-t4-2b-q4_0`, which serves the
same values as 16-bit (10.00 GB) because SGLang's own Marlin is empty below sm_80.

- **Four patches** (`patches/`): the 3050 rig's two, `turing-smem.diff` from the q4_0 rig, and
  `marlin-sm75-via-vllm.diff`. The last makes SGLang's `gptq_marlin_gemm` and `gptq_marlin_repack` call
  vLLM 0.33's compiled `torch.ops._C.marlin_gemm` / `gptq_marlin_repack` when compute capability < 8 (no
  act-order there), and lowers the two sm_80 gates (`query_marlin_supported_quant_types`,
  `CompressedTensorsWNA16.get_min_capability`) to 75. It also turns Marlin's atomic-add reduction off unless
  `SGLANG_MARLIN_USE_ATOMIC_ADD=1`, as vLLM does: SGLang 0.5.23 hard-codes it on (`if not True:`), which
  fires only for float16, so only here, and made greedy text vary between repeats (3 of 6 prompts).
  **It needs vLLM installed beside SGLang**, as it is in
  `/opt1/pyuser-engines`. Originals of the four patched files are in `~/sglang-orig-2026-11-10/` on the host.
- **This rig's `int4-embeddings.diff` differs from the 3050's in two lines**: unpacked rows, and the dense
  table the drafter reads through `.weight`, take the dtype the model is built in instead of bf16. Turing's
  Marlin takes only fp16 (or int8) activations, SGLang builds the model in float16 below sm_80, and a bf16
  table beside float16 hidden states gave the drafter float32 activations.
- `kernel_check.py`: SGLang's repack + scale permutation + GEMM on this card against fp16 matmul on the
  dequantized weights (relative error 0.7–2.0e-3 at E2B's shapes, 1–256 rows, 2026-11-10).
- **FlashInfer JIT on sm_75** needs `CUDA_HOME=/opt1/cuda-13.4` and `CPATH=/opt1/cuda-extra-include`;
  `serve.py` exports both from `tpu.env`.

## Shared by the six T4 engine rigs

- **Host:** GCE `n1-standard-2` (2 vCPU, 7.4 GiB RAM), Tesla T4 (TU104, sm_75, 15,360 MiB), driver 620.70.06.
  `sudo swapon /opt1/swapfile` after a reboot (16 GB, not in fstab); the 16-bit SGLang load and vLLM need it.
- **Interpreter:** `/usr/bin/python3.13` with `PYTHONUSERBASE=/opt1/pyuser-engines` (vLLM 0.33.0, SGLang
  0.5.23, litert-lm 0.19.0, torch 2.14.0+cu130, transformers 5.20.2). `python3` on this host is pyenv 3.12
  with a CPU torch. `/opt1/pyuser` (vLLM 0.29) belongs to `gpu-vllm-t4-2b{,-w4a16}` and is left alone.
- **One engine on the card at a time.** Ports: llama.cpp 8080, vLLM 8010, mistral.rs 8020, SGLang 8030,
  PyTorch 8100, LiteRT 9379; the older vLLM T4 rigs use 8000.
- **Forked 2026-11-05** from the `local-*-3050oem-2b-*` rig of the same engine, for the T4 run of
  `docs/serving-engines-3050`. Same weights by SHA-256 or revision, same engine versions, except where
  this file says otherwise. Code came across; measurements did not.
- Tests are `unittest`. Never a virtualenv. `tpu.env` decides and is committed.
