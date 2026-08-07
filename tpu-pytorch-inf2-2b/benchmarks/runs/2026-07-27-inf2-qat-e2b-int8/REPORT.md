# Gemma 4 E2B QAT int8 requant on AWS Inferentia2 — phase 2 result

**Run:** 2026-07-26/27
**Build box:** `inf2.8xlarge` spot, `us-east-1b` (us-east-1f had no spot capacity),
AMI `ami-0c13e7feb3fe2e01e` (gemma4-optb-inf2-20260706), terminated after upload
**Model:** `google/gemma-4-E2B-it-qat-q4_0-unquantized`, int8-sym-g32 requant
**Recipe:** Option-B two-graph KV-cache path, `KV_BUCKET=32`, `KV_MAX=128`,
compiler args `--model-type transformer --auto-cast all --auto-cast-type bf16`

## What was built

1. `quant/quantize_int8.py` (numpy+stdlib, torch-free) quantized 519
   decoder-layer linears (2.32 B params) to symmetric int8, group size 32
   along in_features, fp32 scales. Worst tensor mean relative error 0.74 %
   (a vision-tower projection); non-layer tensors (embeddings, PLE
   projections, norms, `lm_head`) passed through byte-identical, so the
   slim-host serving path needs no changes.
   Exclude regex used: `'^(?!.*\.layers\.)|embed|lm_head'`.
2. `quant/qat_e2b_int8_trace.py` swaps every quantized `nn.Linear` for a
   `QuantLinear` storing `weight_i8` + `weight_scale` and dequantizing to
   bf16 in-graph; the CPU reference runs the same int8 math.

## Results

| Variant | Decode (fixed 96-step) | Neff size | SEQ_MATCH |
| --- | ---: | ---: | --- |
| bf16 QAT baseline (7/25) | 46.3 tok/s | 3.42 GB | True |
| int8, `INLINE=1` (weights inlined) | **47.3 tok/s** | 5.52 GB | True |
| int8, `INLINE=0` (weight separation) | **12.0 tok/s** | 5.73 GB | True (+ reload parity) |

## Conclusion: int8 gives no perf win on this path

- **Inlined:** the Neuron compiler constant-folds the in-graph dequant back
  into full-precision weights — identical speed to bf16 and a *larger* neff.
- **Weight-separated:** folding is prevented but execution is ~4× slower;
  unusable.

**Revised diagnosis (2026-07-27):** decode IS weight-bandwidth-bound — the
arithmetic settles it. 3.4 GB of in-graph weights at 21.1 ms/step is
~160 GB/s effective, right at Inferentia2's HBM bandwidth (~190 GB/s spec).
Both results follow from one mechanism:

- `INLINE=1`: folding stores bf16 constants → same 3.4 GB read/step → same
  speed as bf16.
- `INLINE=0`: the un-folded dequant materializes a bf16 temp through HBM
  each step (read 2.3 GB int8 + write 4.6 GB bf16 + read it back ≈ 11.5 GB
  @ 160 GB/s ≈ 72 ms predicted; 83 ms observed) — the same materialization
  penalty the TPU repo measured for its in-graph path.

So int8 failed not because bandwidth doesn't matter, but because this
stack offers no way to keep the per-step weight read compressed: the
tracer folds inline constants, and separated weights pay the
materialization tax. Only a fused NKI dequant-matmul kernel (weights never
leave HBM as bf16) would break that — not worth it at E2B size.
**The bf16 QAT graphs remain the production serving artifact.** No int8
serving box was deployed — it would be strictly worse (same speed, added
quantization error). Bandwidth-boundedness has a silver lining: lockstep
batching should be nearly free (weights are read once per step regardless
of batch) — measured in the follow-up batch-4 run.

## Side result: the 23 tok/s mystery is resolved

The 2026-07-26 smoke run measured ~23 tok/s over HTTP and flagged a possible
2× regression vs the 46 tok/s baseline. The in-process fixed-length decode
benchmark on identical-recipe graphs reproduces **47.3 tok/s**, confirming
the HTTP number was request/latency measurement artifact, not a graph
regression.

## Artifacts (`s3://xbill-gemma4-patches-2b/qat-e2b-int8/`)

- `model_int8_g32.safetensors` (7.6 GB) + `model_int8_g32.quant_config.json`
  (per-tensor error stats) — the migration-ready int8 checkpoint; still the
  planned interchange format for the TorchTPU port at GA.
- `qat_e2b_int8_{prefill,decode}.pt` (inlined) and `..._sep_*.pt`
  (weight-separated) neffs, both SEQ_MATCH-validated.
- `quantize_int8.py`, `qat_e2b_int8_trace.py`, `build_int8*.sh`.

## Gotcha fixed along the way

`libneuronpjrt-path` lives in `/opt/aws_neuronx_venv_pytorch_2_8/bin/`, not
`/opt/aws/neuron/bin` — invoking the venv python by absolute path without
the venv bin on PATH breaks the `torch_neuronx` import.

## Cross-platform context (tpu-pytorch, 2026-07-25/26 runs)

The TorchTPU repo ran the mirror-image experiments on v6e-1 and the results
sharpen the diagnosis (insights only — no TPU-side code or files move to
AWS):

- **int8-in-graph lost there too, for a measured reason.** On TPU the
  in-graph dequant path is bandwidth-bound: XLA materializes the bf16 temp
  regardless of storage dtype, so bytes-read decides — packed int4 in-graph
  is a win (E2B 157.5 tok/s at 3.56× less HBM) while int8 in-graph is a
  regression (116.5). Their rule is symmetric: **in-graph path → int4;
  custom-kernel path → int8** (nibble unpack is VPU-bound, int8 needs no
  unpack). Our int8-in-graph choice sat on the wrong side of both halves.
- **Why TPU keeps the memory win and inf2 can't:** `torch.compile` treats
  weights as runtime buffers, so the dequant survives compilation;
  `torch_neuronx.trace` bakes weights as compile-time constants, so the
  Neuron compiler legitimately folds the dequant (our INLINE=1 result). The
  inf2 analog of their Pallas escape would be an NKI custom kernel — only
  worth it at 12B-class sizes, which don't fit inf2.xlarge anyway.
- **The structural levers are quantified now.** On E2B, device-resident
  static KV cache + lockstep batch-4 reached 809.9 tok/s aggregate
  (202.5/stream) vs 227.6 single-stream no-cache; and the no-cache design
  (analog of Option-B's KV-as-graph-I/O) degrades 2.5× at 1024 context
  while cached decode stays flat. Weight compression was worth at most
  ~30% there; KV+batch structure was worth 3.6×.

## If more decode speed is ever needed on inf2

Weight compression is the wrong lever (confirmed on both platforms for
E2B-class models). In order of expected value, per the TPU evidence:
device-resident KV cache (drop the per-token host↔device KV round trip),
lockstep batching (amortizes per-step fixed costs), larger KV buckets,
or the NxD inference stack. A fused NKI dequant-matmul kernel is the
last resort and only pays at model sizes inf2.xlarge can't hold.

## Migration-at-GA design update

The TorchTPU port loads the **public** `google/gemma-4-E2B-it-qat-w4a16-ct`
checkpoint directly (int4→int8 conversion at load time, zero port changes).
Our `model_int8_g32.safetensors` is therefore NOT the migration interchange
artifact — it stays an inf2-local experiment record. The migration surface
reduces to one thing: the OpenAI-compatible /v1 HTTP contract, which
`optb_server_qat.py` already serves. TPU-side E2B reference points for the
eventual cutover: 157.5 tok/s (int4 in-graph, no cache, b1) to 809.9 tok/s
aggregate (cached port, int8 kernel, b4) on one v6e-1, vs 47.3 tok/s here.
