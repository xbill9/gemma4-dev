# E2B data-type sweep on one MI300X

Written 2026-10-07, before a droplet was created for it. Everything above "Results" stays as
written once the first arm starts; changes go in dated amendments at the end.

## The question

Which of the E2B repacks published under `xbill9/` serve on an MI300X through the shipped ROCm vLLM
image, and how fast is each against bf16 on the same card, image and day?

## What the card supports

Measured on the part 2026-09-16 and 2026-09-18; `../HARDWARE.md` §MI300X has the evidence.

| Format | gfx942 | What it means for an arm |
| --- | :---: | --- |
| bf16 / fp16 | yes | the reference; 1307 TFLOP/s peak |
| fp8 `e4m3fnuz` | yes | the only format faster than bf16 at the GEMM level: 1.50x summed over E2B decode shapes |
| fp8 `e4m3fn` | no | raw GEMMs raise `HIPBLAS_STATUS_NOT_SUPPORTED`. The fp8 repacks store `e4m3fn`; vLLM's compressed-tensors path converts to `fnuz` at load, which this sweep tests |
| int8 | M > 16 only | `torch._int_mm` 0.20x bf16 at M = 64; vLLM's W8A8 path uses its own scaled-mm kernels |
| int4 (W4A16) | no native int4 math | dequantized in `TritonW4A16LinearKernel`, the only ROCm W4A16 path for gfx942 + bf16 + group 32 |
| fp4 / MX | no | gfx950 and later |
| GGUF | — | compiled out of the ROCm image |

Stage 0 re-checks this on the new droplet: `gemm_decode_shapes.py` (bf16 / fp8 / int8, graph
replay) on the pinned image, before any arm runs.

## Arms

One rig per checkpoint. The rigs differ only in the checkpoint lines of `tpu.env`.

| Encoding | Rig | Checkpoint | GB |
| --- | --- | --- | ---: |
| bf16 | `gpu-vllm-mi300x-2b` | `google/gemma-4-E2B-it`, served text-only | 10.28 |
| q4w4a16 | `gpu-vllm-mi300x-2b-q4w4a16` | `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text` | 6.59 |
| w8a8 | `gpu-vllm-mi300x-2b-w8a8` | `xbill9/gemma-4-E2B-it-qat-w8a8-int8` | 7.41 |
| fp8 | `gpu-vllm-mi300x-2b-fp8` | `xbill9/gemma-4-E2B-it-qat-q4_0-fp8-text` | 7.42 |
| q4w4a16ple4 | `gpu-vllm-mi300x-2b-q4w4a16ple4` | `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-ple4` | 3.22 |
| q4w4a16emb4 | `gpu-vllm-mi300x-2b-q4w4a16emb4` | `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` | 2.86 |
| w8a8emb4 | `gpu-vllm-mi300x-2b-w8a8emb4` | `xbill9/gemma-4-E2B-it-qat-w8a8-ct-text-emb4` | 3.69 |
| fp8emb4 | `gpu-vllm-mi300x-2b-fp8emb4` | `xbill9/gemma-4-E2B-it-qat-q4_0-fp8-text-emb4` | 3.69 |

Not arms: `-qat-q4_0-w4a16-ct` (multimodal; the same linears as `q4w4a16`), `-qat-q4_0-exact-gguf`
(GGUF is compiled out of the image) and `-inferentia2` (Neuron-compiled).

## Method

- **One droplet, one session, one image digest.** `check_image` runs on
  `vllm/vllm-openai-rocm:nightly-rocm100`, and `dtype_sweep.py --image` takes the `@sha256:` it
  resolves to; the driver refuses a tag.
- Shared serving settings from each rig's `tpu.env`: `--max-model-len 32768`,
  `--gpu-memory-utilization 0.90`, prefix caching on, the gemma4 reasoning and tool-call parsers,
  no `--quantization` flag. The bf16 arm runs with `LIMIT_MM_PER_PROMPT='{"image": 0, "audio": 0}'`.
- Grid: concurrency 1 / 8 / 64 × input 128 / 1024 / 8192, output 512, 3 repeats, a distinct
  `--seed-base` per arm.
- Per arm the driver files `boot.log`, `boot-highlights.txt`, `verify_capabilities.txt` and the
  suite output under that rig's `benchmarks/runs/<run-id>/`, with the report in
  `benchmarks/reports/`. `dtype-summary.json` (here, under the same run id) holds every ratio to
  bf16, computed by the driver.
- **Boot checks before any number is read:** the linear kernel named in the log, `Model loading
  took` (a quantized arm near bf16's ~10 GiB was dequantized at load), and the KV pool against
  bf16's.
- An arm that does not serve within 30 minutes is recorded as failed with its log. **No patching
  inside this sweep**; a patched image is a separate study.

## Predictions

- `q4w4a16`: 0.8x–1.2x bf16 at every cell. E2B decode is bound by overhead, not weight bandwidth.
- `w8a8`: below 1.0x. The card's int8 GEMMs measured slower than bf16.
- `fp8`: loads and serves as `e4m3fnuz`. Online fp8 served at 0.53x–0.80x bf16 on the 0.19.1 image;
  the checkpoint route is expected to land in the same range.
- `*emb4`, `*ple4`: fail at load on `weight_packed` for the embedding tables, unless the nightly's
  `VocabParallelEmbedding` takes a quantization method from compressed-tensors.

## Run

```bash
# after the droplet exists, is tagged `gemma`, and ~/amd-gputools `make scaffold` has run
python3 dtype_sweep.py --droplet <name> --image vllm/vllm-openai-rocm@sha256:<digest> \
    --run-id <date>-dtype-sweep-mi300x
```

Re-running with the same `--run-id` resumes, skipping arms already `ok` or `failed`.

## Budget

$1.99/hr from create to destroy, on credit ($62.68, expiring 2026-10-15). Estimated 4–5
droplet-hours: a ~35 GB image pull, ~45 GB of checkpoints, eight arms of nine cells. Hard cap 8
hours. The droplet is destroyed in the console when the sweep finishes.

## Results

*(empty — filled in after the runs)*
