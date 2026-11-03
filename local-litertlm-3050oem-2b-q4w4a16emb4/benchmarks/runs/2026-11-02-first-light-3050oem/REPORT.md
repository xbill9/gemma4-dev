# 2026-11-02 — first light: the exact QAT Q4_0 grid in LiteRT-LM, RTX 3050 OEM

**Question:** can the xbill9 QAT Q4_0 repack run under LiteRT-LM on the 3050, with its values unchanged,
and how fast?

**Answer:** yes. Every int4 code and scale in the built `.litertlm` equals the repack's, the whole decode and
prefill graphs run on the GPU, and it generates at 81.48 tok/s and reads prompts at 5,118 tok/s, against
85.81 and 4,963 for Google's `.litertlm` on the same card (0.95x and 1.03x). It has no drafter.

## How it is built (`build.sh`)

1. **Convert** `google/gemma-4-E2B-it-qat-q4_0-unquantized` (revision `6befbaca`) with litert-torch 0.10.0:
   `--quantization_recipe=dynamic_wi4b32_afp32 --externalize_embedder=True --apply_gpu_composites=True`.
   Int4 in blocks of 32 with one fp16 scale per block; q/k/v and gate/up fused.
2. **Write the exact codes** with `repack_litertlm.py` from `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`
   (revision `db715f83`): 214 tensors, 5,030,936,576 codes, every repack weight used, each tensor re-read
   and checked equal (`repack.log`, `repack_report.json`).
3. **Bundle** with `litert-lm pack`, taking the tokenizer and LLM metadata from Google's `.litertlm`.

Result: 2,840,220,768 bytes, SHA-256 `c880dac6…88da8bb`.

## What the converter does to the grid, and what the edit restores

The converter quantizes each block itself: codes −7…7 and scale = max|w| / 7. On the repack's grid the codes
run −8…7 and the scale is the trained step. Against the repack, the converter's codes differed in 6.6% to
18.0% of each tensor (median 16.0%); 2.49% of all repack codes are −8, which the converter never writes.
All 157,216,768 repack scales are exact in fp16: 58,650,624 bf16 (linears) and 98,566,144 already fp16
(vocabulary tables), each compared after a bf16 → fp16 → float round trip.

## Speed (`litert-lm benchmark --backend gpu -p 1024 -d 256`, drafter off)

| build | runs | prompt reading | generation |
| --- | ---: | ---: | ---: |
| Google `gemma-4-E2B-it.litertlm` (int2/int4/int8, per row) | 3 | 4,963 tok/s | 85.81 tok/s |
| converter, blocks of 32, no GPU composites | 1 | 3,789 | 44.44 |
| converter, one scale per row, no GPU composites | 3 | 3,954 | 48.95 |
| converter, blocks of 32, GPU composites | 3 | 5,514 | 80.14 |
| **exact repack** (the last row with the repack's codes) | 3 | **5,118** | **81.48** |

The Google row is from the int248 rig's first-light run. Without GPU composites the converter's graph decodes at
about half speed whether scales are per row or per block, so the block size is not what costs speed; the
composites are. In all converted builds every prefill and decode node runs on the WebGPU delegate; the two
embedding lookups run on the CPU (XNNPACK), as in Google's file (`bench-*.log`).

## Replies

One greedy reply ("List the planets of the solar system with one fact about each.") from the exact build was
coherent and correct, so the GPU kernels handle the −8 codes. Accuracy is scored in
`../2026-11-02-accuracy-3050oem`.

## Build notes

- litert-torch is installed with `pip --target` into its own directory with an explicit `--no-deps` list
  (a plain `--target` install pulls a second torch with CUDA wheels). Its pin `xdsl==0.28.0` needs
  `typing_extensions<4.13`, which lives only in that directory. A stand-in `tensorflow.lite.python.schema_py_generated`
  replaces TensorFlow, which `litert_torch`'s `bonsai_flux2` module imports at load time.
- Gemma 4 export needs `--externalize_embedder=True`.
- litert-torch 0.10.0's bundling step fails against litert-lm-builder 0.19.0 (`no attribute 'TfLiteModelType'`)
  after the quantized graphs are written; `litert-lm pack` bundles them instead.
- One export takes 13–15 minutes on the i7-5820K and peaks near 45 GB of the host's 46 GB.
