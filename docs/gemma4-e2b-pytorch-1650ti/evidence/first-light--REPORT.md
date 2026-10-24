# 2026-10-23 — first light: Gemma 4 E2B QAT W4A16 in plain PyTorch on a GTX 1650 Ti

Host `yogagpu`: GTX 1650 Ti with Max-Q Design (TU117, sm_75, 3,732 MiB visible to PyTorch), driver 615.71.10.
torch 2.13.0+cu130, transformers 5.16.1, compressed-tensors 0.17.0, accelerate 1.14.0, Triton 3.7.1, Python 3.14.7.
Checkpoint `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text` at `315aec06`, 6.11 GiB on disk.

One prompt ("Explain in two sentences why the sky is blue."), 19 prompt tokens, greedy, up to 64 new tokens;
every run stops at EOS after 57. tok/s is new tokens over wall time including the prompt read.

#### Result

| Path | tok/s | Peak CUDA MiB | Tokens vs bf16 | File |
| --- | ---: | ---: | --- | --- |
| compressed-tensors unpack per forward, bf16 | 0.58 | 1,948 | identical | `run-bf16-stock-unpack.jsonl` |
| Own int4 unpack + matmul, bf16, eager | 2.31 | 1,930 | reference | `run-bf16-eager.jsonl` |
| Same, fp16, `torch.compile` | 9.38 | 1,822 | identical 57/57 | `run-fp16-fused.jsonl` |
| Triton int4 GEMV for decode, fp16 | 11.81 | 1,821 | identical 57/57 | `run-fp16-triton.jsonl` |
| Triton GEMV + captured CUDA graph, static KV cache | **18.14** | **1,842** | identical 57/57 | `run-fp16-triton-cudagraph.jsonl` |

The reply: "The sky appears blue because of a phenomenon called Rayleigh scattering, where the Earth's atmosphere
scatters shorter, blue wavelengths of sunlight more effectively than longer, red wavelengths. This scattered blue
light is what reaches our eyes from all directions, making the sky look blue during the day."

For scale, `local-llamacpp-1650ti-2b-q4_0` on the same card records 67.6–73.8 tok/s single-stream decode on the
QAT Q4_0 values. Different engine, different measurement; read the ratio as indicative.

#### Memory

- Unpacked to bf16 on the CPU (`sizes_cpu.jsonl`): decoder layers 3.579 GiB, `embed_tokens` 0.75 GiB,
  `embed_tokens_per_layer` 4.375 GiB, 8.731 GiB in all. The layers alone do not fit the card.
- Packed on the GPU: 1.734 GiB (int4 layers plus fp16/bf16 `embed_tokens`, which is tied to `lm_head`).
  `embed_tokens_per_layer` stays in CPU RAM; each step looks up one row set there and copies it to the GPU.

#### The weights are Google's QAT grid

`qat_check.py` / `qat_check2.py` dequantize all 276 packed linears and compare them with
`google/gemma-4-E2B-it-qat-q4_0-unquantized`: **0 of 1,876,819,968 int4 codes disagree**. 9.97% of the
dequantized bf16 values differ from Google's by at most 0.0039 (6.2% of one group step), consistent with the
repack storing each group's step in bf16.

#### fp16 on Turing

Inductor does not compile bf16 for sm_75, so the fast paths run fp16. Largest |activation| leaving any decoder
layer: 112.1 (fp16 max 65,504), no non-finite values, on this one prompt.

#### What each step fixed

- **transformers 5.16.1 has no packed int4 compute path.** It applies the config with `run_compressed=False`
  (`quantizer_compressed_tensors.py:114`), and compressed-tensors unpacks every layer to bf16 at the first
  forward, which runs out of memory on this card. transformers 5.20.2 is the same at line 118; its
  `use_optimized_inference` covers W8A8 FP8 on SM89+ only. compressed-tensors 0.19.0 removes `CompressedLinear`.
  `smoke_cuda.py` removes the decompress hook and unpacks each linear inside its own forward.
- **accelerate's CPU offload copies the whole 4.38 GiB per-layer table to the GPU every forward.** The scripts load
  it into CPU RAM themselves and look rows up there.
- **The compressed-tensors unpack costs 14.7 ms on a 12288x1536 linear, the matmul 0.25 ms** (`micro.py`).
  A vectorized unpack is bit-exact against it (`micro2.py`); compiled in fp16 it takes 0.67 ms.
- **The compiled path still writes an fp16 copy of every weight and reads it back.** Profile over 16 decode steps:
  `aten::mm` 1,063 ms, the Triton unpack 562 ms, ~7,700 kernel launches per token. `w4gemv.py` reads packed int4
  directly: 2.65–2.83x the compiled path per linear on five of six E2B shapes and 1.84x on the smallest
  (2048x1536), relative error 2–4e-4, 45–49 GB/s read on the five larger shapes (`micro3.jsonl`).
- **A captured CUDA graph removes the launch cost**, with one fix. `StaticSlidingWindowLayer.get_seq_length` returns
  a Python int, and the sliding-window mask takes its query offset from it (`masking_utils.py:829`), so the graph
  froze the offset at capture and output broke from the fifth replayed token (`graphdiff`). Reporting the layer's
  on-device `cumulative_length` instead makes graph logits equal the plain step's at every position (max diff 0.0).
  Valid while the cache (`max_cache_len=256`) is shorter than the 512-token window, which the script asserts.

#### Not measured

Prefill speed on its own, prompts other than this one, contexts beyond 256 tokens, accuracy suites, more than one
request at a time, and GPU busy time with the graph in place. `lm_head` (tied to `embed_tokens`) is still fp16
and reads 0.75 GiB per token by arithmetic.

#### Reproduce

From this directory, with the system `python3`:

```
python3 smoke_cuda.py                          # compressed-tensors unpack, bf16
python3 fused_cuda.py bf16 eager               # writes the reference tokens the others compare against
python3 fused_cuda.py fp16 fused [profile]
python3 fused_cuda.py fp16 triton [profile]
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 fused_cuda.py fp16 triton cudagraph
python3 fused_cuda.py fp16 triton graphdiff    # graph vs plain logits per position
python3 micro.py; python3 micro2.py; python3 micro3.py
python3 sizes_cpu.py; python3 qat_check.py; python3 qat_check2.py
```

`fused_cuda.py ... cudagraph` and `graphdiff` read `run-bf16-eager.jsonl`, so run the bf16 reference first.
