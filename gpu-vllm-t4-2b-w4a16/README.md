# gpu-vllm-t4-2b-w4a16

vLLM on one NVIDIA Tesla T4 already attached to a Compute Engine VM, serving
Gemma 4 E2B as **4-bit weights end to end**:
[`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4).

Forked 2026-09-29 from [`gpu-vllm-t4-2b`](../gpu-vllm-t4-2b/), which serves the bf16
reference build on the same GPU. Slot 5 is the only difference between the two.
Read [`CLAUDE.md`](CLAUDE.md) before changing anything.

## Measured

vLLM 0.29.0, `--dtype float16 --gpu-memory-utilization 0.90 --max-model-len 16384
--max-num-seqs 8`, compile cache warm, 2026-09-29:

| Build | Model loading | KV cache | Decode, c=1 |
|---|---:|---:|---:|
| Google `-qat-w4a16-ct` | 8.02 GiB | — | — |
| xbill9 multimodal repack, `--language-model-only` | 6.37 GiB | 707,617 | — |
| xbill9 `-text` (bf16 embeddings) | 6.33 GiB | 711,539 | 81.6 tok/s |
| **xbill9 `-text-emb4` (served)** | **2.86 GiB** | **1,099,362** | **109.7 tok/s** |

Single-stream only; there is no concurrency sweep for this rig yet. Evidence in
[`evidence/`](evidence/).

## How the checkpoint was built

Google's QAT export stores bf16 values that already sit on a 4-bit grid, in groups
of 32. The embedding tables are on it too. Each step below recovers that grid
rather than quantizing, and aborts on any off-grid group.

```bash
PY="env PYTHONUSERBASE=/opt1/pyuser /usr/bin/python3.13"
$PY repack/repack_q4_0.py repack SRC OUT                 # linears -> W4A16
$PY repack/text_only.py OUT OUT-text                      # drop vision/audio towers
$PY repack/embed_int4.py OUT-text OUT-text-emb4 --embed-tokens   # PLE, embed_tokens, lm_head -> int4
```

Write outputs under `/opt1`: unchanged shards are hard-linked, and the root disk
has a few GB free.

## Serving

```bash
VLLM_T4_RIG=~/gemma4-dev/gpu-vllm-t4-2b-w4a16 ~/bin/vllm-t4 start
```

or the MCP server's `start_vllm_server`. Not `make serve`: its short-lived
interpreter kills vLLM about a second after launch (see `~/bin/vllm-t4`).
**One rig serves at a time**: this rig and
`gpu-vllm-t4-2b` share the GPU and port 8000, and each tracks only its own process.
After changing `MODEL_NAME`, restart once: the first start compiles from scratch
and undersizes the KV cache.
