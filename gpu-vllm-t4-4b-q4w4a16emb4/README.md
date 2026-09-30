# gpu-vllm-t4-4b-q4w4a16emb4

vLLM on one NVIDIA Tesla T4 already attached to a Compute Engine VM, serving
Gemma 4 E4B as **4-bit weights end to end**:
[`xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text-emb4`](https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text-emb4)
(4.58 GB: QAT W4A16 linears, int4 per-layer embeddings, `embed_tokens` and
`lm_head`, text only).

Forked 2026-09-30 from [`gpu-vllm-t4-2b-w4a16`](../gpu-vllm-t4-2b-w4a16/), which
serves the E2B build of the same repack on the same GPU with the same flags. Slot 4
is the only difference between the two. Read [`CLAUDE.md`](CLAUDE.md) before
changing anything.

## Measured

Nothing yet. E4B's KV costs 56 KiB/token against E2B's 18
([`MODELS.md`](../MODELS.md)), so none of the parent's numbers carry over.

## Serving

```bash
./vllm-t4 start      # also: stop | status | query | log | swap
```

`vllm-t4` drives the rig it sits in: it turns on the swapfile vLLM needs on this
7.8 GB host, refuses to start unless the Turing clamp is confirmed, launches vLLM
detached with every flag read from `tpu.env` (including the E4B drafter), and waits
for `/health`. Not `make serve`: its short-lived interpreter kills vLLM about a
second after launch.

**One rig serves at a time**: this rig, `gpu-vllm-t4-2b-w4a16` and `gpu-vllm-t4-2b`
share the GPU and port 8000, and each tracks only its own process. `~/bin/vllm-t4`
defaults to the E2B rig. The first start after a model change compiles from scratch
and undersizes the KV cache; restart once before measuring.

## How the checkpoint was built

With the scripts in [`repack/`](repack/), the same ones the parent used for E2B:

```bash
PY="env PYTHONUSERBASE=/opt1/pyuser /usr/bin/python3.13"
$PY repack/repack_q4_0.py repack SRC OUT                 # linears -> W4A16
$PY repack/text_only.py OUT OUT-text                      # drop vision/audio towers
$PY repack/embed_int4.py OUT-text OUT-text-emb4 --embed-tokens   # PLE, embed_tokens, lm_head -> int4
```

It is already published, so serving does not need a rebuild.
