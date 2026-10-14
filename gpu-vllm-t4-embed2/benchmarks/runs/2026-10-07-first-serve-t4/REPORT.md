# 2026-10-07 — first serve of EmbeddingGemma 2 on the attached T4

**What was measured:** `google/embeddinggemma-2` (snapshot `914f7f89`) served by vLLM's pooling
runner on one Tesla T4 (SM 7.5, 15,360 MiB) attached to this `n1-standard-2` VM, at float32.
One day after the model's release, and on the vLLM nightly, because support (`02b83919aa`,
PR #60254) is in no release.

**Stack** (`evidence/packages.txt`, `evidence/requirements-lock.txt`): vLLM
`0.31.1rc1.dev63+gd169c5a90`, torch `2.13.0+cu130`, transformers `5.19.0`, triton `3.7.1`,
sentence-transformers `6.1.0`, Python 3.13.5, all in `PYTHONUSERBASE=/opt1/pyuser-embed2`.
The encoder-attention clamp (`evidence/encoder-clamp.diff`) applied.

## Headline

| | |
| :--- | :--- |
| Parity vs sentence-transformers (fp32, CPU) | **1−cos ≤ 7.0e-9** on 5 texts, 16–1,818 tokens; max element diff 1.7e-5 |
| Weights on device | 1.03 GiB text only · 1.63 GiB text + vision |
| Device memory in use while serving | 1,979 MiB (no KV cache: encoder-only) |
| Time to healthy | 96 s text only · 177 s / 202 s text + vision |
| Throughput, ~500-token inputs | 3,909 input tok/s at c=1 (120 ms p50); 4,234 at c=4; 3,594 at c=16 |
| Throughput, ~2,000-token inputs | 2,191 input tok/s at c=1 (940 ms p50); 2,212 at c=4; 2,067 at c=16 |
| Matryoshka | `dimensions` 256/128 equal the renormalised prefix to 4e-8; 300 rejected (400) |
| Vision | serves; each text query ranks its own image first (0.761 vs 0.622, 0.780 vs 0.637) |

**One stream saturates the GPU.** Concurrency adds queueing latency and no throughput:
fp32 on Turing runs on CUDA cores (8.1 TFLOPS peak, no fp32 tensor-core path), and
2 × 271M × 3,909 tok/s ≈ 2.1 TFLOP/s of linear work alone. Batch on the client side,
not with concurrent requests.

## What it took, in order (`evidence/failed-routes.txt`)

1. **Stock kernel, `--dtype float32`** — dies during profiling:
   `OutOfResources: shared memory, Required: 69632, Hardware limit: 65536` in
   `triton_prefill_attention.context_attention_fwd`. That is the ENCODER kernel; the
   generative T4 rigs' clamp is on the unified decoder kernel and never runs here.
2. **Kernel probe** (`scripts/tilefit.py`, `scripts/tilefit2.py`): at head_dim 256, BLOCK 16
   fits (max |err| 5.7e-6 vs torch). At head_dim 512, no fp32 tile fits — 16 is `tl.dot`'s
   floor and still needs 66,560 B. q/k/v in fp16 at BLOCK 16, fp32 accumulate: fits,
   relative error 4.6e-4 on random inputs.
3. **`--attention-backend FLEX_ATTENTION`** — fails twice for reasons unrelated to Turing,
   then for the same smem reason: its `create_block_mask` is compiled with
   `mode="reduce-overhead"` and the sliding and full masks overwrite each other's CUDA-graph
   output (`--enforce-eager` does not avoid it); with that removed and `num_stages=1`, its
   fp32 template needs exactly 65,536 B. Abandoned; both edits reverted.
4. **The clamp** (`patch_triton_encoder_turing.py`): BLOCK ≤ 16 at head_dim ≥ 256, and the
   head_dim-512 calls in fp16 with copy-back. Default TRITON_ATTN backend, CUDA graphs
   left as vLLM chose (it skips encoder capture). This is what the parity number measures.

Also hit, not a model problem: pip's resolver backtracked through every `filelock`/`httpx2`
release for 20+ minutes; `uv pip compile` resolved the same request in seconds, and the
nightly index had already moved past the version first pinned (it lists only its newest
wheel), so the lock pins the per-commit wheel URL.

## What this run does not establish

- **Image parity.** Images serve and rank sensibly; they were not compared against
  sentence-transformers, whose reference does not fit in host RAM beside the server.
- **Audio and video.** Disabled (`audio: 0, video: 0`); untried.
- **Retrieval quality.** No MTEB or similar; the parity number says the T4 computes what the
  reference computes, not how good that is.
- **Repeat variance.** One pass per throughput cell, 32–64 requests each.

## Files

- `evidence/parity-vs-sentence-transformers.txt` — from `scripts/parity.py`, run in two
  phases (`PHASE=vllm`, then the server stopped and `PHASE=ref`) because both in RAM at once
  was OOM-killed.
- `evidence/matryoshka-and-vision.txt` — `scripts/mm.py`.
- `evidence/throughput.jsonl` — `scripts/bench.py`, against the text + vision server.
- `evidence/serve-*.log` — engine logs, request lines removed.
