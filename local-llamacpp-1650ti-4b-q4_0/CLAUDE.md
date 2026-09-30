# CLAUDE.md — local-llamacpp-1650ti-4b-q4_0

The **E4B parallel of `local-llamacpp-1650ti-2b-q4_0`**, forked from it on 2026-09-30. Same card, same
llama.cpp binary (`f95b0d9`), same flags; **slot 4 is the only slot that differs.** Read the E2B rig's
`CLAUDE.md` for everything this file does not repeat — building llama.cpp on this host (never a bare
`-j`), the `local` platform, the arm attestation, why KV stays f16 and `FORCE_MMQ` stays off, the two
empty-reply paths, and why prefill does not batch. Those are properties of the card and the engine; they
are **measured there, not here**, and `tpu.env` labels each carried-over number `MEASURED ON E2B`.

## What it serves

`gemma-4-E4B-it-q4_0-exact.gguf` (SHA-256 `5462cc10…`, 4.22 GB): Google's E4B QAT GGUF metadata byte for
byte, with every weight matrix, both embedding tables and `per_layer_model_proj` rebuilt as Q4_0 from
`google/gemma-4-E4B-it-qat-q4_0-unquantized` on the trained grid step. Built by the E2B rig's v2
`gguf_exact.py`, which needed no change: E4B has the same tensor set. Published 2026-09-30 as
`MODEL_NAME`, public. **KLD vs bf16 (CPU): 0.00103 against Google's 0.0353**, top token 98.4% against 90.6%. Evidence: `benchmarks/runs/2026-09-30-exact-gguf-e4b-1650ti/`.

**E4B is not a 4B model.** 4.5B effective, 8.0B total (`@MODELS.md`). llama-bench reports 7.46 B.

## It fits, with ~820 MiB to spare — MEASURED 2026-09-30

| | Google's E4B GGUF | exact rebuild | E2B exact v2, for scale |
|---|---:|---:|---:|
| CUDA0 model buffer | 2696 MiB | **2493 MiB** | 1224 MiB |
| KV at 8192 (full + sliding) | 128 + 40 | 128 + 40 | 48 + 12 |
| compute buffer | 110 | 110 | 123 |
| `nvidia-smi` total | 3052 MiB | **2848 MiB** | 1480 MiB |

3671 MiB is free before load. The model buffer was predicted to the MiB from the tensor table, and the
KV to the MiB from `@MODELS.md`'s E4B geometry, before anything was loaded.

**It fits only because `per_layer_token_embd` is lazy** — `[10752, 262144]`, 1.585 GB, 38% of the file,
served by `GET_ROWS` out of the mmap exactly as on E2B. Resident, E4B would need ~4.4 GiB and would not
load. So the E2B rig's rules bind harder here: **never `--no-mmap`**, and do not lower `-ngl` "to be safe".

**The margin is a third of E2B's.** ~820 MiB free against ~2.2 GiB. Only the full-attention KV grows with
context (16 KiB/token here, 3 layers × 2 KiB on E2B), so the arithmetic says ~50K more tokens of context
would fit — UNMEASURED, and compute buffers grow too. Measure before raising `CONTEXT_SIZE` or
`PARALLEL_SLOTS`; the E2B rig's "memory does not bind" conclusions do not transfer.

## It shares the card and the port with the E2B rig

Both serve on `127.0.0.1:8080`, and 1480 + 2848 MiB does not fit in 4096. **Stop one before serving the
other.** `attest.py` identifies the device but not the model: check `/v1/models` or the process's `-m`
before attributing a number to either rig.

## Speed — MEASURED 2026-09-30, llama-bench ABBA

- Exact file vs Google's: **tg128 1.058–1.063x**, pp512 1.01–1.03x, every pass and order. Half of E2B's
  gain (1.11–1.12x); the tied Q6_K head it replaces is a smaller share of what E4B decode streams.
- Absolute: **tg128 ~39.8 tok/s, pp512 ~173 t/s** — about 0.49x E2B's on this card. Rough ratio, different
  sessions.

## Paired sweep with the CPU arm — MEASURED 2026-09-30

`benchmarks/runs/2026-09-30-paired-sweep-1650ti`: the E2B pair's ABBA protocol against
`local-llamacpp-cpu-4b-q4_0` (same file, same commit, `-ngl 0`). 32/32 cells. GPU **4.20x** decode,
**3.84x** prefill, **3.96x** end-to-end — within a few percent of the E2B pair's 4.37 / 3.80 / 3.99.
E4B runs at **~0.49x E2B's decode on the GPU and ~0.51x on the CPU**, matching the 2.0x resident bytes.
The CPU arm drifted −8% between its passes, so a single-order run misstates the ratio by 4–5% here.

## Not known yet

KLD on CUDA, any concurrency run, the quality of thinking-on output (the paired sweep ran thinking on but measured only speed),
and anything above 8192 context. `DEMO.md` did not come across: the rehearsed
demo is the E2B rig's.
