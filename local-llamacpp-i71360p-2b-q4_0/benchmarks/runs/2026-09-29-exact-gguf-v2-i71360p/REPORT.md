# 2026-09-29 — exact Q4_0 GGUF v2: `per_layer_model_proj` at Q4_0, i7-1360P

Follows `../2026-09-29-exact-gguf-i71360p`. Same host, llama.cpp `fc07d781e` native CPU build,
`-t 8`.

## Finding

`per_layer_model_proj.weight` `[1536, 8960]` was the last weight matrix outside Q4_0: F16 in
Google's GGUF, copied unchanged into the first rebuild. It is QAT data like the rest — 0 of 430,080
groups of 32 off the 4-bit grid in `-qat-q4_0-unquantized`, 96.74% of values bit-identical after
rebuild (`build_report.json`). Stored as Q4_0:

| | first rebuild | v2 |
|---|---|---|
| File | 2,640,154,592 B | **2,620,370,912 B** (19,783,680 B / 0.75% smaller) |
| SHA-256 | `25f21f14…` | `419db9a6bf3bc15d770c85ccf9216827aa88c64fe8f3d72fbe5674f48efe2dc8` |
| Mean KL divergence vs bf16 | 0.001761 ± 0.000077 | 0.001683 ± 0.000066 |
| 99th percentile KL divergence | 0.016158 | 0.014232 |
| Same top token | 97.892 ± 0.225 % | 98.137 ± 0.212 % |
| PPL / PPL(bf16) | 1.020957 ± 0.005062 | 1.020320 ± 0.005050 |

The two files agree within the error bars. Everything in v2 is Q4_0 except 1.1 MB of F32 norms and
scale vectors, and `per_layer_token_embd` is 51% of the file.

**Speed is unchanged** (`bench-v1-v2-abba.md`, pp512/tg128, r=5, both orders): generation v2/v1 is
0.997 with v1 first and 1.014 with v2 first. Prefill again follows run order (0.773, 1.372) and is not
claimed. First light through the rig's MCP tools served the v2 file and answered correctly
(`first-light.md`).

## Method

- `gguf_exact.py` is the first run's builder with one line added: `per_layer_model_proj.weight` in
  the tensor map. `--bf16` rebuilt the reference with the same map, so its `per_layer_model_proj` is
  the source's bf16 bytes rather than Google's F16 copy.
- All KL divergence figures here come from this rig's native `llama-perplexity` against that
  reference (wikitext-2 test, 16 × 512 tokens). They are not differenced against the first run's
  Docker figures, which used the earlier reference; the first rebuild scores 0.001761 here against
  0.001753 there.
- Throughput: `llama-bench`, 60 s pause before and 90 s between passes, `x86_pkg_temp` logged.

## Published

`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf` has served v2 since Hub revision `74b2c92` (2026-09-29); v1 (`25f21f14…`) is at revision `e5d65c8`. The upload is byte-identical to this run's file
(SHA-256 checked on the Hub), rebuilt independently on an i7-10750H in
`local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-09-29-exact-gguf-v2-1650ti/`.
