# 2026-10-13 — `per_layer_token_embd` in VRAM vs in the mmap, RTX 3050 OEM

**Question:** the 8 GiB card has ~6.3 GiB unused at the serving settings. Does putting the
per-layer embedding table (1.32 GB, 51% of the file) on the card make anything faster?

**Answer: prefill, slightly; decode, no.** Card against host, mean of two passes each:

| test | host | card | card/host |
| --- | ---: | ---: | ---: |
| pp512 | 3845.18 | 3982.67 | **1.036** |
| pp2048 | 3759.36 | 3848.42 | **1.024** |
| tg128 | 115.91 | 115.98 | 1.001 |

Cost: **1260 MiB** of VRAM (`nvidia-smi` 1510 → 2770 MiB of 8192). Adopted:
`OVERRIDE_TENSORS=per_layer_token_embd\.weight=CUDA0` in `tpu.env`, passed as `-ot` by
`make serve` and the MCP server.

## Setup

`ple_ab.sh`, copied from `local-llamacpp-1650ti-2b-q4_0`'s `2026-09-29-ple-offload-1650ti`. v2
exact GGUF (`419db9a6…`), llama.cpp `77c4540ce`, `-ngl 99 -fa 1 -t 6`. The only difference
between the arms is `-ot 'per_layer_token_embd\.weight=CUDA0'`. llama-bench, r=5, ABBA order
(host, card, card, host), GPU cooled to ≤40 °C before each pass. Raw output in `ple-ab.md`,
allocation logs in `alloc-{host,card}.log`, ratios in `summary.txt`.

## It really moved

| buffer | host (default) | card (`-ot`) |
|---|---:|---:|
| CUDA0 model | 1223.91 MiB | **2483.91 MiB** (+1260) |
| CPU_Mapped model | 1260.00 + 216.00 MiB | 216.00 MiB |
| CUDA0 KV | 48 + 12 MiB | 48 + 12 MiB |
| CUDA0 compute | 128.02 MiB | 128.02 MiB |
| CUDA_Host compute | 38.53 MiB | 21.03 MiB |
| `nvidia-smi` process | 1510 MiB | **2770 MiB** |

The 216 MiB left on the host is `token_embd`'s input-lookup copy, so mmap stays on.

## The gain is consistent

Both card passes beat both host passes at both prompt lengths (pp512: 3986.60 and 3978.74
against 3856.99 and 3833.36; pp2048: 3853.13 and 3843.71 against 3767.65 and 3751.08). The
pp512 cells carry about ±110 t/s of within-cell spread each; pp2048 about ±15, and its 2.4% gap
is far outside that. Decode sits within 0.2 tok/s in all four passes.

On a 512-token prompt the gain is about 4.5 ms of prompt processing (133.2 → 128.6 ms at the
measured rates). The 1650 Ti measured no gain from the same move (within 0.7%); this card
prefills 11x faster, so the host-side row gather the move removes is a larger share of the work.

## Bigger prefill micro-batch: no gain

`-ub 2048 -b 2048` against the default: pp2048 0.997x with the table on the host, 1.009x with it
on the card; decode unchanged. Not adopted.
