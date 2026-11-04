# 2026-11-04 — SGLang, 1 to 64 requests at once on the 3050

SGLang 0.5.23 (torch 2.14.1+cu130, flashinfer 0.7.2.post1) on the RTX 3050 OEM, serving
`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` @ `db715f83`, the checkpoint and revision vLLM served the
day before. Same `sweep.py` and arguments as the vLLM, llama.cpp and mistral.rs sweeps: 512 in / 128 out and
128 in / 512 out, 1 to 64 requests at once, 5 repeats, unique prompts, thinking on, decode timed from the
stream, no drafter. Server from `tpu.env` (`command.txt`): up to 64 running requests, radix cache off, decode
CUDA graphs for 1 to 64 requests, prompt reading without graphs, the rig's two patches applied.
`run.sh` drives it; `compare.py` prints `COMPARE.txt`.

#### Result

Aggregate output tok/s, median of 5 repeats:

| requests | SGLang 512/128 | vLLM | SGLang 128/512 | vLLM |
|---|---|---|---|---|
| 1 | 103.4 | 105.4 | 119.6 | 120.0 |
| 4 | 279.7 | 294.0 | 434.7 | 435.0 |
| 16 | 487.1 | 531.0 | 1,269.1 | 1,235.9 |
| 64 | 544.0 | 603.1 | 1,711.3 | 1,651.2 |

**On long replies SGLang is level with vLLM at every level** (1.00–1.04x, peak 1,711 against 1,651
tok/s at 64 requests). **On long prompts it gives 0.90–0.98x vLLM**, falling as requests rise. Its time
per output token is lower than vLLM's at every level of both shapes (33.2 against 59.6 ms at 64 requests on
512/128), but its time to first token is higher there (10.8 s against 6.7 s), which reads as prompt reading
costing SGLang more than vLLM on this card (inferred; prompt reading runs without CUDA graphs here).

Against llama.cpp (no drafter, 64 slots) SGLang gives 1.08–1.72x on 512/128 and 1.12–2.70x on 128/512.
Against mistral.rs it is behind at 1–2 requests on 512/128 (0.96x, 0.89x) and ahead from 4 on, up to
2.10x on 512/128 and 5.56x on 128/512 at 64 requests. Output grows 14.3x from 1 to 64 requests on 128/512
and 5.3x on 512/128. Repeats agree within 0.9%.

Peak memory 6,966 MiB of 8,192 (`MEM_FRACTION_STATIC=0.80`: 3.25 GB of weights, 120,862 full-attention and
45,121 sliding-window cache tokens), against vLLM's 7,082.

The vLLM, llama.cpp and mistral.rs sweeps ran on 2026-11-03 and earlier on 2026-11-04; same card, harness and
arguments.

#### Files

`in512-out128/`, `in128-out512/` (`concurrency.json` and per-request records), `server.log`, `gpu.csv`
(once a second), `vram.txt`, `versions.txt`, `command.txt`, `run.log`.
