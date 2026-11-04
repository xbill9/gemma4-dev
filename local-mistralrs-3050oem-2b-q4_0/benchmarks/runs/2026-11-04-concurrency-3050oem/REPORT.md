# 2026-11-04 — mistral.rs, 1 to 64 requests at once on the 3050

mistral.rs v0.9.5 serving the exact QAT Q4_0 E2B GGUF on the RTX 3050 OEM, through the same `sweep.py` and arguments as the llama.cpp and vLLM sweeps of 2026-11-03: 512 in / 128 out and 128 in / 512 out, 1 to 64 requests at once, 5 repeats, unique prompts, thinking on, decode timed from the stream, no drafter. Server from `tpu.env`: at most 4 requests running (more do not fit beside an 8,192-token cache on this card), 1,024 tokens per scheduler step; later requests wait in its queue.

Total output tok/s (`COMPARE.txt`, from `compare.py`):

| requests at once | 512 / 128: mistral.rs | llama.cpp | vLLM | 128 / 512: mistral.rs | llama.cpp | vLLM |
|---|---|---|---|---|---|---|
| 1 | 107.9 | 95.3 | 105.4 | 111.8 | 107.1 | 120.0 |
| 2 | 200.1 | 160.9 | 183.3 | 221.4 | 193.2 | 231.5 |
| 4 | 277.7 | 201.2 | 294.0 | 309.4 | 262.8 | 435.0 |
| 8 | 270.2 | 228.1 | 422.2 | 308.7 | 312.6 | 777.2 |
| 16 | 264.7 | 321.5 | 531.0 | 308.1 | 508.9 | 1,235.9 |
| 32 | 260.8 | 354.7 | 587.3 | 307.8 | 592.7 | 1,544.3 |
| 64 | 258.5 | 367.6 | 603.1 | 308.0 | 633.3 | 1,651.2 |

## Findings

- **Up to 4 requests mistral.rs is ahead of llama.cpp** (1.13–1.38x on 512 / 128, 1.04–1.18x on 128 / 512) and its first token comes sooner (85 against 202 ms for one request, 512 / 128).
- **From 4 requests it stops gaining**: 258–278 and 307–309 tok/s whatever the load, because 4 is the most it runs at once on this card. By 16 requests llama.cpp leads (0.82x and 0.61x), and at 64 mistral.rs gives 0.70x and 0.49x llama.cpp and 0.43x and 0.19x vLLM. Waiting requests show it as time to first token: 14.5 s and 42.9 s at 64.
- **Time per token stays flat** (8.9 ms at 1–2 requests, 12.6–13.3 ms at 4 and above) where llama.cpp's grows to 93–111 ms at 64: each running request is fast, there are only 4 of them.
- Peak GPU memory 4,755 MiB (`gpu.csv`). Repeat spread under 2% in every cell.

Memory sets the limit: after load mistral.rs holds 4.1 GB (it keeps the per-layer table and the image and audio encoders on the card), and its planner reserves working memory per running request. On a card with more memory the same build would run more requests at once (untested).

Files: `run.sh`, `compare.py`, `COMPARE.txt`, `in512-out128/`, `in128-out512/` (sweep.py output), `server.log`, `gpu.csv`, `vram.txt`, `command.txt`, `versions.txt`, `run.log`.
