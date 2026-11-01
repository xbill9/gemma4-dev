# 2026-11-01 — why E4B keeps more of its decode speed at depth than E2B, GTX 1650 Ti

Same card, torch 2.13.0+cu130, both models at `max_ctx` 40960. `profile_depth.py`: the seeded records prompt read once
(61 and 32,761 tokens), 63 tokens decoded, then `torch.profiler` over 50 graph replays, two passes each, the whole run
done twice per model (`profile-2b.jsonl`, `profile-4b.jsonl`, `ANALYSIS.txt`). `run.sh` exited 1 on a `tail` at its end,
after all data was written.

| | E2B | E4B |
| --- | ---: | ---: |
| Decode tok/s, short | 87.73 | 44.93 |
| Decode tok/s, ~32.8k held | 52.11 | 34.91 |
| Speed kept | 59% | 78% |
| Replay step, short / 32.8k | 11.37 / 19.08 ms | 22.10 / 28.50 ms |
| Int4 matrix kernel, either depth | 9.33 ms | 18.8 ms |
| Decode attention at 32.8k | 8.75 ms | 7.87 ms |

**Almost the whole rise is the seven full-attention layers**, in both models; sliding layers and the int4 kernel do not
move. The rise is 7.71 ms on E2B and 6.40 ms on E4B.

**Both models have seven full layers of 8 query heads at 512 wide. They read K/V differently:**

| | E2B | E4B |
| --- | ---: | ---: |
| KV heads in a full layer | 1 | 2 |
| Attention calls per full layer | 1, 8 query heads | 2, 4 query heads each |
| Time per call at 32.8k | 1.11 ms | 0.45 ms |
| Time per layer | 1.11 ms | 0.89 ms |
| Read rate per call, 67 MB K+V | 61 GB/s | 153 GB/s |

E4B's calls run at the 156 GB/s measured copy rate of this card, so they are limited by memory; E2B's run at 0.39 of it,
so they are limited by compute. E4B reads twice the K/V per layer and still spends less time, because each read carries
half the query heads.

E4B also starts from a step twice as long, so the same milliseconds are a smaller share. Both effects lift its speed
kept. Attention at depth is 46% of E2B's step and 28% of E4B's.

**A kernel test isolates the split.** `attn_split.py`, no model, random K/V, one layer at 32,824 keys: 8 heads in one call
0.97–1.11 ms, the same 8 heads split 4+4 over the same K/V 0.92 ms with identical output, 4+4 over two K/V pairs 0.92 ms.
Splitting E2B's calls in the
real engine was not run; at 0.05–0.19 ms a layer it would save at most about 1.3 ms of a 19 ms step.

llama.cpp was not profiled, so this explains the engine's side of the ratio, not llama.cpp's.
