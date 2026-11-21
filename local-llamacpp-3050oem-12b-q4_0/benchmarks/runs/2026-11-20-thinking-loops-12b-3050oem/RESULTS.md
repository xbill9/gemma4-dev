# 12B thinking loops: greedy decoding, not the q4_0 cache

The eight GSM8K problems whose thinking-on reply hit the 32,768-token limit first in wave 3
(`../2026-11-17-thinking-12b-3050oem/gsm8k-repack-none-thinkon.jsonl.partial`, exact Q4_0 repack, no drafter),
resent with one setting changed at a time. llama.cpp 7182ef71e, thinking on, `cache_prompt: false`, one request at a time.

| Setting | Cache | Finished on their own | Right | Hit the limit, repeating |
|---|---|---|---|---|
| wave 3 (greedy, temperature 0) | q4_0, -c 81920 | 0 of 8 | 0 | 8 |
| Google's generation_config.json: temperature 1.0, top_k 64, top_p 0.95, seed 1 | q4_0, -c 81920 | 8 of 8 | 7 | 0 |
| greedy, temperature 0 | f16, -c 40960 | 0 of 8 | 0 | 8 |

Sampled reply lengths: 9146, 9241, 9076, 2626, 11894, 2476, 6069, 1596 tokens.
The f16 server used 7772 MiB. "Repeating" = a 200-character block of the reply occurs again later in it.

Files: `loops.py` (resend), `f16.sh` (stop wave 3, f16 server, resend, restart), `google-q4kv.jsonl`, `greedy-f16kv.jsonl`, `notes.txt`, `server-f16.log`.
