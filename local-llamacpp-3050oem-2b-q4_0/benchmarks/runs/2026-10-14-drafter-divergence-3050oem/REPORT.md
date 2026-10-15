# 2026-10-14 — why greedy text with a drafter matches no drafter only sometimes, RTX 3050 OEM

**Question:** `2026-10-14-drafter-gaps-3050oem` found E4B's greedy replies with the drafter
identical to no drafter at `-c 8192` (18 of 18) and mostly different at `-c 131072` (4 of 14).
Is the cause the server context, the prompt cache (on in the long-context run), or the prompt
length (the 8k runs used ~25-token prompts; the long-context run put up to 120k tokens of text
ahead of the question)?

**Answer: prompt length.** The server context changes nothing, and the prompt cache is a separate
source of different text that applies with or without a drafter.

- **Server context: no effect.** Every reply at `-c 16384` is identical to the same reply at
  `-c 131072`, 84 of 84 on each model.
- **Prompt length decides whether E4B's drafted text matches.** With the cache off, E4B at 2
  drafted tokens matches no drafter on 2 of 2 replies with no text ahead, 1 of 3 at 1,024 tokens,
  and 0 of 3 at 2,048, 4,096 and 8,192. Same with the cache on, except 1 of 3 at 8,192.
- **E2B at 2 drafted tokens differs at every length**, including no text ahead, as already known
  (2 matches in 56 drafted replies: the same 2,048-token prompt with the cache off, at each context).
- **The prompt cache changes greedy text by itself.** With no drafter, cache on against cache off
  gives different text on 22 of 28 replies on E2B and 18 of 28 on E4B, and every difference is on a
  prompt partly served from the cache (only the new part read). The 4 prompts with no text ahead,
  read in full either way, all matched. So a reply depends on how the prompt was split into reads,
  drafter or not.
- **Each setup repeats exactly.** The repeated no-drafter arm matches the first on every reply,
  at both contexts and both cache settings, on both models.

**What this means for benchmarks:** byte-identical greedy text needs no drafter **and** the
prompt cache off. With a drafter, E4B's text matches no drafter for short prompts only (under
about 1k tokens); the earlier "identical at 2 or 3 drafted tokens" result holds for that case.

## Method

`div.py`, llama.cpp `77c4540ce`, the rigs' serving flags, one llama-server per (context, arm) on
port 8099; `common.py` is the gap run's. Prompts: the first 0 / 1,024 / 2,048 / 4,096 / 8,192
tokens of `/usr/share/doc/NVIDIA_GLX-1.0/README.txt` (SHA-256 in `doc.sha256`) followed by a story
request, a code request, or (with text) a request to summarize it; 14 prompts. Each server answers
all 14 with the cache off, then all 14 with it on, in the same order. Arms: no drafter, exact Q4_0
drafter at 2 tokens, no drafter again; contexts 16,384 and 131,072. Greedy, 256 tokens. `run.sh`
ran E2B 20:30–20:48 and E4B 20:48–21:16 and restored the live E4B server. `summarize.py` writes
`e2b/summary.txt` and `e4b/summary.txt`; `results.jsonl` keeps every reply's text.

## Replies identical to the first no-drafter arm

E4B (`e4b/summary.txt`), of the questions at each length:

| arm, context, cache | 0 | 1,024 | 2,048 | 4,096 | 8,192 |
|---|---:|---:|---:|---:|---:|
| no drafter again, either context, either cache | 2/2 | 3/3 | 3/3 | 3/3 | 3/3 |
| drafter 2 tokens, either context, cache off | 2/2 | 1/3 | 0/3 | 0/3 | 0/3 |
| drafter 2 tokens, either context, cache on | 2/2 | 1/3 | 0/3 | 0/3 | 1/3 |

E2B (`e2b/summary.txt`): no drafter again 3/3 everywhere (2/2 at 0); drafter at 2 tokens 0 at every
length except 1 of 3 at 2,048 with the cache off.

Where drafted replies part from no drafter (first differing word): E4B words 1–103, E2B 1–126.

## Cache on against cache off, no drafter

| prompt | E2B | E4B |
|---|---|---|
| no text (24 and 29 new tokens either way) | 2/2 same | 2/2 same |
| first prompt at each length, prefix reused from the previous length | 1 of 4 same | 1 of 4 same |
| second and third prompts, text reused, only the question read (14–24 tokens) | 0 of 8 same | 2 of 8 same |

Counted at `-c 16384`; `-c 131072` is identical.
