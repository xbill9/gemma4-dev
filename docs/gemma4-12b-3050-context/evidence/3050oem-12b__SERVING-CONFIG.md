# Serving Gemma 4 12B on the RTX 3050 OEM (8 GB): the chosen parameters

Final findings of `benchmarks/runs/2026-10-17-kvcache-12b-3050oem` (with
`2026-10-16-accuracy-12b-3050oem` as the f16 reference), llama.cpp `7182ef71e`, exact Q4_0 12B
(`xbill9/gemma-4-12B-it-qat-q4_0-exact-gguf`) and exact Q4_0 12B drafter
(`xbill9/gemma-4-12B-it-qat-assistant-q4_0-exact-gguf`).

**Workload:** one long agentic conversation at a time on a dedicated card, its context growing
turn by turn. What counts: the most context that fits, generation speed with a lot of that
context held, and unchanged answers.

## The configuration

```
llama-server -m gemma-4-12B-it-q4_0-exact.gguf -ngl 99 -c 81920 -ctk q4_0 -ctv q4_0 -fa 1 \
  -t 6 -tb 12 --parallel 1 --reasoning off --metrics \
  -md gemma-4-12B-it-qat-assistant-q4_0-exact.gguf --spec-type draft-mtp -ngld all --spec-draft-n-max 2
```

`tpu.env`: `CONTEXT_SIZE=81920`, `KV_CACHE_TYPE=q4_0`, `SPEC_DRAFT_N_MAX=2`, `PARALLEL_SLOTS=1`.
7,853 MiB of 8,192 in use; the card must otherwise be idle.

| what you get | value |
| --- | ---: |
| context | 81,920 tokens |
| generation, 0 / 16k / 32k / 49k / 65k tokens held | 55.1 / 42.4 / 37.8 / 33.5 / 30.8 tok/s |
| prompt reading, 8k / 32k / 65k of new text | about 828 / 589 / 406 tok/s |
| GSM8K (1,319) / BFCL simple (400), one at a time | 0.9613 / 0.9500 |
| codes recalled from a long document, 32k / 64k | 58 / 57 of 80 |

## Why this combination

The 12B's weights take 6,390 MiB, so on an 8 GB card **every choice is a trade for memory**
between the context cache and the drafter. Three facts decide it:

1. **The drafter is worth more than any context it costs.** It doubles generation with nothing
   held (55.1 against 27.5 tok/s) and still gives 1.70x at 32k tokens and 1.60x at 65k. Dropping it
   buys context (81,920 → 131,072) at half the speed on every turn.
2. **A smaller cache costs no answers.** Against the f16 cache, paired question by question,
   q8_0 and q4_0 score level on GSM8K and BFCL with and without the drafter (every 95% range
   includes zero), and return the same share of planted codes at every depth they share.
3. **A smaller cache costs little speed, and buys a lot of context.** With the drafter, q4_0
   generates at 0.96x f16 with nothing held and 0.85x at 16k, and holds 3.3x the tokens (81,920
   against 24,576). q8_0 sits between them on context (49,152) and is 1.04x q4_0 at 32k, a gap
   within this card's run-to-run spread.

So the drafter stays, and the cache goes as small as llama.cpp offers: q4_0. The context is the
most that loads beside the drafter (86,016 fails).

## Trade-offs

| setting | context | tok/s at 0 / 16k / 32k / 65k | choose it when |
| --- | ---: | --- | --- |
| f16 cache + drafter | 24,576 | 57.5 / 49.8 / – / – | conversations stay short; best speed under 24k |
| q8_0 cache + drafter | 49,152 | 55.1 / 43.3 / 39.3 / – | conversations stay under 49k |
| **q4_0 cache + drafter** | **81,920** | **55.1 / 42.4 / 37.8 / 30.8** | **long conversations: served** |
| q4_0 cache, no drafter | 131,072 | 27.5 / 24.1 / 22.2 / 19.2 (16.8 at 98k) | a conversation must go past 81,920 |
| f16 cache, no drafter | 40,960 | 28.1 / 25.7 / 24.8 / – | never: fewer tokens and slower than q4_0 + drafter |

- **Speed against context:** the served setting gives up 4% of f16's speed with nothing held and
  15% at 16k, for 3.3x the context.
- **Speed against the full context:** the last 49,152 tokens of the model's 131,072 cost the
  drafter, which is half the generation speed at every depth.
- **Greedy text** with q4_0 matches f16 word for word less often (GSM8K 82 of 1,319 replies
  against 294 for q8_0), and the drafter changes greedy text too; the scores do not change.
- **A long prompt read from scratch is slow:** at about 406 tok/s, 65k tokens take about 160 s
  (arithmetic). In one conversation the prompt cache reuses what was already read, so each turn
  reads only its new text.

## Parameters tested and left at default

| parameter | result |
| --- | --- |
| drafted tokens | 1: 42.9 / 30.9 / 27.3 tok/s at 0 / 32k / 65k, well behind 2. 3: 56.0 / 39.3 / 32.1, level with 2 within the spread. 2 kept. |
| `-ub` (prompt batch) | 1024 and 2048 do not load beside the drafter, even at 65,536. Without it, 1024 reads prompts faster in one load but holds 114,688 tokens, not 131,072. 512 kept. |
| `--parallel` | 1: llama.cpp splits the context across slots, and the workload is one conversation. |

## Gaps

What these results do not cover, most important first:

1. **GSM8K and BFCL ran with short prompts.** Quality with a long conversation held rests on the
   recall test alone; tool-call accuracy at 32k–80k tokens held is untested.
2. **f16 does not fit past 40,960 tokens**, so q4_0's recall beyond 32k has no f16 reference. The
   fall from 85% of codes at 16k to about 70% is level across caches up to 32k; beyond that it is
   attributed to the model by inference.
3. **No replayed multi-turn agentic session.** The tests put one document ahead of one question;
   prompt cache reuse across many turns, with the sliding-window layers, was not measured.
4. **One load per speed setting.** Two loads of one setting read a 32k prompt at 589 and 673
   tok/s, so gaps under about 15% (q8_0 against q4_0, 3 against 2 drafted tokens) are unresolved.
5. Generation at 128k held, thinking on, sampled decoding, flash attention and thread count were
   not measured on the 12B.
