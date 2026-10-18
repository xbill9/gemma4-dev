# 2026-10-17 — context cache type for the 12B on the RTX 3050 OEM: context, speed, answers

**Which context cache type, drafter setting and context size serve one long conversation best on
the 8 GB card?** The workload is one agentic session at a time (`--parallel 1`) whose context keeps
growing, so speed with a deep context and the most context that fits count most. Same binary
(llama.cpp `7182ef71e`), model, drafter and flags as the rig; only `-ctk/-ctv`, `-c`, the drafter,
`-ub` and the drafted-token count vary.

- `kv_sweep.py` (`run.sh` step 1): f16 / q8_0 / q4_0 cache, each with and without the drafter at
  2 tokens, at the largest `-c` that loads. Generation and prompt reading with 0 to 98k tokens of a
  long document (`/usr/share/doc/NVIDIA_GLX-1.0/README.txt`, hash in `doc.sha256`) in context,
  three questions per depth, greedy, 256 tokens, prompt cache on. `summarize.py` → `summary.txt`.
- `tune.py`: the q4_0 cache's limits with the drafter, prompt batch size (`-ub` 1024, 2048) and
  1 or 3 drafted tokens. `recall.py`: 8 six-digit codes spread through the document, all asked for
  at once, 10 trials per depth, same seeds for every arm. `summarize2.py` → `summary2.txt`.
- GSM8K (1,319) and BFCL simple (400) through `jev-tpu-v5e1/gen_eval.py`, one request at a time,
  at each cache's largest `-c`: q8_0 by `run.sh`, q4_0 by `run_q.sh`. `compare.py` → paired by
  question against the f16 runs of `../2026-10-16-accuracy-12b-3050oem` with the same drafter
  setting.

## Result

**Served now: q4_0 cache, drafter at 2 tokens, `-c 81920`.** 3.3x the f16 setting's context
(24,576) at the same answers, 0.96x its generation speed with nothing in context and 0.85x at 16k.
Without the drafter the q4_0 cache holds the full 131,072 tokens at half the speed.

### Context that fits (7,853 MiB of 8,192 at most)

| cache | with drafter | without |
| --- | ---: | ---: |
| f16 | 24,576 | 40,960 |
| q8_0 | 49,152 | 65,536 |
| q4_0 | **81,920** (86,016 fails) | **131,072** |

`-ub 1024` or `2048` with the drafter fails to load even at 65,536 (the drafter's compute buffer
does not fit); without the drafter `-ub 1024` loads at 114,688 and fails at 131,072.

### Generation tok/s by tokens already in context (geomean of three questions)

| cache, drafter | 0 | 8k | 16k | 32k | 49k | 65k | 98k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f16, drafter | 57.5 | 52.2 | 49.8 | – | – | – | – |
| f16, none | 28.1 | 26.1 | 25.7 | 24.8 | – | – | – |
| q8_0, drafter | 55.1 | 45.7 | 43.3 | 39.3 | – | – | – |
| q8_0, none | 27.6 | 25.4 | 24.4 | 22.6 | 21.0 | – | – |
| **q4_0, drafter** | **55.1** | **44.0** | **42.4** | **37.8** | **33.5** | **30.8** | – |
| q4_0, none | 27.5 | 25.2 | 24.1 | 22.2 | 20.6 | 19.2 | 16.8 |

- The drafter keeps paying with a deep context: q4_0 2.00x at 0 tokens, 1.70x at 32k, 1.60x at
  65k. Drafted tokens kept 0.61–0.71 at every depth and cache type.
- Cache type matters little without the drafter (q4_0 0.94x f16 at 16k) and more with it (q4_0
  0.85x f16 at 16k, q8_0 0.87x): a quantized cache slows the drafter's checks more than single
  steps. q4_0 is 0.96x q8_0 at 32k with the drafter.
- Prompt reading is level across cache types: about 840 t/s for 8k of document, 720 at 16k, 600
  at 32k, 485 at 49k, 408 at 65k, 333 at 98k.
- **1 or 3 drafted tokens** (q4_0, 81,920): 1 token 42.9 / 30.9 / 27.3 tok/s at 0 / 32k / 65k,
  well behind 2; 3 tokens 56.0 / 39.3 / 32.1 against 2 tokens' 55.1 / 37.8 / 30.8 in a separate
  load, within the run-to-run spread of this card (prompt reading at 32k measured 589 and 673 t/s
  in two loads of the same setting). 2 stays.
- **`-ub 1024`** without the drafter: generation level (22.0 / 16.7 tok/s at 32k / 98k against
  22.2 / 16.8), prompt reading 705 / 399 t/s against 599 / 333, but at 114,688 tokens instead of
  131,072, and within the same run-to-run spread. Not adopted.

### Answers, paired by question against the f16 cache

| | GSM8K | minus f16 (95% range) | BFCL simple | minus f16 (95% range) |
| --- | ---: | ---: | ---: | ---: |
| f16, drafter (`-c 24576`) | 0.9598 | | 0.9500 | |
| f16, none (`-c 24576`) | 0.9629 | | 0.9475 | |
| q8_0, drafter (`-c 49152`) | 0.9598 | +0.0000 (−0.0053, +0.0053) | 0.9475 | −0.0025 (−0.0075, 0) |
| q8_0, none (`-c 65536`) | 0.9591 | −0.0038 (−0.0099, +0.0015) | 0.9475 | 0 (0, 0) |
| **q4_0, drafter (`-c 81920`)** | **0.9613** | +0.0015 (−0.0045, +0.0076) | **0.9500** | 0 (−0.0100, +0.0100) |
| q4_0, none (`-c 131072`) | 0.9621 | −0.0008 (−0.0076, +0.0061) | 0.9500 | +0.0025 (−0.0075, +0.0125) |

Every range includes zero. Questions scored differently from f16: GSM8K 12 / 15 / 16 / 19 and
BFCL 1 / 0 / 4 / 5 for the four rows. Word-for-word identical GSM8K replies fall with the cache's
precision (294 and 334 of 1,319 for q8_0, 82 and 69 for q4_0); the scores do not. Wall time is set
by the drafter, not the cache: GSM8K 6,566 / 6,905 / 7,032 s with the drafter, 15,362 / 15,256 /
15,625 s without.

### Recall of eight codes spread through the document (codes right of 80)

| cache, drafter, `-c` | 16k | 32k | 49k | 64k | 98k | 127k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f16, none, 40,960 | 69 | 55 | – | – | – | – |
| q8_0, none, 65,536 | 68 | 58 | 55 | 57 | – | – |
| q4_0, none, 131,072 | 67 | 58 | – | 59 | 48 | 56 |
| q4_0, drafter, 81,920 | – | 58 | – | 57 | – | – |

Level across cache types at every depth they share, and the drafter changes nothing. Recall falls
with depth on every cache: about 85% of codes at 16k, 70% from 32k on (60% at 98k, 70% at 127k); all eight in
one reply in at most 3 of 10 trials at 16k and at most 1 of 10 beyond. That is the model's limit
on this task, not the cache's.

## Serving setting

`tpu.env`: `KV_CACHE_TYPE=q4_0`, `CONTEXT_SIZE=81920`, drafter at 2 tokens, `-ub` at the default
512. For a conversation that must go past 81,920 tokens, empty `SPEC_DRAFT_MODEL` and set
`CONTEXT_SIZE=131072`: the same answers at about half the generation speed.
