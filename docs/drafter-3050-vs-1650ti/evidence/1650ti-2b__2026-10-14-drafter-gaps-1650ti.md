# 2026-10-14 — drafter gaps on the GTX 1650 Ti: draft length, Google's non-QAT drafter, long conversations

**Answer: on this card 1 drafted token is the best setting for both models, Google's non-QAT
drafter trails the exact Q4_0 one at every length, and with long text in context E2B's drafting
stops paying while E4B's keeps it.** E2B at 1 token goes from 1.27x with nothing ahead to 1.03x at
32k tokens ahead and 0.89x at 120k; E4B at 1 token holds 1.52x to 1.45x from 0 to 32k. On the RTX
3050 both kept paying to 120k (E2B 1.23x, E4B 1.41x at 1 token).

## Setup

The laptop `yogagpu` (GTX 1650 Ti Max-Q, 4096 MiB, driver 615.71.09), llama.cpp `77c4540ce`
(build 813, `host.txt`), the RTX 3050 runs' build, from `~/llama.cpp-77c4540ce`; the rigs keep
serving on `f95b0d9`. `speed.py`, `longctx.py`, `common.py` and the summarizers are copied from
`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-3050oem`, with the changes
noted in their headers. `run.sh` ran everything in one pass, 10:05–14:31 on 2026-10-15
(`driver.log`).

- **Drafters**: `q4x` is the exact Q4_0 QAT drafter the rigs serve; `nq` is Google's non-QAT
  `google/gemma-4-E?B-it-assistant` as Q8_0, built on the 3050 host and copied here. Both `nq`
  files and the long-context document pass the 3050 run's hashes (`sha256.log`).
- **Targets**: each rig's exact Q4_0 GGUF and serving flags from its `tpu.env`.
- The first piece of the queue, E4B GSM8K and BFCL with the drafter at 1 token, belongs to
  `../2026-10-14-drafter-accuracy-1650ti` and is reported there (0.9310 and 0.9150, level with no
  drafter).

## Draft length and the non-QAT drafter, one request at a time (`e?b/speed/summary.txt`)

`-c 8192 --parallel 1`, six prompts, greedy ×3 plus one sampled pass (temperature 0.8, seed
1234), 256 tokens, prompt cache off, card cooled before each arm. Geometric mean over the six
prompts; "same text" counts greedy replies byte-identical to no drafter, of 18.

| greedy | E2B tok/s | x | acceptance | same text | E4B tok/s | x | acceptance | same text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no drafter | 76.5 | 1.000 | — | — | 38.0 | 1.000 | — | — |
| exact Q4_0, 1 token | **90.3** | **1.179** | 0.602 | 0 | **56.5** | **1.488** | 0.635 | 18 |
| exact Q4_0, 2 tokens | 83.6 | 1.093 | 0.469 | 0 | 54.5 | 1.436 | 0.506 | 3 |
| exact Q4_0, 3 tokens | 77.3 | 1.010 | 0.378 | 0 | 50.9 | 1.342 | 0.414 | 3 |
| non-QAT, 1 token | 88.7 | 1.159 | 0.583 | 0 | 55.7 | 1.467 | 0.595 | 18 |
| non-QAT, 2 tokens | 80.1 | 1.047 | 0.430 | 0 | 53.7 | 1.415 | 0.490 | 3 |
| non-QAT, 3 tokens | 73.2 | 0.956 | 0.342 | 0 | 48.8 | 1.286 | 0.399 | 3 |
| exact Q4_0, 2 tokens, repeated last | 83.5 | 1.091 | 0.469 | 0 | 53.4 | 1.406 | 0.506 | 3 |

Sampled: E2B 1.163 / 1.042 / 0.944x (exact Q4_0, 1/2/3 tokens) and 1.136 / 1.019 / 0.930x
(non-QAT); E4B 1.463 / 1.404 / 1.294x and 1.416 / 1.363 / 1.221x.

- **Each extra drafted token costs on this card.** At 3 tokens E2B is level with no drafter (1.010x)
  or behind it (non-QAT 0.956x). On the 3050, E2B was best at 2 tokens (1.40x).
- **The non-QAT drafter accepts fewer tokens** (0.583 against 0.602 at 1 token on E2B, 0.595
  against 0.635 on E4B) and is behind at every length, by 1.4–5.3%. Same direction as on the 3050.
- **Greedy text**: E4B at 1 token matches no drafter on all 18 replies with either drafter; at 2
  and 3 tokens, 3 of 18. E2B drafted text differs from no drafter on every reply, as in
  `../2026-10-14-drafter-accuracy-1650ti`.
- **Repeat**: the exact Q4_0 2-token arm run last is within 0.2% (E2B) and 2.0% (E4B) of its
  first run.
- **VRAM**: E2B 1,491 / 1,571 / 1,609 MiB and E4B 2,879 / 2,963 / 3,001 MiB for no drafter / exact
  Q4_0 / non-QAT.

## Long conversations (`e?b/longctx/summary.txt`)

The first D tokens of the NVIDIA driver README (English technical prose, same file as the 3050
run), then one of three requests: a story and a code request unrelated to it, and a summary of it.
Greedy, 256 tokens, prompt cache on so the document is read once per arm; the rate is decode only.
Context: E2B at `-c 131072`; E4B at `-c 49152`, the largest of 131072 / 98304 / 69632 / 49152
that loaded with the non-QAT drafter in 4 GiB, so its deepest point is 32,768 tokens.

Mean of x against no drafter over the requests at each depth:

| tokens ahead | E2B 1 token | E2B 2 tokens | E2B non-QAT 2 | E4B 1 token | E4B 2 tokens | E4B non-QAT 2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.266 | 1.177 | 1.144 | 1.524 | 1.501 | 1.494 |
| 8,192 | 1.176 | 1.068 | 1.031 | 1.533 | 1.467 | 1.407 |
| 32,768 | 1.034 | 0.873 | 0.821 | 1.454 | 1.254 | 1.238 |
| 65,536 | 0.965 | 0.717 | 0.688 | | | |
| 120,000 | 0.887 | 0.602 | 0.578 | | | |

No-drafter decode falls from 76.9 to 31.0 tok/s (E2B, 0 to 120k) and 37.9 to 26.1 (E4B, 0 to 32k).

- **E2B's gain is gone by 32k tokens ahead and reverses beyond.** Acceptance holds (code
  0.88–0.91 at 1 token at every depth); what changes is the speed: the story request at 1 token
  is 1.12x with nothing ahead and 0.78x at 120k. Code stays ahead longest (1.03x at 120k).
- **E4B keeps most of its gain at 32k**, 1.45x at 1 token. At 2 tokens the story request drops
  to 0.98x there, so its loss starts with the longer draft.
- **1 token leads at every depth on both models**; the 3050's E4B was best at 2 tokens at every
  depth.
- **Against the 3050** (`2026-10-14-drafter-gaps-3050oem`, `-c 131072` for both models): E2B at 1
  token 1.407 → 1.233x from 0 to 120k, E4B 1.607 → 1.475x from 0 to 32k and 1.408x at 120k.
  Different card and host, same build, document and requests.
- **Replies identical to no drafter**: E2B 2 of 14 at 1 token, 0 at 2 tokens either drafter; E4B
  7 of 8 at 1 token, 2 of 8 at 2 tokens either drafter.
- **VRAM after the last request**: E2B 2,339 / 2,539 / 2,577 MiB (no drafter / exact Q4_0 /
  non-QAT), E4B 3,563 / 3,689 / 3,727 MiB of 4,096.

The 1650 Ti rigs serve at `-c 8192`, where 1 token still pays on both models.
