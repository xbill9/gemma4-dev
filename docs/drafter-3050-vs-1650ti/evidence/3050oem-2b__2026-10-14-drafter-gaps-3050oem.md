# 2026-10-14 — drafter gaps on the RTX 3050 OEM: 1 drafted token, Google's non-QAT drafter, long conversations

**Questions:** three things `2026-10-14-drafter-q4exact-3050oem` left unmeasured on this card, for
E2B (`local-llamacpp-3050oem-2b-q4_0`) and E4B (`local-llamacpp-3050oem-4b-q4_0`):

1. 1 drafted token (the 1650 Ti's best E2B setting).
2. Whether drafting keeps paying with a long conversation in context. The rigs serve 131,072
   tokens; every earlier drafter speed run used 8,192.
3. Google's non-QAT drafter `google/gemma-4-E?B-it-assistant`, which on TPU v5e accepted far more
   drafted tokens with QAT targets than the QAT drafter (`QUANTIZATION.md`).

**Answers:**

1. **E2B: 1 token is 1.38x against 1.40x at 2, and its greedy text is identical to no drafter**
   (18 of 18; 0 of 18 at 2 or 3). E4B: 1 token is well behind (1.54x against 1.72x).
2. **Drafting keeps paying, by less.** E2B at 2 tokens: 1.51x with no text ahead of the question,
   1.36x at 32k, 1.21x at 120k, where 1 token edges ahead (1.23x). Acceptance holds; each
   verification step costs more. E4B holds up better: 1.86x at 0, 1.64x at 32k, 1.55x at
   120k, and 2 tokens stays ahead of 1 at every depth.
3. **The non-QAT drafter is level or behind on llama.cpp.** E2B greedy 1.34x against 1.40x at 2
   tokens, lower acceptance at every count; E4B greedy level (1.73x against 1.72x), sampled behind
   (1.65x against 1.71x). The TPU result does not carry over.

**No serving change.** Both rigs keep the exact Q4_0 QAT drafter at 2 tokens. E2B at 1 token is
the setting to use when greedy output must match serving without a drafter, for short prompts;
with long text ahead of the question the match no longer held (see "Greedy text at `-c 131072`").

## Method

llama.cpp `77c4540ce`, the rigs' serving flags (`-fa 1`, f16 KV, `per_layer_token_embd` on the card,
`--parallel 1`, `--reasoning off`), one llama-server per arm on port 8099, GPU cooled to ≤40 °C
before each arm. `run.sh` drives both scripts for both models and restores the live E4B server.

- **Drafters.** `q4x`: the exact Q4_0 QAT drafter the rigs serve (`tpu.env` `SPEC_DRAFT_MODEL`).
  `nq`: Google's non-QAT drafter converted with llama.cpp's `convert_hf_to_gguf.py --outtype q8_0`
  (`convert.txt`): E2B from revision `2d874ef7`, 97,817,696 B, SHA-256 `3c1cfe8e…`; E4B from
  `8d0031ea`, 98,653,280 B, `90b79bc9…`. Q8_0 because the non-QAT weights are not on a 4-bit grid,
  and the earlier run found Q8_0, Q4_0 and bf16 QAT drafters accept equally.
- **`speed.py`**, `-c 8192`: the six prompts of the earlier runs, greedy ×3 plus one sampled pass
  (temperature 0.8, seed 1234), 256 tokens, prompt cache off. Arms: none; q4x at 1/2/3; nq at
  1/2/3; q4x at 2 again last. Geometric mean over prompts of the median of repeats
  (`summarize_speed.py` → `*/speed/summary.txt`).
- **`longctx.py`**, `-c 131072`: the first D tokens of `/usr/share/doc/NVIDIA_GLX-1.0/README.txt`
  (English technical prose, SHA-256 in `doc.sha256`), D = 0, 8192, 32768, 65536, 120000, then a
  story request and a code request unrelated to it and a request to summarize it. Greedy, 256
  tokens, prompt cache on so each depth's text is read once per arm; the decode rate excludes
  reading the prompt. Arms: none, q4x at 1 and 2, nq at 2 (`summarize_longctx.py` →
  `*/longctx/summary.txt`).

## Speed, `-c 8192`

| greedy, tok/s (x vs none) [acceptance] | E2B | E4B |
|---|---|---|
| none | 108.2 | 58.3 |
| q4x, 1 token | 148.9 (1.38x) [0.609] | 89.8 (1.54x) [0.624] |
| **q4x, 2 tokens (serving)** | **151.2 (1.40x)** [0.475] | **100.0 (1.72x)** [0.481] |
| q4x, 3 tokens | 143.0 (1.32x) [0.384] | 99.3 (1.70x) [0.398] |
| nq, 1 token | 147.5 (1.36x) [0.593] | 89.4 (1.53x) [0.614] |
| nq, 2 tokens | 144.9 (1.34x) [0.433] | 100.7 (1.73x) [0.483] |
| nq, 3 tokens | 135.0 (1.25x) [0.338] | 97.5 (1.67x) [0.386] |
| q4x, 2 tokens again | 151.6 (1.40x) | 100.0 (1.72x) |

- **Sampled** (temperature 0.8): E2B q4x 1.33 / 1.37 / 1.29x at 1/2/3, nq 1.31 / 1.34 / 1.26x;
  E4B q4x 1.53 / 1.71 / 1.67x, nq 1.49 / 1.65 / 1.60x.
- **Greedy text identical to no drafter:** E2B 18 of 18 at 1 token with either drafter, 0 of 18
  at 2 or 3. E4B 18 of 18 in every arm.
- **Repeat arm** within 0.3% (E2B) and 0.0% (E4B) of the first q4x-n2 pass; both match the
  morning run (151.1 and 100.0).
- **VRAM:** E2B 2795 / 2875 (q4x) / 2915 MiB (nq); E4B 4415 / 4499 / 4537 MiB.

## Long conversations, `-c 131072`, E2B

Mean over the three questions of decode tok/s against no drafter at the same depth (the summary
question is not asked at 0):

| tokens ahead of the question | none, tok/s | q4x 1 token | q4x 2 tokens | nq 2 tokens |
|---:|---:|---:|---:|---:|
| 0 | 108 | 1.41x | **1.51x** | 1.44x |
| 8,192 | 100 | 1.38x | **1.39x** | 1.38x |
| 32,768 | 85 | 1.33x | **1.36x** | 1.33x |
| 65,536 | 72 | 1.27x | **1.29x** | 1.26x |
| 120,000 | 56 | **1.23x** | 1.21x | 1.17x |

- By question at 120k, q4x 2 tokens: code 1.54x, summary 1.09x, story 0.99x.
- Acceptance barely moves with depth (code at 2 tokens 0.85 at 0, 0.79 at 120k), so the shrinking
  gain is the verification step getting dearer with a long context, not the drafter guessing worse.
- No-drafter decode at 120k (56.2 tok/s) agrees with llama-bench's 58 at the same depth
  (`2026-10-13-long-context-3050oem`).
- VRAM after the last request: 3643 MiB without a drafter, 3843 with q4x, 3881 with nq.

## Long conversations, E4B

| tokens ahead of the question | none, tok/s | q4x 1 token | q4x 2 tokens | nq 2 tokens |
|---:|---:|---:|---:|---:|
| 0 | 58 | 1.61x | **1.86x** | 1.83x |
| 8,192 | 54 | 1.55x | **1.79x** | 1.78x |
| 32,768 | 46 | 1.48x | 1.64x | **1.65x** |
| 65,536 | 38 | 1.46x | **1.62x** | 1.59x |
| 120,000 | 30 | 1.41x | **1.55x** | 1.48x |

- By question at 120k, q4x 2 tokens: code 1.86x, summary 1.57x, story 1.23x.
- E4B keeps more of its gain than E2B (1.55x against 1.21x at 120k). A likely reason, not measured:
  E4B's own step costs about twice E2B's (29.5 against 56.2 tok/s at 120k), so the drafter's share
  of each step is smaller.
- VRAM after the last request: 6463 MiB without a drafter, 6667 with q4x, 6705 with nq.
- **Interrupted once.** `run.sh` and `longctx.py` stopped at 19:56 after the no-drafter arm,
  with no END line, and an E4B server came up on port 8080 at 19:57:14 from outside the run.
  The three drafter arms ran afterwards from `run_e4b_rest.sh` (`longctx.py ... q4x-n1,q4x-n2,nq-n2`,
  appending to the same `results.jsonl`), 19:59–20:14, same flags and text.

## Greedy text at `-c 131072`

Replies identical to no drafter, of 14 per arm (`summary.txt`): E2B q4x-n1 2, q4x-n2 1, nq-n2 1;
E4B q4x-n1 13, q4x-n2 4, nq-n2 4. At `-c 8192` (speed arms) E2B at 1 token and every E4B arm
matched 18 of 18. **Cause found in `2026-10-14-drafter-divergence-3050oem`: prompt length**, not
`-c`; the speed arms' prompts were ~25 tokens. The prompt cache also changes greedy text on its own,
with or without a drafter.
