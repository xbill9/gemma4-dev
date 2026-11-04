# vLLM 0.33.0 accuracy with the drafter, RTX 3050 OEM — 2026-11-03

GSM8K (1,319, 2,048-token limit) and BFCL simple (400, 512-token limit) through `jev-tpu-v5e1/gen_eval.py`,
greedy, one request at a time. This is `../2026-11-03-accuracy-3050oem` with one change: the W4A16 drafter at 4
drafted tokens, vLLM's best for one request in `../2026-11-03-drafters-3050oem`. The command is in `command.txt`.
`compare.py` prints `COMPARE.txt`. llama.cpp's figures are from its 2026-10-14 run on `77c4540ce`
(`local-llamacpp-3050oem-2b-q4_0/.../2026-10-14-accuracy-fullctx-3050oem`, exact Q4_0 drafter at 2 tokens).

#### Result

| | GSM8K | BFCL simple | GSM8K wall | BFCL wall |
|---|---|---|---|---|
| vLLM + drafter | 0.8992 (1,186) | 0.9225 (369) | **1,479 s** | **57.9 s** |
| vLLM | 0.9014 (1,189) | 0.925 (370) | 4,360 s | 139.9 s |
| llama.cpp + drafter | 0.8969 (1,183) | 0.925 (370) | 2,831 s | 97.8 s |
| llama.cpp | 0.9014 (1,189) | 0.9225 (369) | 4,993 s | 152.4 s |

**The drafter changes no score.** The paired ranges include zero:
- vLLM with the drafter against vLLM without it: GSM8K −0.0023 [−0.0106, +0.0061], BFCL −0.0025 [−0.0075, +0.0000].
- vLLM with the drafter against llama.cpp with its drafter: GSM8K +0.0023 [−0.0091, +0.0136].

**The drafter cuts vLLM's GSM8K time 2.95x** (4,360 → 1,479 s) and BFCL's 2.42x. On llama.cpp it cut them
1.76x and 1.56x. With drafters on both, vLLM finishes GSM8K 1.91x sooner than llama.cpp and BFCL 1.69x sooner.
GSM8K's step-by-step arithmetic drafts well: the 2.95x is above the 1.94x measured on chat prompts.
