# CLAUDE.md — `gpu-llamacpp-g6-4b-q4_0`

llama.cpp serving Gemma 4 E4B as the exact Q4_0 GGUF (`xbill9/gemma-4-E4B-it-qat-q4_0-exact-gguf`, Google's QAT grid)
with Google's QAT assistant drafter as exact Q4_0, on one NVIDIA L4 in an AWS EC2 G6. Created 2026-10-26, the E4B
sister of `gpu-llamacpp-g6-2b-q4_0`: same instance, binary, flags and scripts, so slot 4 is the only difference.

**No `server.py`, MCP server or skill.** The instance comes from `gpu-pytorch-g6-2b-q4w4a16`'s `create_g6_instance`;
this rig is `tpu.env`, a `Makefile` to run on that instance (`toolkit`, `build`, `download`, `draft`, `serve`) and its
measurements. `tpu.env` decides every value. Models live in `/opt/models/e4b` and `/opt/models/e4b-draft`, so both
rigs' models fit on one instance; both serve on port 8080 and run one at a time.

**Result (`benchmarks/runs/2026-10-26-drafter-g6/REPORT.md`)**: no drafter 74.0 tok/s; the drafter 1.443x / 1.646x /
**1.754x** / 1.754x at 1 / 2 / 3 / 4 drafted tokens (129.7 tok/s at 3; sampled, 3 leads 4). Acceptance matches the 3050
and 1650 Ti at each draft length; best length 1 on the 1650 Ti (1.49x), 2 on the 3050 (1.72x), 3 on the L4. 2 tokens
still 1.57x at 120,000 tokens of context. GSM8K 0.9310 / 0.9287 and BFCL 0.910 / 0.915 without / with, level; GSM8K
1.58x sooner at 8 requests. Against E2B on this card: 0.555x without the drafter, 0.596x with it at best, so E4B with
the drafter runs at 0.972x E2B without; GSM8K +0.030 (paired range above zero), BFCL level.

Greedy text with the drafter differs from no drafter even at 1 drafted token (3/18 identical), unlike E4B on the 3050
and 1650 Ti and E2B on this card (18/18). Never use this rig's drafted greedy text as a byte-identical reference.

`SPEC_DRAFT_N_MAX=2` in `tpu.env` is the 3050 rig's serving value, kept so the rigs serve alike; 3 measured 6.5% faster
here on greedy and 6.0% on sampled.

The CUDA 13.1 toolkit, `-t 4 -tb 8`, the full-hash `hf_hub_download` rule and the 24,000-character SSM limit are as in
`gpu-llamacpp-g6-2b-q4_0/CLAUDE.md`.

**Why the card sets the draft length** (`gpu-llamacpp-g6-2b-q4_0/benchmarks/runs/2026-10-26-drafter-cost-g6`): on the 1650 Ti, 3050 and L4 a
step yields the same tokens per draft length (about 1.60 / 1.94 / 2.15 at 1 / 2 / 3); the cost of a step relative to a
plain one is what differs (E2B at 3 tokens: 2.119x on the 1650 Ti, 1.356x here). On the L4 the target checks up to 4
tokens in one pass for at most 1.04x the time of one, so the added time is the drafter's, 0.75–1.18 ms per drafted
token: 33% of an E2B step at 3 tokens and 19% of an E4B step, which is why E4B gains more. With several requests at
once (`mixed.py`, `-kvu`) 2 drafted tokens pay up to 16 requests (E2B 1.06–1.19x, E4B 1.15–1.34x) and lose at 64
(0.87x); 3 tokens lead only for one request.
