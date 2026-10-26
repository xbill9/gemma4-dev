# CLAUDE.md — `gpu-llamacpp-g6-2b-q4_0`

llama.cpp serving Gemma 4 E2B as the exact Q4_0 GGUF (`xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`, Google's QAT grid)
with Google's QAT assistant drafter as exact Q4_0, on one NVIDIA L4 in an AWS EC2 G6. Created 2026-10-26 to measure the
drafter on a datacenter card beside the RTX 3050 and GTX 1650 Ti `local-llamacpp-*` rigs.

**No `server.py`, MCP server or skill.** The instance comes from `gpu-pytorch-g6-2b-q4w4a16`'s `create_g6_instance`
(same AMI, same SSM access); this rig is `tpu.env`, a `Makefile` to run on that instance (`toolkit`, `build`,
`download`, `draft`, `serve`) and its measurements. `tpu.env` decides every value.

**Result (`benchmarks/runs/2026-10-26-drafter-g6/REPORT.md`)**: no drafter 133.4 tok/s; the drafter 1.391x / 1.582x /
**1.631x** / 1.474x at 1 / 2 / 3 / 4 drafted tokens (217.6 tok/s at 3). Acceptance matches the 3050 and 1650 Ti at each
draft length; the card sets the best length (1650 Ti 1 token 1.18x, 3050 2 tokens 1.40x, L4 3 tokens 1.63x). 2 tokens
still 1.49x at 120,000 tokens of context. GSM8K 0.8992 / 0.8984 and BFCL 0.930 / 0.925 without / with, level; GSM8K
1.42x sooner at 8 requests. 1 drafted token keeps greedy text identical to no drafter (18/18).

`SPEC_DRAFT_N_MAX=2` in `tpu.env` is the 3050 rig's serving value, kept so the two rigs serve alike; 3 measured 3%
faster here on greedy and 1% on sampled.

Things that differ from the 3050 rig:

- **The toolkit is CUDA 13.1 from NVIDIA's apt repo** (`make toolkit`): the DLAMI has no `/usr/local/cuda`, and the
  driver (595.112.07) supports CUDA 13.2, so 13.3/13.4 would be newer than the driver.
- **Threads `-t 4 -tb 8`** for the 8 vCPUs of `g6.2xlarge`.
- **Models live in `/opt/models`** on the instance; the hashes in `tpu.env` decide.
- `hf_hub_download` needs the full revision hash; the short `74b2c92` failed silently in a background download.
- SSM returns at most 24,000 characters; fetch result archives in chunks and check their SHA-256.

The PyTorch engine on the same card decodes the same QAT int4 weights at 141.35 tok/s, at the card's memory limit for
one request (`gpu-pytorch-g6-2b-q4w4a16/benchmarks/runs/2026-10-26-first-light-g6`).
