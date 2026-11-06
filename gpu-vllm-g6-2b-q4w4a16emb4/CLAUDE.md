# CLAUDE.md — `gpu-vllm-g6-2b-q4w4a16emb4`

vLLM 0.33.0 serving Gemma 4 E2B from `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` (Google's QAT Q4_0 grid as
compressed-tensors W4A16, vocabulary tables int4, text only) on one NVIDIA L4 in an EC2 G6. Forked 2026-11-05 from
`local-vllm-3050oem-2b-q4w4a16emb4` for the G6 serving-engine flight; slots 1 and 3 are the only slots that differ.

**No `server.py`, MCP server or skill.** The instance is `gpu-pytorch-g6-2b-q4w4a16`'s `create_g6_instance` (g6.2xlarge, us-east-1, the PyTorch 2.13 / Ubuntu 26.04 DLAMI, SSM access); this rig is `tpu.env`, a `Makefile` run on that instance, and its measurements. `tpu.env` decides every value. **Measured once, 2026-11-05** (`benchmarks/runs/2026-11-05-engines-g6`): 140.4 tok/s for one request, 275.3 with the W4A16 drafter at 4 tokens, 537.2 / 711.3 at 4 requests at once, on vLLM's PyPI `torch 2.14.0+cu130`; the serving values are still the 3050 rig's. **Sweeps and accuracy 2026-11-06** (`benchmarks/runs/2026-11-06-suite-g6`): 1,593.7 / 3,715.7 tok/s at 64 requests (512 in / 128 out, 128 / 512), 4,516.3 with the drafter at 4 tokens; GSM8K 0.8969 and BFCL 0.9225, level with the 3050. The flight driver is `gpu-vllm-g6-2b-q4w4a16emb4/flight/engines.py`; `flight/setup.sh` installs every engine in order. One engine serves at a time on the card.

**This rig also holds the flight** (`flight/`): `FLIGHT.md` (order, rigs, ports, unknowns), `setup.sh`, `suite.py` (sweeps and accuracy, resumable) with `proxy.py`, `engines.py`
(the 3050 driver with G6 rig paths, a `mistralrs-card` arm, llama.cpp threads from its `tpu.env`) and `summarize.py`.
The 3050 flight's driver lived in its run directory; file the G6 run as `benchmarks/runs/<date>-engines-g6/` with
copies of both scripts.

Things that differ from the 3050 rig:

- **The interpreter is the instance's system `/usr/bin/python3`** (Ubuntu 26.04: 3.14), user site. Not the DLAMI's torch
  venv, which keeps torch 2.13 for `gpu-pytorch-g6-2b-q4w4a16`.
- **vLLM keeps its torch pin here** (`torch==2.14.0` from PyPI): the 3050 skipped it over a system 2.14.1, and this
  instance has no system torch. Which CUDA that wheel targets against driver 595.112.07 is unverified; `make check`.
- **`make drafter` builds the W4A16 drafter on the instance** with the 3050 rig's `repack/repack_drafter.py` from
  Google's assistant at the pinned revision, into `DRAFTER_DIR`. SGLang's drafter arm reads it from there.
- `sweep.py` and `attest.py` are the 3050 rig's, unchanged.

`python3 -m unittest discover -s tests -v` is offline. `make lint` lints a hardcoded list.
