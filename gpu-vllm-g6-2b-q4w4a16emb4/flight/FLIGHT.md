# G6 serving-engine flight

The 3050 flight (`local-vllm-3050oem-2b-q4w4a16emb4/benchmarks/runs/2026-11-03-engines-3050oem/PAPER-TABLE.md`)
on one NVIDIA L4 in an EC2 G6: the same six engines, the same Gemma 4 E2B weights on the QAT Q4_0 grid, the same
requests and statistics. Scaffolded 2026-11-05; steps 1–5 ran the same day (`../benchmarks/runs/2026-11-05-engines-g6/REPORT.md`) and step 6 on 2026-11-06
(`../benchmarks/runs/2026-11-06-suite-g6/REPORT.md`).

| engine | rig | port | weights |
|---|---|---|---|
| llama.cpp | `gpu-llamacpp-g6-2b-q4_0` (exists, 2026-10-26) | 8080 / flight 8099 | exact Q4_0 GGUF |
| vLLM 0.33.0 | `gpu-vllm-g6-2b-q4w4a16emb4` | 8010 | W4A16, int4 tables |
| SGLang 0.5.23 + 2 patches | `gpu-sglang-g6-2b-q4w4a16emb4` | 8030 | the vLLM rig's checkpoint |
| mistral.rs v0.9.5 | `gpu-mistralrs-g6-2b-q4_0` | 8020 | the llama.cpp rig's GGUF |
| PyTorch int4 engine | `gpu-pytorch-g6-2b-q4w4a16` (exists, 2026-10-26) | flight 8100 | W4A16 |
| LiteRT-LM 0.19.0 | `gpu-litertlm-g6-2b-int248`, `gpu-litertlm-g6-2b-q4w4a16emb4` | 9379, 9380 | Google's int248; the exact Q4_0 build |

## Order

1. **Instance**: `gpu-pytorch-g6-2b-q4w4a16`'s MCP `create_g6_instance` (g6.2xlarge spot, us-east-1, the
   PyTorch 2.13 / Ubuntu 26.04 DLAMI, 100 GB gp3). Its bootstrap installs the PyTorch engine.
2. **Code onto the instance** at `/opt/gemma4-dev` (the six rig directories are enough). SSM returns at most
   24,000 characters, so ship a tarball through S3 or in chunks and check its SHA-256.
3. **The exact Q4_0 LiteRT build** exists only on the 3050 host: there,
   `make -C gpu-litertlm-g6-2b-q4w4a16emb4 stage MODEL_S3_URI=s3://...`.
4. **`flight/setup.sh`** on the instance: llama.cpp (toolkit, build at `7182ef71e`, the 3050 flight's commit,
   model, drafter), vLLM (pip, install, check, W4A16 drafter), SGLang, mistral.rs (Rust, build for sm_89),
   LiteRT-LM (Vulkan check, both files).
5. **`flight/engines.py OUT arms`**, one server at a time:
   `vllm,vllm-d4,llama,llama-d2,mistralrs,mistralrs-d1,mistralrs-card,sglang,sglang-d4,pytorch,litert-q4,litert-int248,litert-int248-d`.
   Then `summarize.py OUT`. File the run as `gpu-vllm-g6-2b-q4w4a16emb4/benchmarks/runs/<date>-engines-g6/`
   with `engines.py` and `summarize.py` copied beside the results.
6. **Concurrency and accuracy**, per rig, as on the 3050: each rig's `sweep.py` (512/128 and 128/512 at
   1–64 requests), and GSM8K (1,319) and BFCL simple (400) through `jev-tpu-v5e1/gen_eval.py`, one request at
   a time, greedy, thinking off. **`flight/suite.py OUT sweep:<arm>,acc:<arm>,...`** runs both, one server at a
   time, and skips jobs already marked `DONE`, so a failed job reruns alone; `flight/proxy.py` puts LiteRT-LM's
   `max_tokens` where its server reads it. File the run as `<date>-suite-g6/` with copies of both.

## Unknowns the first steps answer

Answered 2026-11-05:

- vLLM's PyPI `torch==2.14.0` is `+cu130` (CUDA 13.0) and runs on driver 595.112.07.
- The DLAMI lists the L4 as a Vulkan device; LiteRT-LM serves on it.
- SGLang needed nothing beyond the 3050's list; its JIT kernels need `ninja`, found only with `~/.local/bin` on PATH.
- mistral.rs does not load 64 requests (25,316 MiB of planned activations) or 32 with a 65,536-token cache;
  `mistralrs-card` is 32 requests with a 16,384-token cache.
- mistral.rs links `libcurand` (now in its `deps`); pip `--user` scripts need `~/.local/bin` on PATH under SSM.

## Compare with the 3050

Same requests, statistics and arm names, so rows line up by arm. llama.cpp is at `7182ef71e` on both. The
two flights differ in card, host CPU (8 vCPUs here, `-t 4 -tb 8` for llama.cpp) and day; ratios between
engines on one card are the result, a G6 number against a 3050 number is a hardware comparison of two hosts.
