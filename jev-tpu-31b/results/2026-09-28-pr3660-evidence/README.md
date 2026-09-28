# 2026-09-28: QAT W4A16 retest on vllm-project/tpu-inference#3660

Google's `-qat-w4a16-ct` checkpoints for E2B, E4B and 12B, served on one v6e chip (flex-start
`ct6e-standard-1t`, asia-northeast1-b) with the PR branch's `tpu_inference`. All three load and serve,
with memory, KV cache size and throughput matching the 2026-09-25 runs (`../2026-09-25-w4a16-QUANT.md`).

| Model | HBM used | KV cache tokens | Output tok/s, median (range) | Boot | PyTorch fallback |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `google/gemma-4-E2B-it-qat-w4a16-ct` | 7.17 GiB | 1,256,448 | 4169.7 (3940.3 to 4182.0) | 737 s | no |
| `google/gemma-4-E4B-it-qat-w4a16-ct` | 10.15 GiB | 348,160 | 2176.5 (2145.8 to 2183.5) | 797 s | no |
| `google/gemma-4-12B-it-qat-w4a16-ct` | 9.46 GiB | 60,160 | 990.3 (990.1 to 992.3) | 812 s | no |

2026-09-25, same checkpoints: 7.17 / 10.15 / 9.46 GiB, 1,256,448 / 348,160 / 60,160 tokens,
4060 / 2157 / 984 tok/s (12B: 993 with `--hf_overrides`, as here).

Table values are extracted from the boot logs and `*.load.json` in this directory, not transcribed.

#### Build

- Image: `vllm/vllm-tpu@sha256:ea98ed065a86d3e69ec7142b9f4766aa3cd7e475d752e680f4ef7612745aca61`
  (`nightly` on 2026-09-28, vLLM `0.30.1rc1.dev221+gc8d7a7dd1`).
- `tpu_inference`: xbill9/tpu-inference `gemma4-w4a16-moe` at `ba2045c` (#3660 rebased on `main`,
  which contains #3653), plus #3299 (`gh pr diff 3299`, sha256 prefix in `kvshare.diff.sha256-16`).
  The image runs `tpu_inference` as an editable install from `/workspace/tpu_inference`; the PR tree
  replaces it there, and `build.log` confirms the imported `wna16.py` has #3660's `_expert_shape`.
- `tokamax` at the PR's pin, `0.0.15.dev20260927`.

#### Method

`qat_retest.sh` serves each model with `serve.sh`'s flags (`--max-model-len 2048 --max-num-seqs 16
--max-logprobs 32 --generation-config vllm --enable-prefix-caching`, TP=1); E2B and E4B with every
modality limited to 0, 12B with `--hf-overrides '{"architectures":["Gemma4ForCausalLM"]}'`.
Then `../../tpu/w4a16_client.py record` (8 fixed prompts, greedy, 64 tokens) and `load`
(16 concurrent requests of exactly 256 tokens, `ignore_eos`, one warm-up pass, median of 3).
No compile cache, so boot includes the full compile.

#### Scope

Load, serve and throughput only. The 3,880-record suite was not run, so accuracy is not re-measured.
Not covered: 26B (needs the repacked GCS checkpoint), TP above 1, other chips.
