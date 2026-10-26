# 2026-10-26 — first light on the L4 (EC2 G6)

`gpu-pytorch-g6-2b-q4w4a16` through its own `server.py`: `create_g6_instance` (spot `g6.2xlarge`, `us-east-1d`,
`i-0c91ba4c8f801833e`, AMI `ami-0cf9f0c6e35905e06`), `get_install_progress`, `verify_gpu_arch`, `deploy_torch_server`
(build `555f8e0c550f`), `verify_model_health` (✅, reply `ok`, not degenerate, build ids match).

NVIDIA L4 (sm_89, 23034 MiB), driver 595.112.07, torch 2.13.0+cu130, triton 3.7.1, transformers 5.16.1,
compressed-tensors 0.17.0, accelerate 1.14.0, huggingface_hub 1.31.0. Engine files byte-identical to
`local-pytorch-1650ti-2b-q4w4a16`, including its 1650 Ti `w4gemv.py` tilings.

#### Start-up

Install 197 s (`[stage]` lines of `/var/log/torch-install.log`: torch-interpreter 100 s, torch-deps 57 s). The server's
first line: prompt reads chose **fp16** (GEMM 47.22 TFLOP/s fp16, 11.12 fp32), weights 2,498 MiB, K/V 774 MiB for
131,072 tokens, ready 293.2 s after start (first start downloads both checkpoints; later loads 33–38 s).

The first start failed: the install's `pip install --upgrade` took `huggingface_hub` to 2.3.0 and transformers 5.16.1
refuses to import outside `>=1.5,<2`. The bootstrap now pins `huggingface_hub==1.31.0`.

#### Depth: prompt read, decode, recall

`depth_bench.py`, the 1650 Ti run's method (`../../../../local-pytorch-1650ti-2b-q4w4a16/benchmarks/runs/2026-10-23-long-context-1650ti`):
the same seeded prompts in the same depth order, 63 greedy tokens decoded with stop tokens off, engine called directly
on the instance. Prompt tok/s is the whole prompt read from position 0.

| Context | Prompt read, fp16 | Decode | Recall | 1650 Ti decode | L4 / 1650 Ti | File |
| ---: | ---: | ---: | --- | ---: | ---: | --- |
| 61 | 164.6 | **141.35** | ✅ | 87.84 | 1.61x | `short.jsonl` |
| 4,081 | 2,042.8 | **133.08** | ✅ | 80.38 | 1.66x | `short.jsonl` |
| 32,761 | 2,640.2 | **96.14** | ✅ | 51.04 | 1.88x | `deep.jsonl` |
| 130,981 | 1,399.5 (93.6 s) | **51.41** | ❌ (567239 for 492097) | 21.81 | 2.36x | `deep.jsonl` |

Prompt reads are 17.9x (32k) and 21.6x (131k) the 1650 Ti's. Peak allocated 3,395 MiB at 131k.

Recall matches the 1650 Ti prompt for prompt: the 131k prompt gets the same wrong code (567239) on both cards, and the
32,761-token prompt of the short-order run (`short.jsonl`, code 713468) gets 319550 in fp16 and fp32 here, as the
1650 Ti's fp16 run missed it.

fp32 prompt reads (`short-fp32.jsonl`): 158.5 / 1,174.6 / 1,279.9 tok/s at 61 / 4,081 / 32,761, about half of fp16
from 4k on; decode 139.16 / 130.68 / 96.39.

`deep-131072-overflow.jsonl` is a first deep pass at depth 131072: its 131,071-token prompt left no room to decode, so
it carries no decode figure. The 1650 Ti run used 131000.

#### Where a decode step goes

`profile_decode.py`, torch.profiler over 50 replays of the captured step at short context (`profile.json`):
7.287 ms wall per step, 7.298 ms of kernels, 537 kernels per step.

| Kernel | ms per step | Calls |
| --- | ---: | ---: |
| `_w4_gemv` (int4 matrix-vector) | 6.001 | 212 |
| `_dec_split` + `_dec_combine` (decode attention) | 0.601 | 70 |
| everything else (norms, RoPE, GELU, argmax) | 0.696 | 255 |

#### The L4's memory ceiling

`cold_gemv.py` (`cold.jsonl`): a 1 GiB device copy reads and writes at **241.2 GB/s**. The GEMV per shape, with
weights rotated through enough copies that the 48 MiB L2 cannot hold them, runs at 225–247 GB/s on every shape of
1536×768 packed and larger (the 262,144-row vocabulary table: 246.8) and at 53–183 GB/s on the five smallest, which
finish in 2.7–9.7 µs.

Per token the GEMV moves 1.282 GB (1,222.8 MiB, counted by `kbench.py` in the 1650 Ti run) in 6.001 ms: 213.7 GB/s,
0.886 of the copy rate. At the copy rate it would take 5.32 ms, a 6.60 ms step and 151.5 tok/s. Single-request decode
is therefore within about 7% of what this card's memory allows for this engine.

`tune_gemv.py` (`tune.jsonl`) ran v4's autotuner against the pinned v5 tilings on the same shapes with the weights
held in L2; v4 was nowhere faster, so the 1650 Ti tilings stay. Its GB/s figures above 241 are L2 hits, not DRAM.

#### Against v5e-1

`tpu-vllm-v5e1-2b-q4w4a16emb4/benchmarks/runs/2026-10-02-rig-boot-v5e1` (vLLM, the same QAT int4 E2B with int4 tables,
256 output tokens, on the VM): 128.2 tok/s for one request, 446.0 for four, 1,171.0 for sixteen. A different method
(HTTP, 256 tokens), so one request reads as level, the L4 slightly ahead. This engine serves one request at a time;
vLLM's lead is batching.

#### Files

| File | What |
| --- | --- |
| `depth_bench.py`, `short*.jsonl`, `deep*.jsonl`, `*.log` | depth runs |
| `profile_decode.py`, `profile.json` | decode-step kernel profile |
| `cold_gemv.py`, `cold.jsonl` | copy bandwidth and cold-L2 GEMV per shape |
| `tune_gemv.py`, `w4gemv_v4.py`, `tune.jsonl`, `tune.log` | autotune against the pinned tilings |
| `server-journal.log` | the service journal |
