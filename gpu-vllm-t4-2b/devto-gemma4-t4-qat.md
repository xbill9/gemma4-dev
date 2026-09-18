---
title: "Gemma 4 on a Tesla T4: QAT Weights Decode 1.79x Faster Than bf16"
published: false
description: "A step by step deployment of Gemma 4 E2B with vLLM on a single Tesla T4 attached to a Compute Engine VM, and a measured comparison of the QAT w4a16 checkpoint against the bf16 reference on the same card."
tags: gemma, vllm, cuda, machinelearning
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-vllm-t4-2b/devto-t4-qat-cover.1937562f.jpg
---

*This article provides a step by step deployment guide for **Gemma 4 E2B** to a **Tesla T4** hosted GPU enabled system. A suite of Python MCP tools is built to simplify management of the vLLM hosted deployment. The T4 is already attached to the Compute Engine VM the tools run on, so there is nothing to provision: the work is getting a Turing GPU to run Gemma 4, and then finding out which checkpoint it runs fastest.*

[github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-t4-2b](https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-t4-2b)

| | |
|---|---|
| Models | `google/gemma-4-E2B-it` (bf16) and `google/gemma-4-E2B-it-qat-w4a16-ct` (QAT, int4 weights) |
| Hardware | 1x Tesla T4, Turing, compute capability 7.5, 15360 MiB, 70 W |
| Host | Compute Engine `n1-standard-2` — 2 vCPU, 7.8 GB RAM, `us-west2-b` |
| Software | vLLM 0.29.0, torch 2.13.0+cu130 |
| Result | QAT decodes at **72.31 tok/s** per stream against bf16's **40.44**, and serves **215.91 tok/s** at 8 streams against **164.62** |

---

## Where Do I Start?

The GPU is already there. A T4 attached to a Compute Engine VM needs no queued resource, no instance launch and no image, so this rig has no provisioning tools at all. Everything it ships is about the software on the box: which Python, which torch, which patch, and which checkpoint.

The MCP server exposes those steps as tools, in the order their failures get more expensive: `check_host_capacity`, `verify_gpu_arch`, `apply_turing_patch`, `verify_turing_patch`, `start_vllm_server`.

## At This Point You Should Have…

- A Compute Engine VM with one Tesla T4 and the NVIDIA driver installed
- Python 3 for the MCP server, and a second interpreter for vLLM if your root disk is small
- Room for the checkpoints on a disk that can hold them
- The rig checked out from the repository above

## What the Machine Is

```
machine-type: n1-standard-2
zone: us-west2-b
nproc: 2
MemTotal:        7614824 kB
```

```
Tesla T4, 7.5, 15360 MiB, 615.71.09, 1590 MHz, 5001 MHz, 70.00 W, 70.00 W
```

Two vCPU and 7.8 GB of RAM is a small host for this GPU, and the RAM shows up below when the weights load.

## Three Disks, Measured Separately

```
Mounted on    1B-blocks        Avail
/           33570021376   8701743104
/tmp         3898789888   3896131584
/opt1      263086084096 200857096192
~/.cache -> /opt1/cache
```

The root disk is small, `/tmp` is its own 3.9 GB filesystem, and `~/.cache` is a symlink onto the large `/opt1` volume. A single `df /` reports the wrong answer for all three writes the install makes. `check_host_capacity` measures each target path on its own.

## Two Pythons

```
python3: Python 3.12.13 at /home/xbill_glitnir_com/.pyenv/shims/python3
PYTHON_BIN: Python 3.13.5 at /usr/bin/python3.13
```

The MCP server runs under `python3`. vLLM runs under `/usr/bin/python3.13` with `PYTHONUSERBASE=/opt1/pyuser`, so its packages land on the big disk. Every tool that asks about torch or vLLM runs its check under that second interpreter.

## The Install as Found

```
torch                                    2.11.0+cpu
vllm                                     0.26.0
```

A CUDA build of vLLM sitting on the CPU build of torch. `import vllm` fails on `libcudart.so.13`, and a CPU torch reports an empty list for every GPU architecture. Two packages present, neither able to serve.

## Which Torch

The fix is the CUDA 13 torch from the PyTorch index, installed with its dependencies. Installing it with `--no-deps` skips cuDNN and the NVIDIA runtime wheels, and `import torch` then fails:

```
ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory
```

With the dependencies in place, `verify_gpu_arch` asks the interpreter what it can run:

```
torch 2.11.0+cu130
torch.version.cuda 13.0
cuda available: True
arch list: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
device: Tesla T4
capability: (7, 5)
fp16 matmul ok: True
```

`sm_75` is in the published wheel. Turing needs no source build.

## Then the Latest vLLM

The stack was upgraded to vLLM 0.29.0. vLLM pins torch to an exact version (`torch==2.13.0`), so the upgrade moves torch with it:

```
torch                                    2.13.0+cu130
transformers                             5.17.0
triton                                   3.7.1
vllm                                     0.29.0
```

## The 64 KiB Problem

Gemma 4 has two attention widths: 256 in its sliding-window layers and 512 in its global layers. vLLM forces its Triton attention backend for that mix, and at width 512 the kernel asks for more shared memory than Turing allows in one block. `apply_turing_patch` clamps the tile sizes in the installed vLLM so they fit:

```
✅ Patched, in the site-packages this host's `python3` imports.

__FILE__/opt1/pyuser/lib/python3.13/site-packages/vllm/v1/attention/ops/triton_unified_attention.py
CLAMP PRESENT
OCCURRENCES 1
```

Reinstalling vLLM removes the clamp. Run `verify_turing_patch` after every vLLM change; `start_vllm_server` refuses to start until it reports the clamp present.

## Loading the Weights Needs Swap

The first start died during weight loading. The kernel log named the process:

```
Out of memory: Killed process 14015 (VLLM::EngineCor) total-vm:27144376kB, anon-rss:3993204kB, file-rss:69212kB, shmem-rss:98540kB, UID:2110064466 pgtables:11488kB oom_score_adj:0
```

The VM ships with no swap, and 7.8 GB of RAM is too little for vLLM to stage these weights. A 16 GB swapfile on the large disk fixes it:

```
$ sudo fallocate -l 16G /opt1/swapfile && sudo chmod 600 /opt1/swapfile && sudo mkswap /opt1/swapfile && sudo swapon /opt1/swapfile
Swap:          16383           0       16383
```

Swap use peaked at 6,761 MiB for bf16 and 4,770 MiB for QAT while the servers started. Add the swapfile to `/etc/fstab` so it survives a reboot.

## Starting the Server

```
/usr/bin/python3.13 -m vllm.entrypoints.openai.api_server --model google/gemma-4-E2B-it-qat-w4a16-ct --host 127.0.0.1 --port 8000 --dtype float16 --kv-cache-dtype auto --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --max-model-len 16384 --max-num-seqs 8
```

`--dtype float16` because Turing has no bfloat16 datapath. vLLM logs the cast and runs the bf16 weights as fp16.

## What the Engine Allocates

```
Using MarlinLinearKernel for CompressedTensorsWNA16
Model loading took 8.02 GiB memory and 98.721582 seconds
GPU KV cache size: 519,568 tokens, Maximum concurrency for 16,384 tokens per request: 31.71x
```

```
Model loading took 9.8 GiB memory and 157.183441 seconds
GPU KV cache size: 315,974 tokens, Maximum concurrency for 16,384 tokens per request: 19.29x
```

The first block is QAT, the second bf16. The QAT build packs its linear layers to 4 bits and keeps its embeddings and vision tower at full width, so its weights take 1.78 GiB less and the KV cache gets the difference.

## Does It Serve?

```
17*23 -> 391
facts -> The capital of Australia is Canberra and the chemical symbol for gold is Au.
```

Spot checks at temperature 0 on the QAT build. They show the 4-bit kernel producing correct text on this GPU; they are no substitute for an accuracy evaluation.

## Why a Single Stream Tops Out Where It Does

Generating one token reads every weight the text path uses. The per-layer embedding table is a row lookup and the vision and audio towers sit idle, so what gets read is the decoder layers plus the tied embedding that serves as the output layer. From the safetensors headers:

```
== models--google--gemma-4-E2B-it
  bytes read per decode token: 4.597 GB
  bound @ 277.0 GB/s measured stream:   60.3 tok/s
== models--google--gemma-4-E2B-it-qat-w4a16-ct
  bytes read per decode token: 1.862 GB
  bound @ 277.0 GB/s measured stream:  148.8 tok/s
```

277.0 GB/s is the streaming read measured on this T4 part in an earlier run in the same repository. Dividing it by the bytes per token gives each build's ceiling. The QAT build reads 2.47x fewer bytes per token.

## The Power Cap

```
100 %, 1290 MHz, 70.99 W, 0x0000000000000004
100 %, 1290 MHz, 69.97 W, 0x0000000000000004
100 %, 1275 MHz, 70.05 W, 0x0000000000000004
```

During a single-stream bf16 decode the T4 sits at 100% utilization and at its 70 W limit, with the software power cap throttling the SM clock to 81% of its 1590 MHz maximum. The driver allows no higher limit on this card. The host CPU stays light: the engine uses 18-20% of one core.

## The Sweep

`vllm bench serve` with random prompts, 128 output tokens, input lengths of 512 and 4096, concurrency 1, 4, 8 and 16, and three repeats per cell. Each cell and repeat gets its own seed, and both builds get the same seeds, so they see the same prompts. The server flags are identical for both. Sixteen cells, 48 runs, no failed requests, and the largest run-to-run variation in any cell is 2.0%.

`--max-num-seqs 8` caps the engine at eight running sequences, so at concurrency 16 the extra requests queue.

## 512-Token Prompts

| c | bf16 tok/s | QAT tok/s | bf16 per stream | QAT per stream |
|---:|---:|---:|---:|---:|
| 1 | 37.04 | 🥇 62.27 | 40.44 | 🥇 72.31 |
| 4 | 109.05 | 🥇 157.49 | 32.25 | 🥇 54.50 |
| 8 | 164.62 | 🥇 215.91 | 25.49 | 🥇 37.83 |
| 16 | 164.36 | 🥇 213.79 | 24.97 | 🥇 34.69 |

Per stream is 1000 divided by the median time per output token. QAT wins every cell: **1.79x per stream at c=1** and **1.31x total at c=8**. Both builds run on the same card at the same power cap. The QAT build moves fewer bytes per token.

Against the bandwidth ceilings above, bf16 reaches 67% of its 60.3 tok/s and QAT 49% of its 148.8. QAT leaves more headroom, which fits a card that is also out of power.

## 4096-Token Prompts

| c | bf16 tok/s | QAT tok/s | bf16 TTFT | QAT TTFT |
|---:|---:|---:|---:|---:|
| 1 | 12.11 | 14.0 | 7,199 ms | 7,243 ms |
| 4 | 15.16 | 15.82 | 15,304 ms | 15,391 ms |
| 8 | 15.66 | 15.95 | 16,779 ms | 16,879 ms |
| 16 | 15.46 | 15.75 | 81,551 ms | 80,119 ms |

With long prompts the two builds converge at about 15-16 tok/s. Time to first token sets the rate here, and it is the same for both: processing the prompt costs the same whichever way the weights are stored.

## Prefill Is the T4's Weak Point

Time to first token at c=1 grows from 300 ms at 512 tokens to 7,199 ms at 4096 on bf16: **24x the time for 8x the tokens**. Growth that steep points at attention, whose cost rises with the square of the prompt. The likely suspect is the Turing tile clamp in the 512-wide global layers, which trades speed for fitting in 64 KiB. That is an inference; no profile has confirmed it yet.

The engine's prefix cache logged hits during the sweep, averaging about 8% even with a unique seed per cell. Both builds saw identical prompts, so the comparison between them holds, but the absolute prefill times may read slightly faster than a cold T4 delivers.

## Memory Never Binds

At c=8 with 4096-token prompts, eight sequences need 33,792 KV tokens. The bf16 pool holds 315,974, which is 9.4x that. At this serving shape GPU memory has room to spare with either build; QAT's 1.64x larger pool starts to matter only when the engine is allowed more concurrent sequences.

## Against an EC2 Twin

The same repository has an EC2 `g4dn` rig on the same T4 part. It measured bf16 at 512-token prompts at 42.36 tok/s at c=1 and 242.47 tok/s at c=8. That run used vLLM 0.28.0 in a docker image, its own benchmark script with English filler prompts, and a 4 vCPU host. Its c=8 figure is 1.47x this run's bf16, and with four variables changed at once the cause of that gap is unknown.

## What Stops the Meter

```
✅ Sent SIGTERM to vLLM (pid 18703). VRAM is released on exit.
```

`stop_vllm_server` frees the GPU. The VM and its attached T4 bill by the hour whether vLLM runs or not; only stopping the VM stops the charge.

## Summary

The goal of this article was to serve Gemma 4 E2B on a Tesla T4 attached to a Compute Engine VM and find which checkpoint the card runs fastest. The key to the solution was measuring decode as a memory-bandwidth problem: the QAT w4a16 build reads 1.862 GB per token against bf16's 4.597, and vLLM 0.29's Marlin kernel runs it on Turing. The measured results were:

- **72.31 tok/s** per stream for QAT against **40.44** for bf16 at 512-token prompts, 1.79x
- **215.91 tok/s** total at 8 streams for QAT against **164.62**, 1.31x
- About **15-16 tok/s** for both builds at 4096-token prompts, where prompt processing sets the rate
- **519,568** KV tokens for QAT against **315,974** for bf16; neither pool binds at eight sequences
- The published cu130 torch wheel carries `sm_75`, and the 7.8 GB host needs swap to load the weights

One Tesla T4 on one `n1-standard-2` VM in `us-west2-b`, vLLM 0.29.0, three repeats per cell with the same seeds for both builds, largest variation in any cell 2.0%. The prefix cache logged hits despite unique seeds, which affects absolute prefill times but applies equally to both builds. The QAT build was spot-checked for correct output and was not evaluated for accuracy.

The strategy for using MCP for Tesla T4 deployment and benchmarking was validated with an incremental step by step approach.
