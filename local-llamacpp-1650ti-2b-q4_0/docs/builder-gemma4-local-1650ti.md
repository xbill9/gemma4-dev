# Gemma 4 on an Old 4 GB Laptop GPU: QAT Takes It From 9.5 GiB to 1.6

*Subtitle: Running Google's quantization-aware-trained Gemma 4 E2B on a 10th-gen Core i7 laptop with a 4 GB GTX 1650 Ti, managed by an MCP server.*

This article provides a step by step deployment guide for Gemma 4 E2B's quantization-aware-trained (QAT) checkpoint to a local, laptop hosted GPU enabled system — a much older Lenovo Yoga 9, on sale since January 2021, with a 4 GB GTX 1650 Ti. A suite of Python MCP tools is built to simplify management of the llama.cpp hosted deployment.

https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-1650ti-2b-q4_0

Gemma 4 E2B in bfloat16 is 9.5 GiB of weights. This laptop's GPU has 4 GiB. Even a plain int8 conversion would not fit. **The QAT release closes that gap: the same model, trained knowing it would be stored at 4 bits, in a 3.35 GB file of which only 1.31 GiB ever has to be on the GPU.** It loaded in 1618 MiB, left more than half the card free, and decodes at 73.75 tok/s.

#### What is this project trying to Do?

Every other rig in this repository serves Gemma 4 from rented hardware: Cloud TPU, Compute Engine, EC2, Cloud Run. I wanted to know whether the same model would run on the laptop I already had — a 10th-generation Core i7 with a GTX 16-series GPU, no tensor cores and a 4 GB ceiling that no quota request can raise.

It does, and QAT is the reason. The rest of this article is how, and what the old hardware changes about running it.

#### At This Point You Should Have…

- An NVIDIA GPU with a CUDA driver — here a GTX 1650 Ti with Max-Q Design, driver 615.71.09
- The CUDA toolkit — `nvcc` 13.3 here
- Python 3.10 or newer for the MCP server — 3.14.7 here, the system `python3`, no virtualenv
- The repository cloned, and `local-llamacpp-1650ti-2b-q4_0/` as your working directory
- Claude Code, or any MCP client that speaks stdio

#### The Laptop

| | |
| :--- | :--- |
| Machine | Lenovo Yoga 9 15IMH5 |
| On sale | since January 2021, when Windows Central reviewed it |
| CPU | Intel Core i7-10750H, 10th generation (Comet Lake), 12 threads |
| RAM | 15 GiB, as `free` reports it |
| GPU | GeForce GTX 1650 Ti with Max-Q Design, 4096 MiB |
| GPU power | 40 W limit |

The MCP server reports the GPU the same way:

```plaintext
gpu_status
```

```plaintext
 **GPU** — `local-llamacpp-1650ti-2b-q4_0`

NVIDIA GeForce GTX 1650 Ti with Max-Q Design, 7.5, 4096 MiB, 1606 MiB, 2127 MiB, 615.71.09

  GTX 16-series (TU116/TU117): compute capability 7.5 but **no tensor cores**. Do not compare throughput against the T4-based `g4dn`/`g5g` rigs on the strength of a matching compute capability.
```

**Compute capability 7.5 is a trap.** A T4 is also 7.5 and has tensor cores; the GTX 16-series is a cut-down Turing die with the tensor cores removed. llama.cpp notices on its own at init:

```plaintext
The following devices will have suboptimal performance due to a lack of tensor cores:
  Device 0: NVIDIA GeForce GTX 1650 Ti with Max-Q Design
```

#### Why Can't the Normal Model Fit?

The sizes, from this repository's model reference:

| Gemma 4 E2B as | Weights | Fits 4096 MiB? |
| :--- | ---: | :---: |
| bfloat16 | 9.5 GiB |  no |
| int8 | ~4.8 GiB |  no |
| QAT Q4_0 GGUF, whole file | 3.35 GB | barely, on paper |
| QAT Q4_0 GGUF, GPU-resident part | **1.31 GiB** |  yes |

The bf16 checkpoint is more than twice the card. Halving it to int8 is still larger than the card. Only a 4-bit model is in range, and that is where QAT comes in.

#### What Is QAT, and Why Does It Matter Here?

Quantization-aware training simulates quantization *during* training rather than compressing a finished model afterwards. The 4-bit weights are not a post-hoc approximation of a bf16 model: the model was trained knowing it would be stored this way. Google's model card puts the goal plainly:

> This model card is for the new versions of the Gemma 4 family optimized with Quantization-Aware Training (QAT), which allows preserving similar quality to bfloat16 while dramatically reducing the memory requirements to load the model.

That is the difference that matters on a 4 GB card. **Rounding the bf16 model down to 4 bits after the fact would fit too, but nothing in its training would have prepared it for that.** QAT is the version of "fits" that was designed to keep the model's quality close to bf16 on the way there.

Google ships the QAT weights four ways:

| Artifact | What it is | For |
| :--- | :--- | :--- |
| `-qat-q4_0-unquantized` | QAT values in a half-precision container | custom compilation |
| `-qat-q4_0-gguf` | the same values, packed Q4_0 | **llama.cpp — this rig** |
| `-qat-w4a16-ct` | compressed-tensors | vLLM |
| mobile | on-device runtimes | phones |

The GGUF is not a lesser copy. This repository checked it: four norm tensors read out of the GGUF are bit-identical to the ones in the `-unquantized` release, so it is the same QAT model in its native packing.

#### Which QAT Checkpoint?

`google/gemma-4-E2B-it-qat-q4_0-gguf`, which llama.cpp opens directly. The agent reads it off disk:

```plaintext
model_info
```

```plaintext
 **Model** — `local-llamacpp-1650ti-2b-q4_0`

- **Name:** `google/gemma-4-E2B-it-qat-q4_0-gguf`
- **Path:** `/home/xbill/models/gemma-4-E2B-it-qat-q4_0/gemma-4-E2B_q4_0-it.gguf`
- **On disk:** 3.35 GB
- **Quantization slot:** `q4_0` — but the dominant tensor type is **Q6_K**. Both embedding tensors are Q6_K (2.257 GB of 3.334 GB); only the ~1.08 GB transformer body is actually Q4_0.
- **Resident on GPU:** ~1.31 GiB. `per_layer_token_embd` (1.93 GB, 58% of the file) is `TENSOR_READ_LAZY` and is served by GET_ROWS out of the mmap.

Run `inspect_gguf.py` to re-derive the split from the artifact rather than trusting these numbers.
```

Two lines in that output explain why a 3.35 GB file fits in far less than 3.35 GB.

#### Why Only 1.31 GiB Has to Be on the GPU

A 3.35 GB file against 3724 MiB free reads as "barely fits." It is better than that. `make info` reads the tensor table out of the GGUF:

```shell
make info
```

```plaintext
gemma-4-E2B_q4_0-it.gguf
  tensors:  541
  total:    3.334 GB

  largest tensors:
      1926.8 MB  per_layer_token_embd.weight  Q6_K    [8960, 262144]  <- LAZY, host-resident
       330.3 MB  token_embd.weight            Q6_K    [1536, 262144]
        27.5 MB  per_layer_model_proj.weight  F16     [1536, 8960]
        10.6 MB  blk.34.ffn_up.weight         Q4_0    [1536, 12288]

  lazy (never on GPU):        1.927 GB  (58% of file)
  must be resident:           1.407 GB = 1.31 GiB

  by tensor type (the slot-5 token is q4_0; the file mostly is not):
    Q6_K      2.257 GB  (67.7%)
    Q4_0      1.048 GB  (31.4%)
    F16       0.028 GB  ( 0.8%)
```

**One tensor is 58% of the file, and none of it has to be on the GPU.** `per_layer_token_embd` is Gemma 4's per-layer embedding table — the "E" in E2B, and why the model has about 5B parameters in total but 2B effective. llama.cpp creates it with `TENSOR_READ_LAZY` and uses it as a row lookup, not a matmul, so the rows a token needs are read out of the memory-mapped file on the host.

QAT and the lazy table stack. QAT puts the 35 transformer blocks at Q4_0, 1.05 GB where half precision would be 3.7 GB; the lazy table keeps the biggest remaining tensor off the card entirely. **What decode actually streams per token drops from 4.514 GB at half precision to 1.407 GB, a 3.2x cut.**

Two rules follow, and both are the opposite of what a memory-anxious setup would do:

- **Do not lower `-ngl` to save memory.** Full offload leaves over 2 GiB free. Lowering it moves real matmul weights to the CPU and saves nothing that was scarce.
- **Never pass `--no-mmap`.** Lazy reading requires mmap. Turn it off and the 1.93 GB table has to be materialised, and a comfortable fit becomes a failure to load.

#### Step 1 — Build llama.cpp for sm_75

A standard CUDA build, pinned to this card's architecture:

```shell
grep -E "^(GGML_CUDA|CMAKE_CUDA_ARCHITECTURES|CMAKE_BUILD_TYPE)[:=]" ~/llama.cpp/build/CMakeCache.txt
~/llama.cpp/build/bin/llama-server --version
```

```plaintext
CMAKE_BUILD_TYPE:STRING=Release
CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=75
GGML_CUDA:BOOL=ON
version: 0.3.0-dev (build 1, commit 95ef7fc)
```

#### Step 2 — Configure `tpu.env`

One committed env file drives both the `Makefile` and the MCP server. It is called `tpu.env` for consistency with its siblings; there is no TPU here.

```shell
grep -E "^(MODEL_PATH|ENDPOINT|N_GPU_LAYERS|CONTEXT_SIZE|KV_CACHE_TYPE|FLASH_ATTENTION|THREADS|PARALLEL_SLOTS|METRICS)=" tpu.env
```

```plaintext
MODEL_PATH=/home/xbill/models/gemma-4-E2B-it-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
ENDPOINT=http://127.0.0.1:8080
N_GPU_LAYERS=99
CONTEXT_SIZE=8192
KV_CACHE_TYPE=f16
FLASH_ATTENTION=1
METRICS=1
THREADS=4
PARALLEL_SLOTS=1
```

`N_GPU_LAYERS=99` is full offload. Every value was measured on this card rather than copied from a cloud rig.

#### Step 3 — Serve It and Measure the Memory

```shell
make serve
```

`make serve` runs in the foreground on purpose: this is a laptop, nothing is billed, and Ctrl-C is a complete teardown. From another shell:

```shell
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

```plaintext
37073, /home/xbill/llama.cpp/build/bin/llama-server, 1618 MiB
```

**1618 MiB of 4096.** Weights, an 8192-token context and CUDA's own overhead, with more than half the card to spare. Had the lazy table been resident the requirement would have been about 3.5 GB, and the model would have failed to load.

The agent confirms it is serving:

```plaintext
model_server_status
```

```plaintext
 Serving at http://127.0.0.1:8080 (pid 83619). `/health` → 200.
```

The endpoint is a fixed local address, so there is nothing to discover. An early version of this tool read a missing pid file as "not running" and reported  against a healthy server; it now checks which process owns port 8080, and `/health` makes the final call.

#### Where the 1618 MiB Goes

The allocation, as printed by the engine when a sibling rig loaded the same file on this same card through Ollama:

| term | MiB |
| :--- | ---: |
| model buffer (weights) | 1341.78 |
| KV, 3 full-attention layers × 8192 cells | 48.00 |
| KV, 12 sliding-window layers × 1024 cells | 12.00 |
| compute buffer | 122.52 |
| CUDA context + slack | ~94 |

The weights are the QAT body plus the resident embeddings. The KV cache is only 60 MiB, because llama.cpp caps Gemma 4's sliding-window layers at the 1024-token window regardless of context size.

My own derivation before the run predicted 1616 MiB — two MiB off — with 144 MiB of KV and ~130 of overhead. **Both of those terms were wrong, in opposite directions, and the total still matched.** A total that agrees with the hardware does not confirm the terms; read the engine's allocation log.

#### Step 4 — Ask the Model

```plaintext
query_model "In one sentence, what is a TPU?"
```

```plaintext
 **Reply**

A TPU is a specialized hardware accelerator designed by Google specifically to speed up the computationally intensive operations required for training and running machine learning models and deep learning tasks.

---
_(plus 1167 chars of reasoning, suppressed)_
prompt 25 tok · completion 308 tok · 70.5 tok/s
```

A 4-bit Gemma 4 answering on a laptop GPU with no tensor cores, through Claude Code, over MCP. **A one-sentence answer cost 308 completion tokens**, and most of them were thinking.

#### The Second Way to Get an Empty Reply

Gemma 4 emits a thinking block. llama.cpp routes it to `reasoning_content` and leaves `content` empty until the block closes, so a caller that reads only `content` with a modest budget gets `""` and concludes the server is broken. `query_model` defaults to `max_tokens=1024`, and reports the empty case as what it is:

```plaintext
query_model "Name three TPU generations." max_tokens=64
```

```plaintext
 **Reasoning only — no answer yet.** `finish_reason: length` after 64 tokens, all of them thinking.

This is Gemma 4 reasoning, not a broken server. Re-run with a larger `max_tokens` (currently 64).
```

The same applies to benchmarks: **a 128-token generation limit on this model measures the thinking phase and nothing else.**

#### Tuning: What Moved Decode?

`llama-bench -p 512 -n 128 -r 3`, all layers on the GPU:

| config | prefill t/s | decode t/s |
| :--- | ---: | ---: |
| `-fa 1`, f16 KV, `-t 4` | 340.33 | 73.75 |
| `-fa 1`, f16 KV, `-t 6` | 340.39 | 73.74 |
| `-fa 1`, f16 KV, `-t 8` | 338.49 | 72.97 |
| `-fa 0`, f16 KV, `-t 6` | 338.74 | 70.39 |
| `-fa 1`, f16 K, q8_0 V | 202.66 | 65.55 |
| `-fa 1`, q8_0 K, f16 V | 239.09 | 64.91 |

- **Flash attention is a free +4.8% on decode** (70.39 → 73.74). Adopted.
- **Thread count does not matter.** Decode is GPU-bound.
- **Quantizing the KV cache is a loss.** `q8_0` costs 11–12% of decode and 30–40% of prefill. With no tensor cores to hide the dequantization, and a KV cache of 60 MiB, it buys memory that was never scarce. **QAT already did the quantizing that matters.**
- **`GGML_CUDA_FORCE_MMQ` does not pay**, even though llama.cpp suggests it for this card: 337.61 / 73.04 against the stock build's 340.33 / 73.75.

#### Does Concurrency Help?

For decode, yes: `llama-batched-bench` goes from 73.33 to 277.61 tok/s between one and 64 parallel sequences. But that is the decode phase in isolation. Through the HTTP endpoint with 512-token prompts, prefill runs one request at a time — time to first token doubles exactly with every doubling of clients — and aggregate output tops out near 48 tok/s, 1.47x from one client. With short prompts and long answers it reaches 2.6x at 16 clients, but every level between 1 and 16 flips between two speeds from trial to trial.

**For one person on one laptop, concurrency one is the right setting**, and that is what `PARALLEL_SLOTS=1` carries.

#### Managing It with MCP

`server.py` is a single-file MCP server with seven tools: `gpu_status`, `model_info`, `start_model_server`, `stop_model_server`, `model_server_status`, `query_model` and `get_help`. There is no provisioning tool, because there is nothing to provision. It registers under its directory name, so every tool reaches Claude Code as `mcp__local-llamacpp-1650ti-2b-q4_0__<tool>`.

A stdio check confirms the handshake independently of any client:

```shell
{ printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'; sleep 4; } \
  | python3 server.py 2>/dev/null
```

```plaintext
initialize OK: name='local-llamacpp-1650ti-2b-q4_0' version='' proto 2025-06-18
tools/list OK: 7 tools -> get_help, gpu_status, model_info, model_server_status, query_model, start_model_server, stop_model_server
```

#### One Thing Broke: MCP SDK 2.x

Midway through, the server stopped loading. Another project on the machine needed the MCP Python SDK 2.x, all of these projects share one system Python, and 2.x renamed the class this server imports:

```shell
python3 -c "from mcp.server.fastmcp import FastMCP"
```

```plaintext
    raise ModuleNotFoundError(_MESSAGE, name=__name__)
ModuleNotFoundError: No module named 'mcp.server.fastmcp'. This is mcp 2.x, where FastMCP was renamed to MCPServer (from mcp.server.mcpserver import MCPServer) and other APIs changed; see the migration guide at https://py.sdk.modelcontextprotocol.io/v2/migration/#fastmcp-renamed-to-mcpserver or pin 'mcp<2' to keep running v1 code.
```

Pinning back was not an option with a shared interpreter. The fix was two lines, the requirement bound, and the test mock's module path:

```diff
-from mcp.server.fastmcp import FastMCP
+from mcp.server.mcpserver import MCPServer
-mcp = FastMCP(MCP_SERVER_NAME)
+mcp = MCPServer(MCP_SERVER_NAME)

-mcp>=1.2.0,<2
+mcp>=2

-sys.modules["mcp.server.fastmcp"] = _fastmcp_module
+sys.modules["mcp.server.mcpserver"] = _mcpserver_module
```

Touching `start_model_server` for the rename surfaced an older bug: the tool launched `llama-server` without `-fa`, `-t` or `--parallel`, so an MCP-started server came up with **4 slots and 6 threads** instead of the measured configuration — and llama.cpp splits the context across slots. The argv now lives in one function, a test holds it to the `Makefile`, and the running server's real command line matches:

```shell
tr '\0' ' ' < /proc/$(pidof llama-server)/cmdline
```

```plaintext
/home/xbill/llama.cpp/build/bin/llama-server -m /home/xbill/models/gemma-4-E2B-it-qat-q4_0/gemma-4-E2B_q4_0-it.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 8192 -ctk f16 -ctv f16 -fa 1 -t 4 --parallel 1 --metrics
```

```shell
make lint
make test
```

```plaintext
All checks passed!
lint OK
----------------------------------------------------------------------
Ran 30 tests in 0.035s

OK
```

#### And Price/Performance?

There is no price. The laptop was already on the desk: nothing is billed, nothing is reserved, and there is no idle cost to optimise.

What QAT bought is capability, not a discount. Without it this GPU cannot hold Gemma 4 E2B at all. With it, the model takes 1618 MiB and a single conversation decodes at 73.75 tok/s.

#### Teardown

```shell
# make serve is in the foreground:
Ctrl-C
```

Or ask the agent for `stop_model_server`, which sends SIGTERM to whichever process owns port 8080, including one started by `make serve`. Not run for this article; the server is still up.

#### What This Does Not Cover

**Output quality was not measured here.** The claim that QAT holds quality close to bfloat16 is Google's, from the model card; this article measured memory and speed, not accuracy, and never compared the QAT model's answers against a bf16 run.

**No benchmark completed a task.** Every sweep generation hit its token cap inside Gemma 4's thinking block, so the throughput figures are real and the tasks are not.

**Nothing here transfers to a T4.** Same compute capability, different silicon.

#### Summary

The goal of this article was to run Gemma 4 on a much older laptop whose GPU has 4 GB of memory. The key to the solution was Google's quantization-aware-trained checkpoint, packed as a GGUF, plus llama.cpp leaving its largest tensor in host memory. The measured results were:

- **bf16 needs 9.5 GiB and int8 ~4.8 GiB; the QAT GGUF needs 1.31 GiB on the GPU.** Only the 4-bit QAT model fits this card, and it fits with room to spare.
- **Full offload in 1618 MiB of 4096**, with an 8192-token context and a KV cache of just 60 MiB.
- **73.75 tok/s single-stream decode** on a GPU with no tensor cores; flash attention is worth 4.8%.
- **Quantizing further is a loss:** a `q8_0` KV cache costs 11–12% of decode, because QAT already took the memory pressure away.
- **An MCP server manages the whole lifecycle**, and moving it to SDK 2.x took a two-line change that also exposed an MCP-started server running the wrong configuration.

Scope: one Lenovo Yoga 9 15IMH5 (Core i7-10750H, 15 GiB RAM, GTX 1650 Ti Max-Q with 4096 MiB and a 40 W cap), llama.cpp `95ef7fc` built for sm_75 with CUDA 13.3, `google/gemma-4-E2B-it-qat-q4_0-gguf`, and the MCP server on mcp 2.2.0 under Python 3.14.7. `llama-bench` rows are three repeats; serving sweeps are three repeats per level with the prompt cache defeated. The memory split comes from a sibling rig running the same engine through Ollama on the same card. Model quality was not measured.

The strategy for using MCP for a local GPU deployment was validated with an incremental step by step approach.

#### References

* [local-llamacpp-1650ti-2b-q4_0 | GitHub](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-1650ti-2b-q4_0)
* [google/gemma-4-E2B-it-qat-q4_0-gguf | Hugging Face](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
* [Lenovo Yoga 9i 15 review, January 2021 | Windows Central](https://www.windowscentral.com/lenovo-yoga-9i-15)
* [Quantization-Aware Training for Gemma 4 | Google](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)
* [llama.cpp | GitHub](https://github.com/ggml-org/llama.cpp)
* [Migration Guide: v1 to v2 | MCP Python SDK](https://py.sdk.modelcontextprotocol.io/v2/migration/)
