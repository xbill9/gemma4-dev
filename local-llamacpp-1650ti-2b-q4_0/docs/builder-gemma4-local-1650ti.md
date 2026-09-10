# Gemma 4 on a 4 GiB Laptop GPU: llama.cpp, a QAT GGUF, and an MCP Server

*Subtitle: Serving Gemma 4 E2B from a GTX 1650 Ti Max-Q with no tensor cores, and moving its MCP server to the Python SDK 2.x.*

This article provides a step by step deployment guide for Gemma 4 E2B to a local, laptop hosted GPU enabled system — a GTX 1650 Ti Max-Q with 4 GiB of memory and no tensor cores. A suite of Python MCP tools is built to simplify management of the llama.cpp hosted deployment.

https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-1650ti-2b-q4_0

---

#### Why Run Gemma 4 on This Card?

Every other rig in this repository serves Gemma 4 from rented hardware: Cloud TPU, Compute Engine, EC2, Cloud Run. Those rigs spend most of their code on a control plane — finding a zone with capacity, waiting on a queued resource, discovering an endpoint, tearing it all down so nothing stays billed.

This one has no control plane at all. The GPU is in the laptop. Four things follow, and each is a place where cloud-rig habits are actively wrong here:

- **There is no capacity to find.** No zone scan, no quota request, no flex-start wait.
- **The endpoint is not discovered.** It is `127.0.0.1:8080`, known before the process starts.
- **Memory is a hard ceiling.** 4096 MiB. There is no bigger machine type to ask for.
- **Nothing is billed.** Ctrl-C is a complete teardown.

What is left is the half that is the same everywhere: start the model server, check it, ask it something, and measure it honestly.

---

#### At This Point You Should Have…

- An NVIDIA GPU with a CUDA driver — here a GTX 1650 Ti with Max-Q Design, driver 615.71.09
- The CUDA toolkit — `nvcc` 13.3 here
- Python 3.10 or newer for the MCP server — 3.14.7 here, the system `python3`, no virtualenv
- The repository cloned, and `local-llamacpp-1650ti-2b-q4_0/` as your working directory
- Claude Code, or any MCP client that speaks stdio

---

#### What Is the Hardware?

Ask the driver rather than the spec sheet:

```shell
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used,memory.free,driver_version --format=csv,noheader
```

```plaintext
NVIDIA GeForce GTX 1650 Ti with Max-Q Design, 7.5, 4096 MiB, 1606 MiB, 2127 MiB, 615.71.09
```

(That snapshot was taken with the model already loaded. Idle, the card reports 3724 MiB free.)

**Compute capability 7.5 is a trap.** The T4 is also 7.5, and a T4 has tensor cores. The T4 is Turing's TU104 die; the GTX 16-series is TU116/TU117, and Nvidia cut the tensor cores from that silicon. Same compute capability, different chip. llama.cpp notices on its own, at init:

```plaintext
The following devices will have suboptimal performance due to a lack of tensor cores:
  Device 0: NVIDIA GeForce GTX 1650 Ti with Max-Q Design
```

So a throughput number from this card is not comparable to a T4 number, in either direction. It is also a Max-Q part — power-capped at 40 W — and it throttles. Keep that in mind for every benchmark below.

---

#### Which Checkpoint?

`google/gemma-4-E2B-it-qat-q4_0-gguf` — Google's quantization-aware-trained Gemma 4 E2B, published as a GGUF that llama.cpp opens directly. One file:

```shell
ls -l ~/models/gemma-4-E2B-it-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
```

```plaintext
-rw-rw-r-- 1 xbill xbill 3349516256 Sep  3 13:19 /home/xbill/models/gemma-4-E2B-it-qat-q4_0/gemma-4-E2B_q4_0-it.gguf
```

3.35 GB on disk. Hold that number; the next section is about why it is the wrong one to plan with.

---

#### Will It Fit? The Obvious Arithmetic Is Wrong

A 3.35 GB file against 3724 MiB free. Read those side by side and you conclude that full GPU offload barely fits, and you start lowering `-ngl` and capping the context "to be safe."

**That conclusion is wrong.** The repo ships a small script that reads tensor metadata out of the GGUF rather than guessing from the file size:

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
```

**One tensor is 58% of the file, and none of it has to be on the GPU.**

---

#### The Lazy Tensor Is the E in E2B

`per_layer_token_embd` is Gemma 4's per-layer embedding table. Its leading dimension is 8960, which is 256 per-layer input dimensions × 35 layers, over a 262,144-token vocabulary. It is why E2B has about 5B parameters in total but only 2B "effective" ones.

llama.cpp creates it with `TENSOR_READ_LAZY` in `src/models/gemma4.cpp`, documented in `src/llama-model-loader.h` as *"read rows on demand instead of loading whole tensor; requires mmap for now."* It is used as a row lookup (`GET_ROWS`), not a matmul, so the rows a token needs are read out of the memory-mapped file on the host.

Two rules fall out of it:

- **Do not lower `-ngl` to save memory.** Full offload leaves over 2 GiB free. Lowering it moves real matmul weights to the CPU and saves nothing that was scarce.
- **Never pass `--no-mmap`.** Lazy reading *requires* mmap. Turn it off and the 1.93 GB tensor has to be materialised, and a comfortable fit becomes a failure to load.

---

#### "q4_0" Is Mostly Not Q4_0

The same script breaks the file down by tensor type:

```plaintext
  by tensor type (the slot-5 token is q4_0; the file mostly is not):
    Q6_K      2.257 GB  (67.7%)
    Q4_0      1.048 GB  (31.4%)
    F16       0.028 GB  ( 0.8%)
```

Both embedding tables are Q6_K. Only the transformer blocks are Q4_0. Two consequences:

- **Further quantization has little left to take**, and it would take it from the embeddings first. There is no memory problem to solve, so there is no reason to pay for one.
- **A benchmark from this rig is not a clean "q4_0" datapoint.** Record the per-type split beside the number.

---

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

llama.cpp's init message suggests a different build for tensor-core-less Turing: `GGML_CUDA_FORCE_MMQ`, to force its integer matmul kernels. That was built into a separate directory and measured. It is covered under tuning below — it does not pay.

---

#### Step 2 — Configure `tpu.env`

One committed env file is the source of truth for the `Makefile` and the MCP server alike. (It is called `tpu.env` for consistency with its siblings. There is no TPU here.) The serving block:

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

Every one of those values was measured rather than copied from a sibling rig. `f16` KV and `-fa 1` in particular are the opposite of what a cloud-rig habit would pick, for reasons in the tuning section.

---

#### Step 3 — Serve and Measure the Memory

```shell
make serve
```

`make serve` runs `llama-server` in the foreground on purpose: Ctrl-C costs nothing, so the backgrounding and pid tracking a cloud rig needs would be ceremony. Then, from another shell:

```shell
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

```plaintext
37073, /home/xbill/llama.cpp/build/bin/llama-server, 1618 MiB
```

**1618 MiB of 4096.** Had the lazy tensor been resident, the requirement would have been about 3.5 GB and the model would have failed to load. It served instead.

---

####  Tip: A Matching Total Does Not Confirm the Terms

The derivation written down *before* that run predicted 1616 MiB — two MiB off. It looked like a clean confirmation:

```plaintext
1342 MiB  resident weights
+ 144 MiB  KV, 8192 tokens x 18 KiB/token f16
+ ~130 MiB  CUDA context + compute buffers
= 1616 MiB
```

A day later a sibling rig ran the same engine on the same card through Ollama, whose daemon prints its allocations. The real split:

| term | MiB |
| :--- | ---: |
| model buffer (weights) | 1341.78 |
| KV, 3 full-attention layers × 8192 cells | 48.00 |
| KV, 12 sliding-window layers × 1024 cells | 12.00 |
| compute buffer | 122.52 |
| CUDA context + slack | ~94 |

**The weights term was exact, and both of the others were wrong.** llama.cpp caps the sliding-window layers' KV at the 1024-token window whatever the context size, so KV is 60 MiB, not 144. Compute plus context is ~216 MiB (arithmetic), not ~130. Two errors of opposite sign landed two MiB from the measurement.

That is the most dangerous shape an arithmetic can have: agreeing with the hardware reads as confirmation of every term. Read the engine's allocation log; do not derive the terms and then check only the total.

---

#### The MCP Server

`server.py` is a single-file MCP server exposing seven tools:

| Tool | What it does |
| :--- | :--- |
| `gpu_status` | `nvidia-smi`, plus a warning that sm_75 here means no tensor cores |
| `model_info` | the checkpoint and its resident-vs-lazy split |
| `start_model_server` | launches `llama-server` detached, with the `tpu.env` flags |
| `stop_model_server` | SIGTERM; VRAM is released on exit |
| `model_server_status` | `/health` decides; the pid only annotates |
| `query_model` | a chat completion, with Gemma 4's reasoning handled explicitly |
| `get_help` | the tool list, generated from the server itself |

There is no `find_*` tool, no queued resource and no zone skip list. The server is registered under the directory name, `local-llamacpp-1650ti-2b-q4_0`, so every tool arrives in Claude Code as `mcp__local-llamacpp-1650ti-2b-q4_0__gpu_status` and cannot be confused with a cloud rig's.

**`model_server_status` finds a server it did not start.** `make serve` writes no pid file, so an early version reported  against a healthy server — and `stop_model_server` answered " Not running." while the process held the card. Both now look up which process owns the listening socket on port 8080 in `/proc/net/tcp`, and `/health` makes the final call.

---

#### What Broke: MCP SDK 2.x

Nothing in this rig changed. The shared Python did.

Another project on the same machine needed the MCP Python SDK 2.x, and these projects deliberately install into one system `python3` with no virtualenvs — the deployment images use system site-packages, so a venv would diverge dev from what runs. When `mcp` 2.2.0 landed, this rig's server stopped loading:

```shell
python3 -c "from mcp.server.fastmcp import FastMCP"
```

```plaintext
    raise ModuleNotFoundError(_MESSAGE, name=__name__)
ModuleNotFoundError: No module named 'mcp.server.fastmcp'. This is mcp 2.x, where FastMCP was renamed to MCPServer (from mcp.server.mcpserver import MCPServer) and other APIs changed; see the migration guide at https://py.sdk.modelcontextprotocol.io/v2/migration/#fastmcp-renamed-to-mcpserver or pin 'mcp<2' to keep running v1 code.
```

The process dies on the import, before the stdio handshake, so an MCP client sees only a server that failed to connect.

**Pinning back was not available.** This rig's `requirements.txt` already said `mcp>=1.2.0,<2`. But with one shared interpreter, a pin in one project is a downgrade for every project on the machine, and the project that pulled 2.x in requires `mcp>=2`. With no virtualenv, every project on the interpreter has to be on the same major version. So: migrate.

---

#### Step 4 — The Migration

Measure the exposure first. The migration guide's change list, checked against this server:

| 2.x change | This server |
| :--- | :--- |
| `FastMCP` renamed to `MCPServer` |  hit — the only change needed |
| `mcp` no longer installs `httpx` |  exposed; `httpx` was already its own requirement |
| sync `def` handlers move to a worker thread |  every tool is `async def` |
| server `version` reported as empty |  cosmetic, visible in the handshake below |

The code change is two lines:

```diff
-from mcp.server.fastmcp import FastMCP
+from mcp.server.mcpserver import MCPServer
 ...
-mcp = FastMCP(MCP_SERVER_NAME)
+mcp = MCPServer(MCP_SERVER_NAME)
```

The requirement flips its bound:

```diff
-mcp>=1.2.0,<2
+mcp>=2
```

And one change the guide cannot tell you about: **the tests mock the module path.** They replace `mcp` in `sys.modules` before importing `server`, so the fake has to move with the import:

```diff
-sys.modules["mcp.server.fastmcp"] = _fastmcp_module
+sys.modules["mcp.server.mcpserver"] = _mcpserver_module
```

Miss that and the suite fails to import `server` at all, which reads as the migration failing when it is the fake that is wrong.

---

#### The Drift the Migration Exposed

Touching `start_model_server` for the rename surfaced something older. The `Makefile` passed the measured flags; the MCP tool did not. It launched `llama-server` without `-fa`, `-t` or `--parallel`, so a server started through MCP came up with llama.cpp's defaults — **4 slots and 6 threads** — instead of the configuration every benchmark was measured on.

The slots matter more than they look. llama.cpp **splits `--ctx-size` across `--parallel` slots**, so four default slots quietly quarter the context each request can use.

The fix pulls the argv into one function, `_server_command()`, and a test holds it to the `Makefile`:

```python
def test_same_flag_set(self):
    with patch.object(server, "METRICS", "1"):
        cmd = server._server_command()
    self.assertEqual({t for t in cmd if t.startswith("-")}, self._makefile_flags())
```

The server running while this article was written was started through `start_model_server`. Its real command line:

```shell
tr '\0' ' ' < /proc/$(pidof llama-server)/cmdline
```

```plaintext
/home/xbill/llama.cpp/build/bin/llama-server -m /home/xbill/models/gemma-4-E2B-it-qat-q4_0/gemma-4-E2B_q4_0-it.gguf --host 127.0.0.1 --port 8080 -ngl 99 -c 8192 -ctk f16 -ctv f16 -fa 1 -t 4 --parallel 1 --metrics
```

---

#### Step 5 — Lint and Test

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

The suite is offline: it mocks `mcp`, the subprocess boundary and HTTP, and never touches the GPU.

---

#### Step 6 — Test the Protocol by Hand

Unit tests call Python; a client speaks JSON-RPC over stdio. Test that too, holding stdin open with `sleep` so the server does not see end-of-input and exit after `initialize`:

```shell
{ printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'; sleep 4; } \
  | python3 server.py 2>/dev/null
```

The responses are JSON; summarised:

```plaintext
initialize OK: name='local-llamacpp-1650ti-2b-q4_0' version='' proto 2025-06-18
tools/list OK: 7 tools -> get_help, gpu_status, model_info, model_server_status, query_model, start_model_server, stop_model_server
```

 Seven tools on `mcp` 2.2.0. `version=''` is the 2.x change from the table: an unversioned server no longer borrows the SDK's version number.

---

#### Step 7 — Validate: Ask the Model

Through `query_model`, or directly with `make query`:

```plaintext
finish_reason: stop
content: 'A TPU (Tensor Processing Unit) is a custom-built hardware accelerator designed by Google to dramatically speed up the complex matrix operations required for training and running deep learning models.'
reasoning_content chars: 1257
usage: {'completion_tokens': 293, 'prompt_tokens': 25, ...}
```

A one-sentence answer cost 293 completion tokens. Most of them were spent thinking.

---

####  Tip: The Second Way to Get an Empty Reply

The usual warning about `-it` checkpoints is that raw `/v1/completions` returns nothing, so use `/v1/chat/completions`. On Gemma 4 there is a second, unrelated empty-reply path, and it hits the endpoint you were told to use instead.

Gemma 4 emits a thinking block. llama.cpp routes it to `reasoning_content` and leaves `content` empty until the block closes. Same prompt, two budgets:

```plaintext
max_tokens=64    finish_reason: length | content: ''       | reasoning_content chars: 316  | completion_tokens: 64
max_tokens=1024  finish_reason: stop   | content: 'Here are three generations of ...' | reasoning_content chars: 1241 | completion_tokens: 400
```

A caller that reads only `content` with a modest budget gets `""` and concludes the server is broken. So `query_model` defaults to `max_tokens=1024`, a test asserts it stays at 512 or more, and when `content` is empty but `reasoning_content` is not, the tool returns  *"Reasoning only — no answer yet"* instead of a  with an empty body.

The same applies to benchmarks: **a 128-token generation limit on this model measures the thinking phase and nothing else.** The tok/s is real; the task is fictional. Every sweep below is labelled accordingly.

---

#### Tuning: What Moved Decode?

`llama-bench -p 512 -n 128 -r 3`, all 35 layers on the GPU:

| config | prefill t/s | decode t/s | |
| :--- | ---: | ---: | :---: |
| `-fa 1`, f16 KV, `-t 4` | 340.33 | 73.75 |  |
| `-fa 1`, f16 KV, `-t 6` | 340.39 | 73.74 |  |
| `-fa 1`, f16 KV, `-t 8` | 338.49 | 72.97 |  |
| `-fa 0`, f16 KV, `-t 6` | 338.74 | 70.39 | |
| `-fa 1`, f16 K, q8_0 V | 202.66 | 65.55 | |
| `-fa 1`, q8_0 K, f16 V | 239.09 | 64.91 | |

Four findings, and two of them are traps:

- **Flash attention is a free +4.8% on decode** (70.39 → 73.74) and does nothing to prefill. Adopted.
- **Thread count does not matter.** Decode is GPU-bound; `-t 4` leaves cores free.
- **KV quantization is a loss, and a large one.** `q8_0` on either K or V costs 11–12% of decode and 30–40% of prefill. It is a win on the TPU rigs in this repo. Here there are no tensor cores to hide the dequantization, and E2B's KV is tiny — it buys memory that was never scarce.
- **`GGML_CUDA_FORCE_MMQ` does not pay**, even though llama.cpp suggests it for this card. Stock build 340.33 / 73.75; forced-MMQ build 337.61 / 73.04. Decode at batch 1 is bandwidth- and launch-bound, not matmul-bound.

Both traps would be adopted by anyone tuning from another rig's results instead of measuring on this one.

---

#### Benchmark Sweep: Does Concurrency Help?

`llama-batched-bench` says yes, emphatically. Aggregate decode, 128-token prompts:

| parallel sequences | decode t/s | per stream |
| ---: | ---: | ---: |
| 1 | 73.33 | 73.33 |
| 8 | 223.75 | 27.97 |
| 64 | 277.61 | 4.34 |

3.8x. But that tool reports the decode phase *in isolation*. A real serving sweep through the HTTP endpoint — 512-token prompts, 128-token outputs, three repeats per level, prompt cache defeated — tells a different story:

| clients | aggregate tok/s | TTFT ms | per stream |
| ---: | ---: | ---: | ---: |
| 1 | 32.74 | 2135 | 69.89 |
| 2 | 40.59 | 4062 | 55.27 |
| 4 | 45.12 | 7982 | 36.86 |
| 8 | 44.79 | 16212 | 18.74 |
| 16 | 45.89 | 32551 | 10.31 |
| 32 | 48.19 | 61825 | 5.42 |

**Time to first token doubles exactly with every doubling of clients.** That is prefill running strictly one request at a time. Aggregate output saturates near 45–48 tok/s, 1.47x from one client, and past four clients more concurrency buys only latency.

The two results do not conflict. Decode really does batch. But at 512 in and 128 out, prefill is four times the token volume of decode, so the phase that scales is the minority of the work. **Never quote this card's "277 tok/s" without naming the phase.**

---

#### Short Prompts, Long Answers: It Batches, Sometimes

The obvious next shape is the one where decode dominates: 128 in, 512 out, card cooled to 52 °C before every level.

| clients | median tok/s | trials | TPOT ms |
| ---: | ---: | :--- | :--- |
| 1 | 63.77 | 62.9 / 63.8 / 63.8 | 14.43 / 14.48 / 14.49 |
| 2 | 62.58 | **62.3, 62.6 / 97.6** | **17.93** / 29.63 / 29.72 |
| 4 | 128.19 | **94.5 / 128.2, 129.0** | **26.58** / 37.91 / **26.91** |
| 8 | 109.91 | **109.6, 109.9 / 127.8** | 63.74 / **53.86** / 64.14 |
| 16 | 164.07 | 164.0 / 164.1 / 164.4 | 79.97 / 80.10 / 80.05 |

The prediction held: **2.6x from one client to sixteen**, against 1.47x at the prefill-heavy shape. But only 1 and 16 reproduce. Everything between is **bimodal**.

At two clients the slow mode decodes at 29.6 ms per token — exactly twice the single-stream 14.4. Two streams taking turns instead of sharing a batch. Time to first token is constant within every level; only the decode rate jumps.

**It is not thermal**, although that is the reading everyone will reach for on a power-capped Max-Q part (≈39 W of 40 W, clocks 1530–1770 MHz against 2100). Three things rule it out:

- **16 clients is the hottest level, and the fastest and most stable.** Thermal decay cannot peak in the middle.
- **Fast and slow trials occur at the same temperature and clock.**
- **The slow mode reproduces to 0.2% across independent passes.** Throttling is continuous; this is two discrete states.

So this rig quotes no mid-range concurrency number at any shape, and a median of three samples is the wrong statistic for a bimodal cell — it reports whichever mode won two tosses.

---

#### And Context Length?

One client, output capped at 128 tokens, input stepped up 46x (arithmetic):

| input tokens | TTFT ms | decode tok/s | end-to-end tok/s |
| ---: | ---: | ---: | ---: |
| 108 | 393 | 72.55 | 59.72 |
| 661 | 2158 | 70.77 | 32.39 |
| 1279 | 4147 | 70.27 | 21.50 |
| 2527 | 8519 | 68.64 | 12.34 |
| 5027 | 17798 | 66.80 | 6.50 |

**Decode barely notices context: −7.9% (arithmetic) across the whole range.** End-to-end falls 9x (arithmetic), entirely from prefill. On this card the prompt length is the lever for responsiveness; the generation length is not.

(These decode figures are tokens over the measured inter-token span, recomputed from the run's own JSON. The harness's as-run column counted stream *chunks*, and llama-server does not send exactly one chunk per token — 1.024 tokens per chunk at this output length — so it read 2.3–2.4% low.)

---

#### What Did Not Transfer From the Cloud Rigs?

| habit | on the cloud rigs | on this card |
| :--- | :--- | :--- |
| size from the file on disk | roughly right | wrong by 2.4x (arithmetic) — the lazy tensor |
| lower offload to fit | sometimes needed | buys nothing; never needed |
| quantize the KV cache | a win on TPU | −11–12% decode |
| engine's suggested kernel | usually right | `FORCE_MMQ` a wash |
| batch for throughput | scales | only decode scales; prefill is serial |
| discover the endpoint | required | a literal |

Same model, same repository. The hardware is the variable, and it changed the answer in most rows.

---

#### And Price/Performance?

There is no price. Nothing is billed, nothing is reserved, and there is no idle cost to optimise. The honest comparison is not dollars but what the card cannot do: it will not batch prefill, it throttles, and it tops out at 4 GiB forever.

For one interactive session — the normal use of a laptop — 73.75 tok/s single-stream decode in 1618 MiB is plenty.

---

#### Teardown

```shell
# make serve is in the foreground:
Ctrl-C
```

Or ask the agent for `stop_model_server`, which sends SIGTERM to whichever process owns port 8080 — including one started by `make serve` — and VRAM is released on exit. Not run for this article; the server is still up.

That is the whole of it. There is no queued resource to delete and nothing to forget.

---

#### Cheat Sheet

```shell
# what has to be resident (not the file size)
make info

# serve with the measured flags, in the foreground
make serve
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

# mcp 2.x migration, in ONE edit
#   from mcp.server.fastmcp import FastMCP  ->  from mcp.server.mcpserver import MCPServer
#   FastMCP(name)                           ->  MCPServer(name)
#   requirements.txt: mcp>=2   tests: sys.modules["mcp.server.mcpserver"]
make lint && make test

# stdio smoke test: hold stdin open
{ printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"p","version":"0"}}}' \
                '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
                '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'; sleep 4; } | python3 server.py 2>/dev/null

# a reply, with enough budget to get past the thinking block
make query

# never: --no-mmap, a lower -ngl "to be safe", q8_0 KV
```

---

#### Summary

The goal of this article was to serve Gemma 4 from a 4 GiB laptop GPU with no tensor cores, and to manage it with the same kind of MCP agent the cloud rigs use. The key to the solution was reading the artifact and the engine instead of the file size and the spec sheet: 58% of the checkpoint never leaves the host, so full offload fits with room to spare. The results were:

-  Full offload in 1618 MiB of 4096 — the per-layer embedding table stays in the mmap
-  73.75 tok/s single-stream decode with `-fa 1`, f16 KV
-  Two tuning traps: `q8_0` KV costs 11–12% of decode, and `FORCE_MMQ` gains nothing
-  Concurrency scales decode, not prefill: 1.47x at 512 in / 128 out, 2.6x at 128 in / 512 out, bimodal in between
-  The MCP server moved to SDK 2.x with a two-line change, forced by a shared interpreter, and the move exposed an MCP-started server running a configuration no benchmark measured

Scope: one GTX 1650 Ti with Max-Q Design (TU117, compute capability 7.5, 4096 MiB, 40 W cap) in one laptop, llama.cpp `95ef7fc` built for sm_75 with CUDA 13.3, `google/gemma-4-E2B-it-qat-q4_0-gguf`. `llama-bench` rows are three repeats; serving sweeps are three repeats per level with the prompt cache defeated and verified so via `/metrics`. Every generation in the sweeps hit its token cap inside Gemma 4's thinking block, so they measure throughput, not answers. The memory split comes from a sibling rig running the same engine through Ollama on the same card.

The strategy for using MCP for a local GPU deployment was validated with an incremental step by step approach.

#### References

* [local-llamacpp-1650ti-2b-q4_0 | GitHub](https://github.com/xbill9/gemma4-dev/tree/main/local-llamacpp-1650ti-2b-q4_0)
* [Migration Guide: v1 to v2 | MCP Python SDK](https://py.sdk.modelcontextprotocol.io/v2/migration/)
* [llama.cpp | GitHub](https://github.com/ggml-org/llama.cpp)
* [google/gemma-4-E2B-it-qat-q4_0-gguf | Hugging Face](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)

---

*llama.cpp `95ef7fc` (CUDA 13.3), mcp 2.2.0, Python 3.14.7, ruff 0.16.6, NVIDIA driver 615.71.09, one GTX 1650 Ti Max-Q.*
