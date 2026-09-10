---
title: "2B Gemma 4 Deployment with Cloud Run, NVIDIA L4, MCP SDK 2.x, and Claude Code"
published: false
description: "Step by step deployment of Gemma 4 E2B to a Cloud Run NVIDIA L4 GPU with vLLM, managed by a Python MCP server migrated to the MCP SDK 2.x."
tags: mcp, gemma, googlecloud, claudecode
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-2B-cloudrun-devops-agent/docs/deploy/devto-cover.a4522d79.jpg
---

This article provides a step by step deployment guide for Gemma 4 E2B to a Cloud Run hosted GPU enabled system. A suite of Python MCP tools is built to simplify management of the vLLM hosted deployment with Claude Code.

https://github.com/xbill9/gemma4-dev/tree/main/gpu-2B-cloudrun-devops-agent

---

#### What is this project trying to Do?

This project is a DevOps/SRE assistant for a Gemma 4 model served by vLLM on Cloud Run with an NVIDIA L4 GPU. A single-file Python MCP server provides tools to stage the weights, deploy the service, check its health, benchmark it, and tear it down.

Cloud Run is the serverless option. There is no VM to provision and no driver to install: the service scales to zero when idle, and one `gcloud` command attaches the GPU.

Along the way the MCP server itself had to move to the MCP Python SDK 2.x, because a fresh `pip install` stopped it from starting. That migration is covered where it happened — in the MCP server section, before the deploy.

---

#### Where do I start?

The strategy for starting MCP development for model management is a incremental step by step approach.

First, the basic development environment is setup with the required system variables and a working Claude Code configuration.

Then, the Python MCP server is brought up over stdio and validated with Claude Code in the local environment. That server then stages the model, deploys it to Cloud Run, and drives validation and a benchmark sweep against the live endpoint.

---

#### At This Point You Should Have…

- Python 3.10 or newer — mcp 2.x declares `Requires-Python >=3.10`
- Claude Code installed and working
- The Google Cloud SDK, logged in, with application default credentials
- A Google Cloud project with Cloud Run GPU access in your region (this one runs in `us-east4`)
- A GCS bucket named `<project>-bucket` for the model weights

---

#### Setup the Basic Environment

Clone the repository and switch to the Cloud Run directory:

```shell
cd ~
git clone https://github.com/xbill9/gemma4-dev
cd gemma4-dev/gpu-2B-cloudrun-devops-agent
```

Then run **init.sh** once. It checks your `gcloud` login and application default credentials, asks for a project ID, installs the Python requirements, enables the Cloud Run, Secret Manager and related APIs, and grants the default compute service account its roles. It pauses on errors and waits for input, so run it in a terminal:

```shell
source init.sh
```

If your session times out or you need to reset your variables, run **set_env.sh**:

```shell
source set_env.sh
```

```plaintext
Current Environment:
  GOOGLE_CLOUD_PROJECT=aisprint-491218
  GOOGLE_CLOUD_LOCATION=us-east4
  SERVICE_NAME=gpu-2b-l4-devops-agent
  MODEL_NAME=/mnt/models/gemma-4-E2B-it
  VLLM_BASE_URL=<unset — discovered via gcloud>

Cloud Run here is --no-allow-unauthenticated. If calls fail, run: source ./set_adc.sh
```

`VLLM_BASE_URL` can stay unset: the MCP server finds the service URL through `gcloud` when a tool first needs it.

---

#### Model Management Tool with MCP Stdio Transport

One of the key features that the MCP libraries provide is abstracting various transport methods. The tool implementation is the same no matter which transport the MCP client uses to connect.

The simplest transport is stdio — the client launches the server as a local process and talks to it over stdin and stdout. Both must run in the same environment. In this project Claude Code is the MCP client.

The server is created in one line:

```python
# Initialize MCP server (mcp 2.x; FastMCP was renamed MCPServer)
mcp = MCPServer("Self-Hosted vLLM DevOps Agent")
```

That line used to say `FastMCP`. The next section is why it changed.

---

#### Wait — Why MCPServer and not FastMCP?

Nothing in the repository changed. A fresh install did. `requirements.txt` listed `mcp` with no version bound, so the next `pip install` resolved the 2.x line, and the server stopped importing:

```shell
python3 -c "from mcp.server.fastmcp import FastMCP"
```

```plaintext
    raise ModuleNotFoundError(_MESSAGE, name=__name__)
ModuleNotFoundError: No module named 'mcp.server.fastmcp'. This is mcp 2.x, where FastMCP was renamed to MCPServer (from mcp.server.mcpserver import MCPServer) and other APIs changed; see the migration guide at https://py.sdk.modelcontextprotocol.io/v2/migration/#fastmcp-renamed-to-mcpserver or pin 'mcp<2' to keep running v1 code.
```

In Claude Code it showed up less helpfully, as a server listed with `Connection closed`: it died on the import before the handshake.

The error names two fixes. Pinning `mcp<2` is legitimate — the v1 line still gets critical fixes — but these projects install into one system Python with no virtualenvs, so a pin here is a downgrade for every other project on the machine. Migrating keeps the change inside the repository.

The whole code change is the import and the constructor:

```diff
-from mcp.server.fastmcp import FastMCP
+from mcp.server.mcpserver import MCPServer
 ...
-mcp = FastMCP("Self-Hosted vLLM DevOps Agent")
+mcp = MCPServer("Self-Hosted vLLM DevOps Agent")
```

`@mcp.tool()`, `@mcp.resource()`, `mcp.run()` and every tool body stay as they are. Before editing, check the rest of the migration guide's list against your own server:

```shell
grep -c "^@mcp\.\(tool\|resource\)" server.py
grep -A1 "^@mcp\." server.py | grep -c "^def"
grep -n "get_running_loop\|asyncio.run(" server.py || echo "(no matches)"
grep -n "^import httpx" server.py; grep -n "^httpx" requirements.txt
```

```plaintext
28
12
(no matches)
13:import httpx
14:httpx
```

Three things to know from that output:

- **`mcp` 2.x no longer installs `httpx`.** It depends on `httpx2` instead. This server imports `httpx` and was safe only because `requirements.txt` already declared it.
- **The 12 sync handlers now run on a worker thread.** Only code that expects the event loop's thread breaks, and there is none here.
- **The server's version is now blank** unless you pass `version=` to `MCPServer(...)`. Nothing breaks; it shows in the handshake below.

The requirement then says what the code needs — `mcp>=2` in place of the bare `mcp` line.

---

#### 🔎 Tip: Change the Usage Before the Import

This project has a Claude Code hook that runs `ruff check --fix` after every edit. Change the import line first and, for a moment, `MCPServer` is imported but unused — so the hook deletes it:

```plaintext
--- server.py
+++ server.py
@@ -1,3 +1,2 @@
-from mcp.server.mcpserver import MCPServer
 
 mcp = FastMCP("demo")

Would fix 1 error.
```

Make both changes in one edit, or change the usage first. Any editor that runs `ruff check --fix` on save does the same thing.

---

#### Running the Python Code

The project can be linted:

```shell
make lint
```

```plaintext
ruff check .
All checks passed!
ruff format --check .
14 files already formatted
mypy .
Success: no issues found in 6 source files
```

and tested:

```shell
make test
```

```plaintext
----------------------------------------------------------------------
Ran 28 tests in 1.151s

OK
```

The suite compares the registered tool set against a hard-coded list, so a tool that silently failed to register after the rename would fail here. ✅

---

#### Test the Protocol by Hand

Unit tests call Python. A client speaks JSON-RPC over stdio, so test that too. **Hold stdin open with `sleep`** — with a bare `printf` pipe the server sees end-of-input and exits after answering only `initialize`:

```shell
{ printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  '{"jsonrpc":"2.0","id":3,"method":"resources/list","params":{}}'; sleep 5; } \
  | python3 server.py 2>/dev/null
```

Summarised:

```plaintext
initialize OK: name='Self-Hosted vLLM DevOps Agent' version='' proto 2025-06-18
tools/list OK: 27 tools -> cloudrun_analyze_cloud_logging, cloudrun_analyze_gpu_logs, cloudrun_check_gpu_quotas, cloudrun_deploy, ...
resources/list OK: ['config://vllm-deployment-template']
```

🟢 27 tools and the resource. Keep this snippet — it is the fastest way to tell "my server is broken" from "my client config is broken."

---

#### Claude Code .mcp.json

Claude Code reads `.mcp.json` in the project directory. It launches `server.py` with the system `python3`:

```json
{
  "mcpServers": {
    "cloudrun-devops": {
      "command": "python3",
      "args": ["/home/xbill/gemma4-dev/gpu-2B-cloudrun-devops-agent/server.py"],
      "env": {
        "GOOGLE_CLOUD_PROJECT": "aisprint-491218",
        "GOOGLE_CLOUD_LOCATION": "us-east4",
        "VLLM_BASE_URL": "https://gpu-2b-l4-devops-agent-289270257791.us-east4.run.app",
        "MODEL_NAME": "/mnt/models/gemma-4-E2B-it"
      }
    }
  }
}
```

---

#### Validation with Claude Code

Check the connection from Claude Code to the local server:

```shell
claude mcp get cloudrun-devops
```

```plaintext
cloudrun-devops:
  Scope: Project config (shared via .mcp.json)
  Status: ✔ Connected
```

If the server failed at startup earlier in the session, reconnect it from `/mcp` or start a new session to pick up the fixed code.

---

#### Model Lifecycle Management via MCP

The MCP tools cover the whole lifecycle of the Cloud Run deployment. Every tool is prefixed `cloudrun_`. Abridged output of `cloudrun_get_help`:

```markdown
The server is running in CLOUD RUN mode targeting NVIDIA L4 GPU in region us-east4.

🐳 Infrastructure & Deployment
  cloudrun_deploy, cloudrun_destroy, cloudrun_status, cloudrun_update_scaling,
  cloudrun_get_deployment_config, cloudrun_get_gpu_deployment_config, cloudrun_check_gpu_quotas
📊 Model Management
  cloudrun_list_vertex_models, cloudrun_list_bucket_models, cloudrun_save_hf_token,
  cloudrun_get_vertex_ai_model_copy_instructions, cloudrun_get_huggingface_model_copy_instructions,
  cloudrun_get_huggingfacehub_download_path
📊 Monitoring & Status
  cloudrun_get_metrics, cloudrun_get_system_status, cloudrun_get_endpoint,
  cloudrun_get_endpoint_url, cloudrun_get_model_details, cloudrun_verify_model_health
📈 Performance & Benchmarking
  cloudrun_run_benchmark
💬 Interaction & Diagnostics
  cloudrun_query_gemma4, cloudrun_query_gemma4_with_stats, cloudrun_query,
  cloudrun_analyze_cloud_logging, cloudrun_analyze_gpu_logs, cloudrun_suggest_sre_remediation
```

---

#### Stage the Model Weights

Cloud Run mounts the bucket read-only at `/mnt/models` through GCS FUSE, so the weights go to GCS once. Download to a real disk rather than `/tmp`, which on this host is a RAM-backed tmpfs smaller than the model:

```shell
hf download google/gemma-4-E2B-it --local-dir ~/hf-downloads/gemma-4-E2B-it
gcloud storage rsync ~/hf-downloads/gemma-4-E2B-it gs://aisprint-491218-bucket/gemma-4-E2B-it \
  --recursive --exclude='^\.cache/'
gcloud storage ls -l gs://aisprint-491218-bucket/gemma-4-E2B-it/
```

```plaintext
Average throughput: 41.4MiB/s
      4954  2026-09-10T14:51:14Z  gs://aisprint-491218-bucket/gemma-4-E2B-it/config.json
10246621918  2026-09-10T14:55:13Z  gs://aisprint-491218-bucket/gemma-4-E2B-it/model.safetensors
  32169626  2026-09-10T14:51:34Z  gs://aisprint-491218-bucket/gemma-4-E2B-it/tokenizer.json
TOTAL: 9 objects, 10278849571 bytes (9.57GiB)
```

The bucket is shared with other models, and `cloudrun_list_bucket_models` shows the trap:

```plaintext
> cloudrun_list_bucket_models

### Contents of GCS Bucket: aisprint-491218-bucket
- gemma-2b-it/config.json (0.00 MB)
- gemma-2b-it/model-00001-of-00002.safetensors (4716.15 MB)
...
- gemma-4-12B-it-qat-w4a16-ct/model.safetensors (9788.73 MB)
...
```

**Check the architecture, not the folder name.** `gemma-2b-it/` looks like the answer and is the original Gemma, which the `gemma4` parsers cannot serve. `config.json` settles it:

```shell
gcloud storage cat gs://aisprint-491218-bucket/gemma-4-E2B-it/config.json \
  | python3 -c 'import json,sys; c=json.load(sys.stdin); t=c["text_config"]; print(c["model_type"], c["architectures"], "hidden", t["hidden_size"], "layers", t["num_hidden_layers"])'
```

```plaintext
gemma4 ['Gemma4ForConditionalGeneration'] hidden 1536 layers 35
```

---

#### Deploy to Cloud Run

The `deploy-vllm` target in the `Makefile` is the single source of truth for the vLLM and Cloud Run flags. The ones that matter most:

| Flag | Value | Why |
|---|---|---|
| `--gpu-type` | `nvidia-l4` | one L4 per instance |
| `--no-gpu-zonal-redundancy` | | the cheaper L4 SKU |
| `--concurrency` | `4` | requests Cloud Run sends one instance |
| `--max-num-seqs` | `8` | vLLM's batch ceiling |
| `--tool-call-parser`, `--reasoning-parser` | `gemma4` | Gemma 4 tool calling breaks without either |

The service is also `--no-allow-unauthenticated`, so every caller needs an identity token.

```shell
make deploy
```

```plaintext
Deploying container to Cloud Run service [gpu-2b-l4-devops-agent] in project [aisprint-491218] region [us-east4]
Deploying new service...
Routing traffic.....done
Done.
Service [gpu-2b-l4-devops-agent] revision [gpu-2b-l4-devops-agent-00001-ssq] has been deployed and is serving 100 percent of traffic.
Service URL: https://gpu-2b-l4-devops-agent-289270257791.us-east4.run.app
```

`gcloud run deploy` returns only once the startup probe passes, and the probe waits `initialDelaySeconds=180` before its first check. Expect several minutes.

---

#### Demo Mode: Fixed Instances

The default autoscales between zero and one instance, which means a cold GPU start after the service goes idle. A demo cannot wait for that. The `Makefile` takes a `SCALING` variable:

```make
SCALING ?= auto
ifeq ($(SCALING),auto)
SCALING_FLAGS = --scaling=auto --max-instances=1 --min-instances=0
else
SCALING_FLAGS = --scaling=$(SCALING)
endif
```

```shell
make deploy SCALING=1
gcloud run services describe gpu-2b-l4-devops-agent --region us-east4 --format='yaml(metadata.annotations)'
```

```plaintext
    run.googleapis.com/manualInstanceCount: '1'
    run.googleapis.com/scalingMode: manual
```

One L4 now runs until you change it. Plain `make deploy`, and the `cloudrun_deploy` and `cloudrun_update_scaling` tools, all pass `--scaling=auto` on purpose — a service stuck in manual scaling at zero returns 503 to every request. So any of them quietly takes a demo back to scale-to-zero.

---

#### Checking System Status

The status can be checked with an MCP tool:

```plaintext
> cloudrun_get_system_status

### 🌀 GPU Cloud Run System Status (gpu-2b-l4-devops-agent)
- vLLM Health: 🟢 Online (https://gpu-2b-l4-devops-agent-289270257791.us-east4.run.app)
- Cloud Run Service Status: 🟢 Ready
👉 Next Step: Use cloudrun_query_gemma4 to interact with the model.
```

---

#### Cross Check The Deployed Model

Ask vLLM what it loaded:

```shell
curl -s -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  https://gpu-2b-l4-devops-agent-289270257791.us-east4.run.app/v1/models | python3 -m json.tool
```

```json
{
    "object": "list",
    "data": [
        {
            "id": "/mnt/models/gemma-4-E2B-it",
            "object": "model",
            "owned_by": "vllm",
            "max_model_len": 16384
        }
    ]
}
```

The model id is the mount path, not the Hugging Face repo id. The container is started with `--model=/mnt/models/<path>`, so that path is the name the OpenAI API expects.

Then the MCP health check:

```plaintext
> cloudrun_verify_model_health

✅ Model health check PASSED.
Model: /mnt/models/gemma-4-E2B-it
Response: 'Hello! Yes, I am working. I am Gemma 4, a Large La...'
Latency: 0.87 seconds.
```

Right after the first deploy the same check took 2.48 seconds. That answer came through the whole stack: Claude Code called the tool over MCP, and the tool called vLLM on Cloud Run. 🟢

---

#### Benchmark the Model

Ask the agent for `cloudrun_run_benchmark` with its defaults: one warmup request, then 20 requests at each concurrency level of 1, 2, 4 and 8, up to 128 output tokens each, one fixed prompt at temperature 0.

| Concurrency | Req/s | Tokens/s | Avg latency | P95 latency |
|---:|---:|---:|---:|---:|
| 1 | 0.39 | 49.63 | 2.58 s | 2.59 s |
| 2 | 0.75 | 95.36 | 2.68 s | 2.74 s |
| 4 | 1.47 | 188.46 | 2.71 s | 2.78 s |
| 8 | 1.48 | 189.53 | 4.86 s | 5.47 s |

Every request at every level succeeded. Three readings:

**From 1 to 4, throughput scales almost linearly.** 188.46 tokens/s is 3.8x the single-stream 49.63 (arithmetic), while average latency moves from 2.58 s to 2.71 s. The L4 is nowhere near full at 4.

**From 4 to 8, it stops.** Throughput rises 0.6% (arithmetic) while average latency goes from 2.71 s to 4.86 s. Half the requests are waiting.

**The ceiling is a Cloud Run setting, not the GPU.** One instance accepts `--concurrency=4` requests at a time, and vLLM would batch up to `--max-num-seqs=8`. The next section checks that against the same chip with nothing in front of it.

---

#### Compare to Other Deployments

The same checkpoint was served on the same chip — one NVIDIA L4 in an AWS EC2 `g6.2xlarge` — with vLLM and `--max-num-seqs 8`, and swept with `vllm bench serve` at the same concurrency levels and 128 output tokens:

| Concurrency | Cloud Run L4 tok/s | EC2 g6 L4 tok/s | |
|---:|---:|---:|---|
| 1 | 49.63 | 46.09 | 🥇 Cloud Run |
| 2 | 95.36 | 92.6 | 🥇 Cloud Run |
| 4 | 188.46 | 175.83 | 🥇 Cloud Run |
| 8 | 189.53 | 360.17 | 🥇 EC2 g6 |

**Up to 4, the two are the same chip doing the same work** — within 8% of each other at every level, with Cloud Run slightly ahead.

**At 8, the g6 keeps going.** It reaches 360.17 tokens/s, 1.9x Cloud Run's 189.53 (arithmetic), because nothing in front of vLLM caps admissions at 4. That is the direct evidence that Cloud Run's plateau is `--concurrency=4`, not the L4.

Read the rows as a shape, not a leaderboard. The runs differ in engine version, KV-cache dtype, prompt and client location; the scope paragraph at the end lists each.

---

#### And Price/Performance?

Cloud Run bills a GPU service by the second for the whole instance while it runs — GPU, CPU and memory — because the L4 requires `--no-cpu-throttling`. List prices in `us-east4` from the Cloud Billing catalog:

| Component | Price | Per hour |
|---|---:|---:|
| NVIDIA L4, no zonal redundancy | $0.0001867 / s | $0.6721 |
| 8 vCPU, instance-based | $0.000018 / vCPU-s | $0.5184 |
| 32 GiB memory, instance-based | $0.000002 / GiB-s | $0.2304 |
| **One instance** | | **$1.4209** |

That is $34.10 a day in demo mode with one fixed instance (arithmetic). In the default mode an idle service scales to zero instances.

Per million output tokens, from the benchmark (arithmetic):

| Deployment | Tokens/s | $ / hour | $ / M tokens |
|---|---:|---:|---:|
| Cloud Run, concurrency 1 | 49.63 | 1.4209 | 7.95 |
| Cloud Run, concurrency 4 | 188.46 | 1.4209 | 2.09 |
| Cloud Run, concurrency 8 | 189.53 | 1.4209 | 2.08 |
| 🥇 EC2 g6 spot, concurrency 8 | 360.17 | 0.9412 | 0.73 |

The winner is the VM, on two counts: a lower hourly rate, and twice the throughput with no admission cap. Cloud Run's price is list and on-demand; the g6 price is a spot price and can be reclaimed. What Cloud Run buys is no VM to manage and zero cost while idle — for intermittent SRE work that is the number that matters.

---

#### Teardown

```shell
make destroy
```

Not run for this article — the demo service is still up. It deletes the Cloud Run service; the weights stay in the bucket for the next deploy.

---

#### Summary

The goal of this article was to deploy Gemma 4 E2B to a Cloud Run NVIDIA L4 GPU with vLLM, and to manage the whole lifecycle from Claude Code through a Python MCP server. The key to the solution was bringing the MCP server up and validating it locally before the deploy — which is where the move to the MCP SDK 2.x surfaced, and where it cost one import and one class name. The deployment results were:

- 🟢 The migrated MCP server registered all 27 tools and the resource unchanged, and drove staging, deploy, validation and benchmarking
- 🟢 Gemma 4 E2B deployed to Cloud Run with one `make deploy` and passed the MCP health check in 0.87 seconds
- 🟢 Throughput scaled 3.8x from concurrency 1 to 4 with flat latency, peaking at 189.53 tokens/s
- ❌ Past 4, Cloud Run's `--concurrency=4` is the ceiling — the same L4 on EC2 reached 360.17 tokens/s at 8
- ⚠️ One fixed Cloud Run L4 instance lists at $1.4209 an hour, $2.08 per million output tokens at peak

Scope: one Cloud Run instance with one NVIDIA L4 in `us-east4`, vLLM `v0.26.0-cu129`, Gemma 4 E2B with bf16 weights and fp8 KV cache, manual scaling at one fixed instance during the sweep. One sweep of 20 requests per level, a single fixed prompt, 128 max output tokens, driven over the internet from a laptop. The EC2 comparison is a separate run: vLLM 0.28.0, bf16 KV cache, 1,024-token random prompts, 8 to 32 requests per level, driven on the instance itself, spot-priced in `us-east-1`. Costs are arithmetic on list prices, not a bill. MCP server on mcp 2.2.0 and Python 3.14.7.

The strategy for using MCP for Gemma 4 GPU deployment to Cloud Run with Claude Code was validated with an incremental step by step approach.

#### References

* [gpu-2B-cloudrun-devops-agent | GitHub](https://github.com/xbill9/gemma4-dev/tree/main/gpu-2B-cloudrun-devops-agent)
* [Migration Guide: v1 to v2 | MCP Python SDK](https://py.sdk.modelcontextprotocol.io/v2/migration/)
* [GitHub - modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)
