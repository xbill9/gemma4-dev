---
title: "FastMCP Is Now MCPServer: Migrating a Python MCP Server to the MCP SDK 2.x"
published: false
description: "Step-by-step: moving a FastMCP server to the MCP Python SDK 2.x — what broke, what changed, what did not, and a redeploy of its Gemma 4 vLLM backend to a Cloud Run L4 GPU."
tags: mcp, python, googlecloud, vllm
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-queens/main/gpu-2B-cloudrun-devops-agent/docs/devto-cover.5bbae7c1.jpg
---

This article provides a step by step migration guide for a Python MCP server from the MCP Python SDK 1.x (`FastMCP`) to 2.x (`MCPServer`), followed by a step by step deployment of Gemma 4 E2B to a Cloud Run hosted GPU enabled system. A suite of Python MCP tools is built to simplify management of the vLLM hosted deployment.

https://github.com/xbill9/gemma4-queens/tree/main/gpu-2B-cloudrun-devops-agent

---

#### What Broke?

Nothing in the repository changed. A fresh install did.

The project's `requirements.txt` listed `mcp` with no version bound, so the next `pip install` resolved the 2.x line. The server stopped importing:

```shell
python3 -c "from mcp.server.fastmcp import FastMCP"
```

```plaintext
    raise ModuleNotFoundError(_MESSAGE, name=__name__)
ModuleNotFoundError: No module named 'mcp.server.fastmcp'. This is mcp 2.x, where FastMCP was renamed to MCPServer (from mcp.server.mcpserver import MCPServer) and other APIs changed; see the migration guide at https://py.sdk.modelcontextprotocol.io/v2/migration/#fastmcp-renamed-to-mcpserver or pin 'mcp<2' to keep running v1 code.
```

It showed up in Claude Code first, and less helpfully. The MCP server was listed as failed with `Connection closed`: Claude Code launched it with `python3`, it died on the import, and stdio closed before the handshake.

That error message deserves credit. The 2.x package still ships a `mcp/server/fastmcp.py`, and its only job is to raise it. A bare `No module named` would have sent everyone hunting for a broken install. ✅

---

#### Pin or Migrate?

Two fixes, and the error message names both.

| | Pin `mcp<2` | Migrate to `MCPServer` |
|---|---|---|
| Code change | none | import and class name |
| Where the fix lives | every interpreter that runs the server | the repository |
| Shared system Python | downgrades `mcp` for everything on it | nothing global changes |
| Future fixes | the v1 maintenance line | the current line |

The migration guide says the v1.x line keeps receiving critical bug fixes and security patches, so pinning is legitimate. This rig rules it out for a different reason: **these projects install into one system Python, with no virtualenvs**, so a pin in one project is a downgrade for every other project on the machine. Migrating keeps the change inside the repository.

---

#### At This Point You Should Have…

- Python 3.10 or newer — mcp 2.x declares `Requires-Python >=3.10`
- The Google Cloud SDK, logged in, with application default credentials
- A Google Cloud project with Cloud Run GPU access in your region (this one runs in `us-east4`)
- A GCS bucket named `<project>-bucket` for the model weights
- The repository cloned, and the directory above as your working directory

---

#### What Changed in 2.x?

The official migration guide opens with a table of the changes almost every project hits. Here they are against this server — stdio transport, `@mcp.tool()` tools, one `@mcp.resource()`, and no client code:

| Change | First symptom | This server |
|---|---|---|
| `FastMCP` renamed to `MCPServer` | `No module named 'mcp.server.fastmcp'` | ❌ hit |
| `httpx` replaced by `httpx2` | `No module named 'httpx'` | ⚠️ exposed, already safe |
| Sync handlers run on a worker thread | `get_running_loop()` raises in a `def` handler | ⚠️ handlers move threads |
| camelCase fields renamed to snake_case | `'Tool' object has no attribute 'inputSchema'` | ✅ not used |
| Resource URIs are `str`, not `AnyUrl` | `'str' object has no attribute 'host'` | ✅ tests already call `str()` |
| Transport parameters moved off the constructor | `unexpected keyword argument 'port'` | ✅ name only |
| `McpError` renamed to `MCPError` | `cannot import name 'McpError'` | ✅ not used |

Three rows deserve more than a table cell.

**`mcp` no longer installs `httpx`.** v2 depends on `httpx2`, a fork of httpx, instead. `server.py` does `import httpx` for its HTTP probes and its benchmark, and the guide warns what happens next: a `ModuleNotFoundError` whose traceback never mentions mcp. This server escaped only because `requirements.txt` already listed `httpx` on its own line. If your server imports httpx and never declared it, declare it now.

**Sync handlers changed threads.** In v1 a plain `def` tool ran inline on the event loop, so a blocking call stalled every other request on the server. v2 runs sync handlers on a worker thread. The only thing that breaks is code that expects the loop's thread — `asyncio.get_running_loop()` in a `def` handler now raises. This server has none, so its sync tools gain concurrency for free. The move covers `def` handlers only: an `async def` tool that makes a blocking call still blocks the loop, in v1 and v2 alike.

**The server's version went blank.** In v1 an unversioned server reported the installed `mcp` version as its own. In v2 it reports an empty string. Nothing breaks, but it shows up in the handshake in Step 5.

---

#### What Did Not Change

The guide lists the everyday surface that carries over, and it is most of a typical FastMCP server:

- `@mcp.tool()`, `@mcp.resource()` and `@mcp.prompt()` take the same arguments and handler signatures
- Tool return handling: strings, dicts, models and content blocks are wrapped by the same rules
- `list_tools()` and `list_resources()` return the same lists
- `lifespan=` works as before

For this server that means no tool function changed. Every tool body is exactly what it was.

---

#### Step 1 — Find Your Exposure

Measure before editing. Five greps cover everything in the table above:

```shell
grep -c "^@mcp\.\(tool\|resource\)" server.py
grep -A1 "^@mcp\." server.py | grep -c "^def"
grep -n "get_running_loop\|asyncio.run(" server.py || echo "(no matches)"
grep -n "MCP_\|dotenv" server.py || echo "(no matches)"
grep -n "^import httpx" server.py; grep -n "^httpx" requirements.txt
```

```plaintext
28
12
(no matches)
(no matches)
13:import httpx
14:httpx
```

28 handlers, 12 of them sync, none touching the event loop, and `httpx` declared.

The `MCP_` grep checks one more change. v2 no longer reads `MCP_*` environment variables or a `.env` file into server settings. The guide points out that constructor arguments always took precedence, so those variables rarely did anything anyway. This server reads its own configuration from `GOOGLE_CLOUD_PROJECT`, `VLLM_BASE_URL` and friends, so nothing changes.

One check a grep cannot do: **v2 inserted `title` and `description` into the constructor's positional parameters.** A v1 call like `FastMCP("Demo", "You answer questions…")` still runs on v2, but the second string silently becomes the title and stops reaching the model as instructions. Keep the name positional and pass everything else by keyword. This server passes only the name.

---

#### Step 2 — The Rename

The whole code change:

```diff
-from mcp.server.fastmcp import FastMCP
+from mcp.server.mcpserver import MCPServer
 from openai import AsyncOpenAI
 ...
-# Initialize FastMCP server
-mcp = FastMCP("Self-Hosted vLLM DevOps Agent")
+# Initialize MCP server (mcp 2.x; FastMCP was renamed MCPServer)
+mcp = MCPServer("Self-Hosted vLLM DevOps Agent")
```

`@mcp.tool()`, `@mcp.resource()`, `mcp.run()` and every tool body stay as they are. Other submodules moved the same way — `mcp.server.fastmcp.*` is now `mcp.server.mcpserver.*`, and `ctx.fastmcp` is now `ctx.mcp_server` — but this server uses neither.

---

#### 🔎 Tip: Change the Usage Before the Import

This one cost a test run. The project has a Claude Code hook that runs `ruff format` and `ruff check --fix` after every edit. Change the import line first and, for a moment, `MCPServer` is imported but unused. The hook deletes it:

```shell
cat server.py
ruff check --fix --diff server.py
```

```plaintext
from mcp.server.mcpserver import MCPServer

mcp = FastMCP("demo")

--- server.py
+++ server.py
@@ -1,3 +1,2 @@
-from mcp.server.mcpserver import MCPServer
 
 mcp = FastMCP("demo")

Would fix 1 error.
```

The next edit renames the class, and the file now has no import at all:

```plaintext
    mcp = MCPServer("Self-Hosted vLLM DevOps Agent")
          ^^^^^^^^^
NameError: name 'MCPServer' is not defined
```

Make both changes in one edit, or change the usage first. Any editor that runs `ruff check --fix` on save will do the same thing.

---

#### Step 3 — Pin the Floor

Code that imports `mcp.server.mcpserver` cannot run on 1.x, so the requirement should say so:

```diff
-mcp
+mcp>=2
```

The guide's own example also caps the major version, `mcp>=2,<3`. That is the safer line if you would rather meet 3.x on purpose than by `pip install`. Keep `httpx` on its own line in the same file, for the reason above.

---

#### Step 4 — Lint and Test

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

```shell
make test
```

```plaintext
----------------------------------------------------------------------
Ran 28 tests in 1.058s

OK
```

The suite compares the registered tool set against a hard-coded list through `mcp.list_tools()`. That call is on the guide's unchanged list, and it is the test that would catch a tool silently failing to register after the rename. ✅

---

#### Step 5 — Test the Protocol by Hand

Unit tests call Python. A client speaks JSON-RPC over stdio, so test that too. **Hold stdin open with `sleep`**: with a bare `printf` pipe the server saw end-of-input and exited after answering only `initialize`.

```shell
{ printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  '{"jsonrpc":"2.0","id":3,"method":"resources/list","params":{}}'; sleep 5; } \
  | python3 server.py 2>/dev/null
```

The responses are JSON; summarised:

```plaintext
initialize OK: name='Self-Hosted vLLM DevOps Agent' version='' proto 2025-06-18
tools/list OK: 27 tools -> cloudrun_analyze_cloud_logging, cloudrun_analyze_gpu_logs, cloudrun_check_gpu_quotas, cloudrun_deploy, ...
resources/list OK: ['config://vllm-deployment-template']
```

🟢 27 tools and the resource, matching the suite's expected list. Note `version=''` — the unversioned-server change from the table. Pass `version="..."` to `MCPServer(...)` if a client or dashboard displays it.

Keep this snippet. It is the fastest way to tell "my server is broken" from "my client config is broken."

---

#### Step 6 — Register It

Claude Code reads `.mcp.json`. Nothing in it changes for the migration — it launches `server.py` with the system `python3`:

```json
{
  "mcpServers": {
    "cloudrun-devops": {
      "command": "python3",
      "args": ["/home/xbill/gemma4-queens/gpu-2B-cloudrun-devops-agent/server.py"],
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

If the server failed at startup earlier in the session, reconnect it from `/mcp` or start a new session to pick up the fixed code.

---

#### Step 7 — Stage the Model Weights

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

**Check the architecture, not the folder name.** The same bucket already held a `gemma-2b-it/` folder that looked like the answer and was the original Gemma, which the `gemma4` parsers cannot serve. `config.json` settles it:

```shell
gcloud storage cat gs://aisprint-491218-bucket/gemma-4-E2B-it/config.json \
  | python3 -c 'import json,sys; c=json.load(sys.stdin); t=c["text_config"]; print(c["model_type"], c["architectures"], "hidden", t["hidden_size"], "layers", t["num_hidden_layers"])'
```

```plaintext
gemma4 ['Gemma4ForConditionalGeneration'] hidden 1536 layers 35
```

---

#### Step 8 — Deploy to Cloud Run

The `deploy-vllm` target in the `Makefile` is the single source of truth for the vLLM and Cloud Run flags:

| Flag | Value | Why |
|---|---|---|
| `--gpu-type` | `nvidia-l4` | one L4 per instance |
| `--concurrency` | `4` | requests Cloud Run sends one instance |
| `--max-num-seqs` | `8` | vLLM's batch ceiling |
| `--tool-call-parser`, `--reasoning-parser` | `gemma4` | Gemma 4 tool calling breaks without either |
| `--no-allow-unauthenticated` | | callers need an identity token |

```shell
make deploy
```

```plaintext
Deploying container to Cloud Run service [gpu-2b-l4-devops-agent] in project [aisprint-491218] region [us-east4]
Deploying new service...
Creating Revision....................done
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

gcloud only accepts a positive instance count for manual scaling, so this cannot pin the service at zero. The min and max flags are passed only in auto mode, because gcloud's help does not say how they combine with a fixed count. One L4 now runs until you change it.

Plain `make deploy`, and the `cloudrun_deploy` and `cloudrun_update_scaling` tools, all pass `--scaling=auto` on purpose — a service stuck in manual scaling at zero returns 503 to every request. So any of them quietly takes a demo back to scale-to-zero.

---

#### Step 9 — Validate

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

Then ask the agent for `cloudrun_verify_model_health`:

```plaintext
✅ Model health check PASSED.
Model: /mnt/models/gemma-4-E2B-it
Response: 'Hello! Yes, I am working. I am Gemma 4, a Large La...'
Latency: 2.48 seconds.
```

That answer came through the migrated server: Claude Code called the tool over MCP, and the tool called vLLM. 🟢

---

#### Step 10 — Benchmark Sweep

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

**The ceiling is a Cloud Run setting, not the GPU.** One instance accepts `--concurrency=4` requests at a time, and vLLM would batch up to `--max-num-seqs=8`. The next sweep worth running redeploys with `--concurrency=8` and measures the L4 instead of the setting.

And the SDK version cannot move any of these numbers. The benchmark's HTTP calls go from `server.py` to vLLM; MCP only carries the one tool call that starts them. The sweep is here to show the migrated server driving the whole lifecycle, not to measure the SDK.

---

#### Teardown

```shell
make destroy
```

Not run for this article — the demo service is still up. It deletes the Cloud Run service; the weights stay in the bucket for the next deploy.

---

#### Cheat Sheet

```shell
# exposure
grep -rn "mcp.server.fastmcp" .
grep -A1 "^@mcp\." server.py | grep -c "^def"
grep -n "^import httpx" server.py; grep -n "^httpx" requirements.txt

# the rename, in ONE edit
#   from mcp.server.fastmcp import FastMCP  ->  from mcp.server.mcpserver import MCPServer
#   FastMCP("name")                         ->  MCPServer("name")
# requirements.txt: mcp -> mcp>=2 (or mcp>=2,<3), and declare httpx if you import it

make lint && make test

# stdio smoke test: hold stdin open
{ printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"p","version":"0"}}}' \
                '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
                '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'; sleep 5; } | python3 server.py 2>/dev/null

# deploy, demo, teardown
make deploy
make deploy SCALING=1
make destroy
```

---

#### Summary

The goal of this article was to move a Python MCP server off FastMCP and onto the MCP Python SDK 2.x without changing what it does. The key to the solution was measuring exposure against the migration guide before editing, which reduced the change to one import and one class name. The migration results were:

- 🟢 The code change was the import and the constructor; all 27 tools and the resource registered unchanged
- ⚠️ `mcp` 2.x no longer installs `httpx`; this server was safe only because it declared `httpx` itself
- 🟢 The 12 sync handlers now run on a worker thread, so their blocking calls no longer stall the event loop
- ❌ A formatter hook deleted the new import mid-edit; change the usage and the import together
- 🟢 The migrated server deployed Gemma 4 E2B to a Cloud Run L4 and drove a benchmark sweep that peaked at 189.53 tokens/s, capped by Cloud Run's `--concurrency=4`

Scope: mcp 2.2.0 on Python 3.14.7, with the 1.30.0 wheel as the v1 reference. One Cloud Run instance with one NVIDIA L4 in `us-east4`, vLLM `v0.26.0-cu129`, Gemma 4 E2B, in manual scaling at one fixed instance during the sweep. One sweep of 20 requests per level, a single fixed prompt, 128 max output tokens. It measures the serving stack, not the SDK.

Each step validated a different layer of the migrated server: the unit tests, the raw MCP protocol over stdio, and a live Cloud Run deployment.

#### References

* [Migration Guide: v1 to v2 | MCP Python SDK](https://py.sdk.modelcontextprotocol.io/v2/migration/)
* [gpu-2B-cloudrun-devops-agent | GitHub](https://github.com/xbill9/gemma4-queens/tree/main/gpu-2B-cloudrun-devops-agent)

---

*mcp 2.2.0 (mcp-types 2.2.0), Python 3.14.7, ruff 0.16.6, Google Cloud SDK 583.0.0, vLLM v0.26.0 on one NVIDIA L4, Cloud Run `us-east4`.*
