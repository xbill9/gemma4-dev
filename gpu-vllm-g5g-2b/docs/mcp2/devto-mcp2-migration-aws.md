---
title: "FastMCP Is Now MCPServer on AWS: Moving a boto3 EC2 MCP Server to the MCP Python SDK 2.x"
published: false
description: "What basic MCP Python SDK 2.x support takes for an AWS EC2 MCP server built on boto3: one rename in server.py, snake_case in the tests, a floor in requirements.txt, and why neither the wire format nor the boto3 calls change."
tags: aws, mcp, python, ec2
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-vllm-g5g-2b/docs/mcp2/devto-cover.9681da89.jpg
---

This article provides a step by step migration guide for an AWS MCP server from the MCP Python SDK 1.x (`FastMCP`) to 2.x (`MCPServer`). The server manages Gemma 4 E2B on an Amazon EC2 G5g instance, a Graviton2 host with an NVIDIA T4G GPU, and a suite of Python MCP tools built on boto3 simplifies management of the vLLM hosted deployment.

https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-g5g-2b

---

#### What Broke?

The rig's `requirements.txt` listed `mcp` with no version bound. Once the machine's Python moved to mcp 2.2.0, the server stopped importing:

```shell
python3 -c "import server"
```

```plaintext
ModuleNotFoundError: No module named 'mcp.server.fastmcp'. This is mcp 2.x, where FastMCP was renamed to MCPServer (from mcp.server.mcpserver import MCPServer) and other APIs changed; see the migration guide at https://py.sdk.modelcontextprotocol.io/v2/migration/#fastmcp-renamed-to-mcpserver or pin 'mcp<2' to keep running v1 code.
```

In an MCP client this surfaces less helpfully: the server is launched, dies on the import, and stdio closes before the handshake. The same code on mcp 1.30.0 passes all 27 of its tests. ✅

The error message names both fixes, so the first decision is which one.

---

#### Pin or Migrate?

| | Pin `mcp<2` | Migrate to `MCPServer` |
|---|---|---|
| Code change | none | import, class name, tests |
| Where the fix lives | every interpreter that runs the server | the repository |
| Shared Python | holds back `mcp` for everything on it | nothing global changes |
| Future fixes | the v1 maintenance line | the current line |

The migration guide says the v1.x line keeps receiving critical bug fixes and security patches, so pinning is legitimate. This rig migrates because **every project here installs into one system Python, with no virtualenvs**, and the deployment image uses the same layout.

That shared Python has an AWS-specific wrinkle. Upgrading `mcp` on it drew this from pip:

```plaintext
strands-agents 1.55.0 requires mcp<2.2,>=1.23.0, but you have mcp 2.2.0 which is incompatible.
Successfully installed mcp-2.2.0 mcp-types-2.2.0
```

pip installs anyway and only warns. If Strands Agents shares an interpreter with your MCP server, check its `mcp` bound before choosing a 2.x release, or give the two separate interpreters.

---

#### At This Point You Should Have…

- Python 3.10 or newer — mcp 2.x declares `Requires-Python >=3.10`
- The repository cloned, and `gpu-vllm-g5g-2b/` as your working directory
- `pip install -r requirements.txt` done against the interpreter your MCP client launches
- `ruff` on the path for `make lint`

No AWS credentials are needed for any step here. The tests are offline by design, and the stdio check only lists tools, which never runs a handler or calls boto3.

---

#### What Changed in 2.x?

The migration guide's table of changes most projects hit, against this server — stdio transport, 15 `@mcp.tool()` tools, boto3 for every AWS call, and no client code:

| Change | First symptom | This server |
|---|---|---|
| `FastMCP` renamed to `MCPServer` | `No module named 'mcp.server.fastmcp'` | ❌ hit |
| camelCase fields renamed to snake_case | `'Tool' object has no attribute 'inputSchema'` | ❌ hit, in the tests |
| `httpx` replaced by `httpx2` | `No module named 'httpx'` | ⚠️ exposed, already declared |
| Sync handlers run on a worker thread | `get_running_loop()` raises in a `def` handler | ✅ every handler is `async` |
| `MCP_*` env no longer read into settings | settings silently ignored | ✅ `MCP_SERVER_NAME` is the rig's own |

The first row is the one everybody hits. The second is the one this article is about, because the companion Cloud Run migration never saw it.

---

#### Step 1 — Find Your Exposure

Measure before editing:

```shell
grep -c "^@mcp\.\(tool\|resource\|prompt\)" server.py
grep -A1 "^@mcp\." server.py | grep -c "^async def"
grep -A1 "^@mcp\." server.py | grep -c "^def"
grep -n "get_running_loop\|asyncio.run(" server.py || echo "(no matches)"
grep -n "^import httpx" server.py; grep -n "^httpx" requirements.txt
grep -n "Hint\b\|inputSchema\|outputSchema\|mimeType" tests/*.py
```

```plaintext
15
15
0
(no matches)
26:import httpx
2:httpx
43:            name for name, tool in self.tools.items() if tool.annotations.destructiveHint
53:            schema = self.tools[name].inputSchema["properties"]
```

15 handlers, all `async`, none touching the event loop, and `httpx` declared on its own line. The last grep is the one a FastMCP-only checklist misses: **two camelCase attribute reads in the test suite.**

One more grep hit deserves a look. `grep -n "MCP_" server.py` finds `MCP_SERVER_NAME`. v2 no longer reads `MCP_*` environment variables into server settings, but this one is the rig's own: it is read with `os.getenv` and passed in as the server's name, so nothing changes.

---

#### Step 2 — The Rename

The whole change to `server.py`:

```diff
-from mcp.server.fastmcp import FastMCP
+from mcp.server.mcpserver import MCPServer
 from mcp.types import ToolAnnotations
 ...
 MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", RIG_NAME)
-mcp = FastMCP(MCP_SERVER_NAME)
+mcp = MCPServer(MCP_SERVER_NAME)
 READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
 WRITE = ToolAnnotations(destructiveHint=False)
 DESTRUCTIVE = ToolAnnotations(destructiveHint=True)
```

Make both line changes in one edit. An editor that runs `ruff check --fix` on save deletes an import that is momentarily unused, and the next edit leaves `MCPServer` undefined.

The three `ToolAnnotations(...)` lines stay exactly as they are, camelCase and all. That is deliberate, and Step 4 shows why it is safe.

---

#### Step 3 — Run the Tests

```shell
python3 -m unittest discover -s tests
```

```plaintext
ERROR: test_annotations (test_server.ToolCatalogTests.test_annotations)
AttributeError: 'ToolAnnotations' object has no attribute 'destructiveHint'. Did you mean: 'destructive_hint'?
ERROR: test_launch_defaults_to_spot_and_build (test_server.ToolCatalogTests.test_launch_defaults_to_spot_and_build)
AttributeError: 'Tool' object has no attribute 'inputSchema'. Did you mean: 'input_schema'?
FAIL: test_skill_is_complete_in_both_copies (test_server.RepoHygieneTests.test_skill_is_complete_in_both_copies)
Ran 27 tests in 0.012s
FAILED (failures=1, errors=2)
```

The server imports and registers its tools. What broke is Python code that reads protocol models, and here that code is the test suite.

**The failing test is the one that guards EC2 termination.** `test_annotations` asserts that exactly two tools, `stop_g5g_instance` and `terminate_g5g_instance`, carry `destructiveHint`. On an AWS server that flag is how a client knows a tool can end an instance and destroy its root volume. A migration that deleted this test to get green would remove the check that matters most.

The third failure is this rig's own: a test that the generated skill copies match `server.py`. It clears in Step 5.

---

#### Step 4 — Snake Case in Python, camelCase on the Wire

v2 renamed every protocol model field to snake_case for Python attribute access. The fix is two identifiers:

```diff
-            name for name, tool in self.tools.items() if tool.annotations.destructiveHint
+            name for name, tool in self.tools.items() if tool.annotations.destructive_hint
 ...
-            schema = self.tools[name].inputSchema["properties"]
+            schema = self.tools[name].input_schema["properties"]
```

Why the `ToolAnnotations(destructiveHint=True)` constructors in `server.py` did not need touching:

```plaintext
>>> ToolAnnotations(destructiveHint=True).destructive_hint
True
>>> ToolAnnotations(destructiveHint=True).destructiveHint
AttributeError: 'ToolAnnotations' object has no attribute 'destructiveHint'
>>> Tool(...).model_dump(exclude_none=True)
{'name': 'x', 'input_schema': {'type': 'object'}}
>>> Tool(...).model_dump(by_alias=True, exclude_none=True)
{'name': 'x', 'inputSchema': {'type': 'object'}}
```

Constructors accept both spellings; attribute access is snake_case only. The last two lines are the trap the guide warns about: a plain `model_dump()` now emits snake_case keys that other MCP implementations will not recognise, with no error. If your server serialises protocol models itself, add `by_alias=True`. This one does not.

---

#### Step 5 — Pin the Floor

Code that imports `mcp.server.mcpserver` cannot run on 1.x:

```diff
-mcp
+mcp>=2,<3
```

The upper bound is the migration guide's own example: meet 3.x on purpose, not by `pip install`. Keep `httpx` on its own line in the same file. v2 depends on `httpx2` instead, so a server that imports httpx without declaring it loses it on a fresh install, with a traceback that never mentions mcp.

This rig also ships its server as a skill, with copies of `server.py` and `requirements.txt` under `skills/`. `make skill` regenerates them, which clears the third failure. If your MCP server is vendored anywhere, into a container build context or a Lambda package, refresh that copy too.

---

#### Step 6 — Lint and Test

```shell
make lint
```

```plaintext
ruff check server.py refresh_skill.py tests
All checks passed!
lint OK
```

```shell
python3 -m unittest discover -s tests
```

```plaintext
----------------------------------------------------------------------
Ran 27 tests in 0.011s

OK
```

27 tests, the same count that passed on mcp 1.30.0 before the change. ✅

---

#### Step 7 — Test the Protocol by Hand

Unit tests call Python. A client speaks JSON-RPC over stdio, and the wire is where the camelCase question gets its real answer. **Hold stdin open with `sleep`**, or the server sees end-of-input and exits after the first reply:

```shell
{ printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'; sleep 5; } \
  | python3 server.py 2>/dev/null
```

The responses are JSON; summarised:

```plaintext
initialize OK: name='gpu-vllm-g5g-2b' version='' proto 2025-06-18
tools/list OK: 15 tools -> check_g5g_quotas, create_g5g_instance, get_build_progress, get_deployment_config, ...
wire keys on terminate_g5g_instance: ['annotations', 'description', 'inputSchema', 'name', 'outputSchema', 'title']
annotations on the wire: {"destructiveHint": true}
destructive on the wire: ['stop_g5g_instance', 'terminate_g5g_instance']
```

🟢 All 15 tools, and the wire is still camelCase: `inputSchema`, `destructiveHint`. The same two tools are marked destructive. **Clients see no change.** The rename is visible only to Python that reads the models.

Note `version=''`. In v2 a server that does not pass a version reports an empty string instead of the installed SDK's version. Pass `version="..."` to `MCPServer(...)` if anything displays it.

---

#### What About boto3?

boto3 is synchronous, and that is the change most AWS MCP servers should look at hardest.

This rig was already safe. Every tool is `async def`, and each boto3 call goes through one helper:

```python
async def _call(func, **kwargs):
    return await asyncio.to_thread(func, **kwargs)
```

So a slow EC2 or SSM call runs on a worker thread in v1 and v2 alike. v2's change to sync handlers does nothing here.

The common shape is different: a plain `def` tool that calls `ec2.describe_instances()` directly. The migration guide says v1 ran such a handler inline on the event loop, so one slow AWS call stalled every other request on the server, and v2 runs it on a worker thread. That is a free concurrency gain for boto3 code. The only thing it breaks is code in a `def` handler that expects the loop's thread, such as `asyncio.get_running_loop()`. The guide also points out the reverse: an `async def` tool that calls boto3 without a thread still blocks the loop in both versions.

---

#### The Rest of the Fleet

Upgrading one shared Python moves every MCP server on it at once:

```shell
grep -l "from mcp.server.fastmcp import" */server.py | wc -l
grep -l "from mcp.server.mcpserver import" */server.py
```

```plaintext
37
gpu-2B-cloudrun-devops-agent/server.py
local-llamacpp-1650ti-2b-q4_0/server.py
gpu-vllm-g5g-2b/server.py
```

37 servers in the monorepo still import FastMCP, 13 of them EC2 or Inferentia rigs, and none of them starts on this Python until it gets the same rename. That is the cost of migrating over pinning on a shared interpreter, paid up front. The per-server change is small enough to make paying it reasonable.

---

#### Cheat Sheet

```shell
# exposure
grep -rn "mcp.server.fastmcp" .
grep -A1 "^@mcp\." server.py | grep -c "^def"
grep -n "Hint\b\|inputSchema\|outputSchema\|mimeType" tests/*.py
grep -n "^import httpx" server.py; grep -n "^httpx" requirements.txt

# the rename, in ONE edit
#   from mcp.server.fastmcp import FastMCP  ->  from mcp.server.mcpserver import MCPServer
#   FastMCP("name")                         ->  MCPServer("name")
# attribute reads: .destructiveHint -> .destructive_hint, .inputSchema -> .input_schema
# constructors:    ToolAnnotations(destructiveHint=True) still works, leave it
# requirements.txt: mcp -> mcp>=2,<3, and declare httpx if you import it

make lint && python3 -m unittest discover -s tests

# stdio smoke test: hold stdin open
{ printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"p","version":"0"}}}' \
                '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
                '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'; sleep 5; } | python3 server.py 2>/dev/null
```

---

#### Summary

The goal of this article was to give an AWS EC2 MCP server basic MCP Python SDK 2.x support without changing what it does or what its clients see. The key to the solution was auditing the tests as well as the server, because on this rig the SDK's snake_case change broke the test suite and not the server. The migration results were:

- 🟢 The server change was the import and the constructor; all 15 tools registered, and the camelCase `ToolAnnotations(...)` constructors stayed
- ❌ Two camelCase attribute reads broke in the tests, including the one guarding EC2 stop and terminate; the fix is `destructive_hint` and `input_schema`
- 🟢 The wire format did not change: `tools/list` still sends `inputSchema` and `destructiveHint`, and the same two tools are marked destructive
- 🟢 boto3 was already off the event loop through `asyncio.to_thread`, so v2's worker-thread change was a no-op
- ⚠️ Upgrading a shared Python to mcp 2.2.0 put `strands-agents` 1.55.0 outside its `mcp<2.2` bound and stopped 37 other FastMCP servers until they are migrated

Scope: mcp 2.2.0 on Python 3.14.7, with 1.30.0 as the v1 reference, boto3 1.43.90, ruff 0.16.7, on one Debian workstation. Offline unit tests and one stdio handshake; no EC2 instance was provisioned, because nothing in the migration changes an AWS call. The deployment the tools manage is unchanged and is covered in the rig's earlier G5g article.

The strategy for using MCP for migrating an AWS MCP server to the MCP SDK 2.x was validated with an incremental step by step approach.

#### References

* [Migration Guide: v1 to v2 | MCP Python SDK](https://py.sdk.modelcontextprotocol.io/v2/migration/)
* [FastMCP Is Now MCPServer: Migrating a Python MCP Server to the MCP SDK 2.x | dev.to](https://dev.to/gde/fastmcp-is-now-mcpserver-migrating-a-python-mcp-server-to-the-mcp-sdk-2x-2nhj)
* [gpu-vllm-g5g-2b | GitHub](https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-g5g-2b)

---

*mcp 2.2.0 (mcp-types 2.2.0), Python 3.14.7, boto3 1.43.90, ruff 0.16.7.*
