---
name: add-mcp-tool
description: Scaffold a new @mcp.tool() in server.py following this repo's conventions, plus its unit test in test_agent.py. Use when adding a capability to the vLLM DevOps MCP agent.
---

Add a new MCP tool named `$ARGUMENTS` (ask for the name and what it should do if not given).

## 1. Decide the tool's shape

Classify it, because each kind has an established pattern in `server.py`:

- **gcloud/subprocess tool** (deploy, destroy, status, scaling, quotas) — sync `def`, uses
  `subprocess.run([...], capture_output=True, text=True)`. Model on `cloudrun_update_scaling` / `cloudrun_status`.
- **Google Cloud SDK tool** (Secret Manager, Storage, Logging, Vertex) — `async def`, every blocking
  client call wrapped in `await asyncio.to_thread(client.method, request={...})`. Model on
  `cloudrun_save_hf_token` / `cloudrun_analyze_cloud_logging`.
- **Inference tool** (asks the model something) — `async def`, get the client from `get_vllm_client()`
  and the model id from `get_active_model_name(client)`. Model on `cloudrun_query_gemma4` / `cloudrun_suggest_sre_remediation`.
- **Raw HTTP tool** (hitting a vLLM path the OpenAI client doesn't cover, e.g. `/health`, `/metrics`) —
  `async with httpx.AsyncClient(timeout=...)`, with `Authorization: Bearer {get_auth_token()}`.
  Model on `cloudrun_get_metrics` / `cloudrun_get_model_details`.

Read the named reference tool before writing anything — match it rather than inventing a new style.

## 2. Write the tool

Place it near related tools in `server.py` (the file is loosely grouped: infra → models → monitoring →
benchmarks → diagnostics), not appended at the bottom.

Requirements:

- Decorate with `@mcp.tool()`.
- Name it `cloudrun_<verb>_<noun>`. Every tool in this server carries the `cloudrun_` prefix so it does
  not collide with the sibling TPU agent's tools when both MCP servers are loaded. Do not repeat
  "cloudrun" later in the name (`cloudrun_deploy`, not `cloudrun_deploy_cloudrun`).
- The docstring is the tool description the model reads. One line, imperative, says what it does and
  when to use it. Parameters get real type hints and sensible defaults (`service_name: str = DEFAULT_SERVICE_NAME`).
- Return a human-readable `str` prefixed with a status emoji: `✅` success, `🔴` failure, `⚠️` degraded.
- Catch exceptions inside the tool and return the error string — do not let it raise out to the MCP layer.
- Never `print()`. stdout is the MCP stdio channel. Use `logger.info` / `logger.warning` (already
  configured to stderr).
- Truncate large payloads before returning (existing tools slice to ~1000 chars).

### If the tool can make a service exist or stop existing

`get_vllm_url()` memoizes each service's URL in the module-level `_VLLM_URL_CACHE`, and nothing expires
it. A cached URL survives the service it points at, so a tool that **deletes or creates** a service
leaves later calls aimed at a dead or wrong endpoint — and the failure surfaces in some unrelated query
much later, which is miserable to trace. Call `invalidate_vllm_url_cache(service_name)` there, on the
failure path as well as the success path, since a partly-completed delete or deploy still changes what
exists. `cloudrun_destroy` and `cloudrun_deploy` both do this — read either and copy the placement.

Be precise about which tools need it, rather than adding it reflexively:

- **Needs it**: deleting a service, deploying/recreating one, anything that could change the resolved URL.
- **Does not**: shifting traffic between revisions of a service that already exists (the service URL is
  stable across revisions), scaling changes, and every read-only tool. Adding it there is harmless but
  misleading — it implies a URL change that cannot happen.

## 3. Write the test

In `test_agent.py`:

1. Add the tool name to the `EXPECTED_TOOLS` set. This is required — `test_tools_registered` compares
   that set to the live registry with `assertEqual`, so a new tool fails the suite until it is listed.
2. Add a unit test in `TestDevOpsAgent` following the neighbouring style:
   - `@patch("server.subprocess.run")` for gcloud tools; assert on the constructed `cmd` list
     (`mock_run.call_args`), not just the return string.
   - `@patch("server.get_vllm_client")` + `@patch("server.get_active_model_name")` for inference tools.
   - `@patch("server.httpx.AsyncClient")` for raw HTTP tools.
   - Async tests are plain `async def` methods (the class is `unittest.IsolatedAsyncioTestCase`).
   - Tests must not require GCP credentials or network.
3. If the test touches URL discovery at all — calling `get_vllm_url`, patching `server.discover_vllm_url`,
   or exercising a tool that invalidates the cache — bracket it with `server.invalidate_vllm_url_cache()`
   at the start and end. There is deliberately no `setUp` doing this globally, so a URL cached by an
   earlier test would otherwise satisfy your lookup and the test would pass while asserting nothing.
   `test_get_vllm_url_caches_per_service` and `test_destroy_cloudrun_invalidates_url_cache` show the shape.

Where the tool invalidates the cache, prove it rather than trusting it: prime the cache with a patched
`discover_vllm_url`, run the tool, and assert the next lookup re-discovers.

## 4. Verify

```bash
make lint && make test
```

Both must pass. Fix anything ruff or mypy flags rather than adding ignores.

A `PostToolUse` hook already runs `ruff format` + `ruff check --fix` on every `.py` file you edit through
Write/Edit, so formatting is normally settled before you get here — a `make lint` failure is usually mypy
or a ruff rule with no autofix, and is worth reading rather than reformatting blindly. Edits made another
way (a `sed` one-liner, a heredoc) skip the hook; run `ruff format .` yourself in that case.

If mypy reports an error inside a third-party stub rather than in `server.py` or `test_agent.py`, that is
an environment or config problem, not something to paper over with a `type: ignore` — mypy aborts before
checking project code when a stub fails to parse, so fix the config and re-run.

## 5. Update docs

The tool name is hand-listed in three places outside the code, and no test checks any of them —
`EXPECTED_TOOLS` is the only enforced listing, so these drift silently:

- the `cloudrun_get_help` tool's body in `server.py`, which enumerates every tool under a category
  heading — this is the listing callers actually see at runtime, and the easiest to forget
- the "Available Tools" section of `README.md`
- the capabilities list in `GEMINI.md`

Add the tool to the category matching its shape from step 1, wording it the same way in each.
