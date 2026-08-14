---
title: "Build an MCP server in Rust with rmcp: a walk-through 🦀"
published: false
description: "Step-by-step: scaffolding a real MCP server in Rust with the official rmcp SDK — tools, JSON schemas, AWS calls, stdio transport, testing the protocol by hand, and wiring it into Claude Code."
tags: rust, mcp, aws, ai
cover_image: https://raw.githubusercontent.com/xbill9/gemma4-dev/main/gpu-vllm-g5g-2b/images/header-rust-mcp-rmcp.jpg
series: "Rust on the G5g rig"
---

This tutorial walks through building an **MCP server in Rust** with
[`rmcp`](https://crates.io/crates/rmcp), the official Model Context Protocol Rust SDK.

The example is a real one: a devops agent that manages **AWS EC2 G5g** instances — Graviton2
boxes with NVIDIA T4G GPUs — serving Gemma 4 under vLLM. It launches instances, drives them
over SSM, and health-checks the model. There's an existing Python version, so at the end we
can put the two side by side.

Follow along and you'll have a working, registerable MCP server. 🦀

---

#### Why Rust for this?

Worth answering properly, because the weak version of the argument is easy to make and easy to
demolish — and the real one is better anyway.

**Start with what it isn't: these tools are I/O bound.** Every one is an AWS API call —
`describe_instances`, `send_command`, polling SSM — so 100–500 ms of network per call. The
caller's language contributes nothing measurable there. Anyone selling you a Rust rewrite on
raw speed for this workload is selling something.

Three claims that don't hold, so nobody has to make them in the comments:

| Claim | Why it fails |
| --- | --- |
| "462 ms startup is slow" | stdio servers spawn **once per session**, not per call |
| "Rust is faster" | the work is network round-trips to AWS |
| "smaller supply chain" | 241 crates vs 34 Python packages — it's *worse* |

What actually justifies it, for this codebase:

**1. It's a fleet, not a server.** This monorepo has **16 rigs**, each with its own MCP
server. That changes the units:

| All loaded together | 🐍 Python | 🦀 Rust |
| --- | --- | --- |
| Resident memory | 16 × 83 MB ≈ **1.33 GB** | 16 × 12 MB ≈ **192 MB** |
| Session startup | 16 × 462 ms ≈ **7.4 s** | 16 × 2.5 ms ≈ **40 ms** |

A gigabyte of resident Python to expose sixteen tool lists is a real cost.

**2. No shared interpreter.** These rigs install system-wide — no virtualenvs, by policy — so
all sixteen share one Python. Sixteen servers with independently drifting `boto3` and `mcp`
pins in one interpreter is a standing conflict risk. A static binary has no such coupling;
each rig pins whatever it likes in its own `Cargo.lock`.

**3. The schema can't drift from the code.** More on this at Step 3, but it's the one that
survives longest: `schemars` generates the tool schema from the same struct the handler
destructures.

So: **distribution and correctness, not speed.** ✅ If you have one MCP server and it works,
this is not a reason to rewrite it.

---

#### How does this all fit together?

Two halves. The agent and the MCP server run on your machine; the GPU box is remote, and it
has **no inbound SSH** — everything goes through the AWS APIs.

```
   YOUR MACHINE                                     AWS  us-east-1
┌──────────────────────────────┐       ┌───────────────────────────────────────┐
│                              │       │                                       │
│  Claude Code / IDE           │       │  ┌─ EC2 g5g.4xlarge ───────────────┐  │
│         |                    │       │  │  Graviton2 (aarch64)            │  │
│         | MCP · JSON-RPC 2.0 │       │  │  + NVIDIA T4G (SM 7.5)          │  │
│         | over stdio         │       │  │                                 │  │
│         v                    │  EC2  │  │  [PY] vLLM + [RUST] vllm-rs     │  │
│  ┌────────────────────────┐  │  API  │  │  listening on :8000             │  │
│  │ [RUST]                 │──┼──────>│  │                                 │  │
│  │ gpu-vllm-g5g-2b        │  │       │  │  Gemma 4 E2B                    │  │
│  │                        │  │  SSM  │  └─────────────────────────────────┘  │
│  │ rmcp 3.1.2             │──┼──────>│           ^                           │
│  │ tokio · schemars       │  │  Run  │           |  no inbound SSH,          │
│  │ aws-sdk-ec2 / -ssm     │  │  Cmd  │           |  no key pair,             │
│  │ 1 binary · 2.5 ms      │  │       │           |  no port 22 rule          │
│  └────────────────────────┘  │       │                                       │
│        9 tools               │       │  IAM instance profile carries         │
│  list / start / stop /       │       │  AmazonSSMManagedInstanceCore         │
│  terminate / endpoint /      │       │                                       │
│  run_remote / health ...     │       │                                       │
└──────────────────────────────┘       └───────────────────────────────────────┘
```

The agent never talks to the GPU box directly. It calls a tool; the tool calls **EC2** to
manage the instance's lifecycle, or **SSM Run Command** to execute something on it. That's
what lets the box run with no inbound rules at all — which is the main reason this is worth
building as a server rather than a pile of shell scripts.

The `[RUST]` on the right-hand side is vLLM's own Rust frontend — the other article in this
series. This one is the `[RUST]` on the left: the Rust that drives the box.

---

#### What is MCP, in one paragraph?

**Model Context Protocol** is how an AI agent discovers and calls your tools. Your server
advertises a list of tools with JSON Schemas; the client (Claude Code, an IDE, whatever)
calls them over JSON-RPC 2.0. Transport is usually **stdio** — the client spawns your binary
and talks over stdin/stdout.

That last detail matters for the Rust pitch: if the client spawns your process on every
session, **process startup is a user-visible cost**.

---

#### Step 1 — Scaffold

```bash
cargo new --bin rust-mcp --name gpu-vllm-g5g-2b-mcp
cd rust-mcp
```

Now the dependencies. **Feature flags are the thing to get right here** — `cargo add rmcp`
on its own compiles fine and gives you almost nothing:

```bash
cargo add rmcp --features server,macros,transport-io
```

| Feature | What it brings |
| --- | --- |
| `server` | the `ServerHandler` trait and router types |
| `macros` | `#[tool]`, `#[tool_router]`, `#[tool_handler]` |
| `transport-io` | stdio transport |

The crate also ships `client`, `auth`, `elicitation`, `transport-streamable-http-server` and
more, all off by default. Add them when you need them.

Then the rest:

```bash
cargo add tokio --features rt-multi-thread,macros,process,time
cargo add serde serde_json anyhow schemars
cargo add aws-config aws-sdk-ec2 aws-sdk-ssm aws-sdk-secretsmanager
```

Resulting `Cargo.toml`:

```toml
[package]
name = "gpu-vllm-g5g-2b-mcp"
version = "0.1.0"
edition = "2024"

[dependencies]
rmcp = { version = "3.1.2", features = ["server", "macros", "transport-io"] }
tokio = { version = "1.53.1", features = ["rt-multi-thread", "macros", "process", "time"] }
aws-config = "1.10.1"
aws-sdk-ec2 = "1.246.0"
aws-sdk-ssm = "1.118.0"
serde = "1.0.229"
serde_json = "1.0.151"
schemars = "1.2.2"
anyhow = "1.0.104"
```

---

#### 🔎 Tip: where the canonical examples live

`rmcp` moves fast, and rendered docs lag. The **vendored tests on your own disk** are
compiled against the exact version you resolved:

```bash
ls ~/.cargo/registry/src/*/rmcp-3.1.2/tests/
```

`tests/test_tool_macros.rs` is a complete, working server in about 60 lines. When an API
question comes up, that file answers it faster and more reliably than anything else. ⚡

---

#### Step 2 — The server struct

An rmcp server is a struct that owns a `ToolRouter`:

```rust
use rmcp::{
    ErrorData, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::stdio,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
struct G5gServer {
    tool_router: ToolRouter<Self>,
}
```

---

#### Step 3 — Describe your inputs as types

This is the part that sold me on the whole exercise. Your tool's input is a plain struct, and
**`schemars` turns it into the JSON Schema the agent sees** — doc comments and all:

```rust
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct InstanceId {
    /// EC2 instance id, e.g. `i-0123456789abcdef0`.
    instance_id: String,
}
```

That doc comment becomes the field's `description` in the tool schema. Rename the field and
the schema follows. The compiler checks the type your handler destructures. There is no
second artifact to keep in sync. ✅

---

#### Step 4 — Write the tools

`#[tool_router]` on the impl block, `#[tool]` on each method:

```rust
#[tool_router(router = tool_router)]
impl G5gServer {
    fn new() -> Self {
        Self { tool_router: Self::tool_router() }
    }

    #[tool(description = "List EC2 instances tagged ManagedBy=gpu-vllm-g5g-2b.")]
    async fn list_g5g_instances(&self) -> Result<CallToolResult, ErrorData> {
        let conf = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(aws_config::Region::new("us-east-1"))
            .load()
            .await;
        let ec2 = aws_sdk_ec2::Client::new(&conf);

        let resp = match ec2.describe_instances()
            .filters(Filter::builder()
                .name("tag:ManagedBy").values("gpu-vllm-g5g-2b").build())
            .send().await
        {
            Ok(r) => r,
            Err(e) => return ok(format!("❌ describe_instances failed: {e}")),
        };

        let mut rows = Vec::new();
        for res in resp.reservations() {
            for inst in res.instances() {
                rows.push(format!("| `{}` | {} | {} |",
                    inst.instance_id().unwrap_or("?"),
                    inst.instance_type().map(|t| t.as_str()).unwrap_or("?"),
                    inst.state().and_then(|s| s.name())
                        .map(|n| n.as_str()).unwrap_or("unknown"),
                ));
            }
        }
        ok(format!("📡 Instances\n\n| id | type | state |\n|---|---|---|\n{}",
                   rows.join("\n")))
    }
}
```

Tools that take arguments wrap them in `Parameters<T>`:

```rust
    #[tool(description = "Terminate a G5g instance. Permanent — destroys the root volume.")]
    async fn terminate_g5g_instance(
        &self,
        Parameters(args): Parameters<InstanceId>,
    ) -> Result<CallToolResult, ErrorData> {
        // …
    }
```

And a small helper, since every tool returns the same shape:

```rust
fn ok(text: String) -> Result<CallToolResult, ErrorData> {
    Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
}
```

---

#### Step 5 — Implement ServerHandler

`#[tool_handler]` wires the router in, so you never write a dispatch `match`:

```rust
#[tool_handler(router = self.tool_router)]
impl ServerHandler for G5gServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::new(
            ServerCapabilities::builder().enable_tools().build()
        );
        info.server_info = Implementation::new(
            "gpu-vllm-g5g-2b", env!("CARGO_PKG_VERSION")
        );
        info.instructions = Some(
            "Devops agent for AWS EC2 G5g (Graviton2 + NVIDIA T4G) serving Gemma 4 \
             under vLLM. Remote administration goes through SSM; there is no inbound SSH."
                .to_string(),
        );
        info
    }
}
```

💡 These model structs are `#[non_exhaustive]`, so use the constructors (`ServerInfo::new`,
`Implementation::new`) and then assign fields — a struct literal won't compile, even with
`..Default::default()`.

---

#### Step 6 — main

Four lines:

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let service = G5gServer::new().serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
```

```bash
cargo build --release
```

---

#### Step 7 — Test the protocol by hand

An MCP server is a protocol implementation, so test it with a protocol transcript. Three
JSON-RPC lines on stdin — no client required:

```bash
printf '%s\n' \
'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2026-07-28","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
'{"jsonrpc":"2.0","method":"notifications/initialized"}' \
'{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_help","arguments":{}}}' \
| ./target/release/gpu-vllm-g5g-2b-mcp
```

```
initialize OK: gpu-vllm-g5g-2b 0.1.0 proto 2026-07-28
tools/list OK: 9 tools -> get_deployment_config, get_endpoint, get_help,
  list_g5g_instances, run_remote, start_g5g_instance, stop_g5g_instance,
  terminate_g5g_instance, verify_model_health
tools/call get_help OK
```

🟢 rmcp 3.1.2 negotiates the **2026-07-28** spec version by default.

Keep this snippet. It's the fastest way to tell "my server is broken" from "my client config
is broken."

---

#### Step 8 — Register it

Point your MCP client at the binary. For Claude Code, `.mcp.json`:

```json
{
  "mcpServers": {
    "gpu-vllm-g5g-2b": {
      "command": "/abs/path/to/rust-mcp/target/release/gpu-vllm-g5g-2b-mcp",
      "env": { "AWS_REGION": "us-east-1" }
    }
  }
}
```

The server name prefixes every tool — `mcp__gpu-vllm-g5g-2b__list_g5g_instances` — so name it
after the thing it manages, especially if you run several.

Credentials come from the standard AWS provider chain, so whatever
`aws sts get-caller-identity` resolves is what the server gets. Set `AWS_PROFILE` to pick one.

---

#### Tests worth writing

The interesting assertions aren't about the code, they're about the **machine**. Turing has no
bf16 datapath and no fp8, so the serving flags must differ from every L4-class box — exactly
the sort of thing that silently reverts when someone copies a flag set from a neighbour:

```rust
#[test]
fn serve_flags_are_turing_shaped() {
    let f = serve_flags("google/gemma-4-E2B-it", "g5g.2xlarge");
    assert!(f.contains("--dtype float16"), "Turing has no bf16 datapath");
    assert!(f.contains("--kv-cache-dtype auto"), "Turing has no fp8 datapath");
    assert!(!f.contains("attention-backend"));   // not a real vLLM v0.27 variable
}

#[test]
fn unknown_types_are_rejected_and_never_need_swap() {
    assert!(validate_instance_type("t4g.2xlarge").is_err());  // burstable CPU box, no GPU
    assert!(!needs_swap("t4g.2xlarge"));                      // 0 GiB must not read as "tiny"
}
```

That second one earns its keep: `host_memory_gb` returns `0` for an unknown instance type,
and a naive `ram < 16` would decide an unrecognised machine needs a swapfile.

```
running 5 tests
test result: ok. 5 passed; 0 failed; finished in 0.00s
```

---

#### 🐍 vs 🦀 — the scoreboard

Cold start measured the way a client experiences it: spawn the process, send `initialize` +
`initialized` + `tools/list`, stop the clock when the tool list comes back. Seven runs, median.

| | 🐍 Python (FastMCP) | 🦀 Rust (rmcp) |
| --- | --- | --- |
| Cold start to `tools/list` | **462.0 ms** (437–530) | **2.5 ms** (1.8–2.9) |
| Peak RSS | 83 MB | **12 MB** |
| Artifact | Python runtime + 34 packages | one binary, 39.4 MB (**19.9 MB** stripped) |
| Direct dependencies | **3** | 14 |
| Total resolved packages | **34** | 241 |
| Tools implemented | **15** | 9 |
| Source | 759 lines | 560 + 52 of tests |
| Clean release build | **n/a** | 5 m 28 s |

**185x on cold start** — but per the section up top, resist quoting that on its own. One
server spawned once a session, 460 ms, nobody notices. It only becomes a number worth having
when you multiply it by sixteen rigs, and even then it's the **12 MB vs 83 MB** row that does
the heavier lifting.

Read the rest honestly too. 241 resolved packages against 34 means the static binary is
**not** a smaller supply chain, just the same one audited in `Cargo.lock`. The port covers 9
tools to Python's 15 — the provisioning path (cloud-init rendering, AMI resolution, spot
options) is the fiddly half and isn't ported. And 5 m 28 s of clean build against an
interpreter that starts instantly is a real cost while you're iterating. 📊

---

#### So, worth it?

For a single MCP server that already works: **no**. Don't rewrite it.

For sixteen of them sharing one system Python, shipped to machines that shouldn't need a
Python environment at all: yes — and note that neither half of that sentence is about speed.
It's a packaging answer.

The part that'll still be true next year is `schemars`. The tool schema the agent sees is
generated from the same struct the handler destructures, checked by the compiler, documented
by the doc comments on its fields. In the Python version the schema, the runtime types and the
docs are three artifacts that agree by convention — and go quiet when they stop agreeing.

The 462 ms is a bonus. The schema not being able to lie about the code is the reason. ✅

---

#### Cheat sheet

```bash
# scaffold
cargo new --bin my-mcp && cd my-mcp
cargo add rmcp --features server,macros,transport-io
cargo add tokio --features rt-multi-thread,macros
cargo add serde serde_json schemars anyhow

# canonical examples for YOUR resolved version
ls ~/.cargo/registry/src/*/rmcp-*/tests/test_tool_macros.rs

# build + smoke test
cargo build --release
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2026-07-28","capabilities":{},"clientInfo":{"name":"p","version":"0"}}}' \
              '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
              '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  | ./target/release/my-mcp
```

Four macros to remember: `#[tool_router]` on the impl, `#[tool]` on each method,
`#[tool_handler]` on the `ServerHandler` impl, `Parameters<T>` around your input struct.

---

*Rust 1.97.1, rmcp 3.1.2, aws-sdk-ec2 1.246.0, edition 2024. Startup measured on the dev host
— it's a comparison of two MCP servers, not a hardware result. Single machine, seven runs per
side, median reported.*
