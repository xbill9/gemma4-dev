//! gpu-vllm-g5g-2b — Rust MCP server.
//!
//! A port of this rig's single-file FastMCP `server.py` onto the official Rust
//! MCP SDK (`rmcp`). Single file on purpose: the Python original is one file,
//! and keeping the shape identical is what makes the two comparable.
//!
//! The rules from the rig's CLAUDE.md carry over unchanged:
//!   * the AWS SDK and its standard credential provider chain — never shell out
//!     to the `aws` CLI;
//!   * SSM Run Command for remote administration; no inbound SSH, no key pair;
//!   * subnet, security group and instance profile are required arguments — this
//!     server creates no network or IAM policy of its own;
//!   * instance discovery is scoped to `ManagedBy=gpu-vllm-g5g-2b`;
//!   * never hardcode an endpoint or an AMI id — both are resolved at runtime.

use anyhow::Result;
use aws_sdk_ec2::types::Filter;
use rmcp::{
    ErrorData, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::stdio,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration. Mirrors tpu.env, which is the rig's source of truth. A real
// environment variable always wins, exactly as the Python server does it.
// ---------------------------------------------------------------------------

const RIG_NAME: &str = "gpu-vllm-g5g-2b";

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn aws_region() -> String {
    env_or("AWS_REGION", "us-east-1")
}
fn model_name() -> String {
    env_or("MODEL_NAME", "google/gemma-4-E2B-it")
}
fn instance_type_default() -> String {
    env_or("INSTANCE_TYPE", "g5g.2xlarge")
}
fn dtype() -> String {
    env_or("DTYPE", "float16")
}
fn kv_cache_dtype() -> String {
    env_or("KV_CACHE_DTYPE", "auto")
}
fn gpu_memory_utilization() -> String {
    env_or("GPU_MEMORY_UTILIZATION", "0.90")
}
fn max_model_len() -> String {
    env_or("MAX_MODEL_LEN", "16384")
}
fn max_num_seqs() -> String {
    env_or("MAX_NUM_SEQS", "8")
}
fn vllm_port() -> String {
    env_or("VLLM_PORT", "8000")
}

// ---------------------------------------------------------------------------
// Instance sizing. (gpu_count, host_ram_gb) per G5g size.
//
// G5g is the only Graviton+GPU family AWS ships. One T4G on every size except
// 16xlarge/metal, which carry two — that is topology, not a sharding choice, so
// tensor-parallel size is derived from it rather than configured.
// ---------------------------------------------------------------------------

const G5G_SIZES: &[(&str, u32, u32)] = &[
    ("g5g.xlarge", 1, 8),
    ("g5g.2xlarge", 1, 16),
    ("g5g.4xlarge", 1, 32),
    ("g5g.8xlarge", 1, 64),
    ("g5g.16xlarge", 2, 128),
    ("g5g.metal", 2, 128),
];

/// Host RAM below this needs a swapfile before the model will load. Measured
/// 2026-08-13 on g5g.xlarge (7,757 MiB usable): loading E2B fails with
///
///   RuntimeError: unable to mmap 10246621918 bytes from model.safetensors:
///   Cannot allocate memory (12)
///
/// The failure is the *mapping*, not residency — the kernel declines to map a
/// 10.2 GB file against 7.5 GiB of RAM and no swap, before a single page is
/// faulted in. 16 GiB of swap took the same instance to 44.24 tok/s, which is
/// indistinguishable from the 4xlarge's 43.1: decode is GPU-bandwidth-bound, so
/// vCPU count barely matters once the weights are resident.
const SWAP_BELOW_HOST_RAM_GB: u32 = 16;

fn size_entry(instance_type: &str) -> Option<(u32, u32)> {
    G5G_SIZES
        .iter()
        .find(|(name, _, _)| *name == instance_type)
        .map(|(_, gpus, ram)| (*gpus, *ram))
}

fn is_g5g(instance_type: &str) -> bool {
    size_entry(instance_type).is_some()
}

fn gpu_count(instance_type: &str) -> u32 {
    size_entry(instance_type).map(|(g, _)| g).unwrap_or(0)
}

fn host_memory_gb(instance_type: &str) -> u32 {
    size_entry(instance_type).map(|(_, r)| r).unwrap_or(0)
}

/// True when host RAM is too small to mmap the checkpoint without swap.
fn needs_swap(instance_type: &str) -> bool {
    let ram = host_memory_gb(instance_type);
    ram > 0 && ram < SWAP_BELOW_HOST_RAM_GB
}

/// Only the size list is enforced. Small hosts are supported, not rejected —
/// user data provisions a swapfile for them.
fn validate_instance_type(instance_type: &str) -> Result<(), String> {
    if is_g5g(instance_type) {
        return Ok(());
    }
    let mut names: Vec<&str> = G5G_SIZES.iter().map(|(n, _, _)| *n).collect();
    names.sort_unstable();
    Err(format!(
        "instance_type must be one of {}",
        names.join(", ")
    ))
}

fn tensor_parallel_size(instance_type: &str) -> u32 {
    gpu_count(instance_type)
}

/// vLLM flags for Turing. Deliberately unlike the L4 rigs' flag set: SM 7.5 has
/// no bf16 datapath and no fp8, so `--dtype float16` and `--kv-cache-dtype auto`
/// are what actually execute. There is deliberately no attention-backend flag —
/// vLLM v0.27 does not recognise `VLLM_ATTENTION_BACKEND` and forces TRITON_ATTN
/// for Gemma 4's heterogeneous head dims regardless.
fn serve_flags(model: &str, instance_type: &str) -> String {
    format!(
        "--model {model} --host 0.0.0.0 --port {port} \
         --dtype {dtype} --kv-cache-dtype {kv} \
         --tensor-parallel-size {tps} \
         --gpu-memory-utilization {util} \
         --max-model-len {mml} --max-num-seqs {mns}",
        port = vllm_port(),
        dtype = dtype(),
        kv = kv_cache_dtype(),
        tps = tensor_parallel_size(instance_type),
        util = gpu_memory_utilization(),
        mml = max_model_len(),
        mns = max_num_seqs(),
    )
}

// ---------------------------------------------------------------------------
// AWS helpers. Standard provider chain — whatever `aws sts get-caller-identity`
// resolves is what this server gets.
// ---------------------------------------------------------------------------

async fn aws_config() -> aws_config::SdkConfig {
    let region = aws_config::Region::new(aws_region());
    aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(region)
        .load()
        .await
}

fn managed_filter() -> Filter {
    Filter::builder()
        .name("tag:ManagedBy")
        .values(RIG_NAME)
        .build()
}

/// Render an error the way the Python server does, so transcripts read alike.
fn err_text(context: &str, e: impl std::fmt::Display) -> String {
    format!("❌ {context}: {e}")
}

fn ok(text: String) -> Result<CallToolResult, ErrorData> {
    Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
}

// ---------------------------------------------------------------------------
// Tool parameter types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct InstanceId {
    /// EC2 instance id, e.g. `i-0123456789abcdef0`.
    instance_id: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DeploymentConfigArgs {
    /// Subnet to launch into. Required — this server creates no networking.
    subnet_id: String,
    /// Security group id. Required — this server creates no networking.
    security_group_id: String,
    /// IAM instance profile name. Needs AmazonSSMManagedInstanceCore.
    iam_instance_profile: String,
    /// G5g size. Defaults to INSTANCE_TYPE from the environment.
    #[serde(default)]
    instance_type: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct RunRemoteArgs {
    /// EC2 instance id.
    instance_id: String,
    /// Shell command to run through SSM Run Command.
    command: String,
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct G5gServer {
    tool_router: ToolRouter<Self>,
}

#[tool_router(router = tool_router)]
impl G5gServer {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    /// Help and configuration for this rig.
    #[tool(description = "Show this rig's configuration and the Turing/aarch64 constraints that shape it.")]
    async fn get_help(&self) -> Result<CallToolResult, ErrorData> {
        let it = instance_type_default();
        ok(format!(
            "# {RIG_NAME}\n\n\
             Serving **{model}** under vLLM on AWS EC2 G5g — a Graviton2 (aarch64) host \
             with an NVIDIA T4G (Turing, SM 7.5, 15,360 MiB).\n\n\
             | setting | value |\n|---|---|\n\
             | region | `{region}` |\n\
             | default instance type | `{it}` |\n\
             | dtype | `{dtype}` |\n\
             | kv-cache-dtype | `{kv}` |\n\
             | tensor-parallel-size | `{tps}` (derived from GPU count) |\n\
             | needs swapfile | `{swap}` |\n\n\
             Turing has no bf16 datapath and no fp8, so the fp8 KV cache that is standard \
             advice on L4 and A10G buys nothing here. vLLM forces the TRITON_ATTN backend \
             for Gemma 4's heterogeneous head dims (sliding 256 / global 512) and \
             `VLLM_ATTENTION_BACKEND` is not a recognised variable in v0.27.\n\n\
             Serve flags:\n\n```\n{flags}\n```\n",
            model = model_name(),
            region = aws_region(),
            dtype = dtype(),
            kv = kv_cache_dtype(),
            tps = tensor_parallel_size(&it),
            swap = needs_swap(&it),
            flags = serve_flags(&model_name(), &it),
        ))
    }

    /// Generate a deployment configuration without launching anything.
    #[tool(description = "Generate a G5g deployment configuration (no AWS calls, launches nothing).")]
    async fn get_deployment_config(
        &self,
        Parameters(args): Parameters<DeploymentConfigArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let it = args
            .instance_type
            .unwrap_or_else(instance_type_default);
        if let Err(e) = validate_instance_type(&it) {
            return ok(err_text("invalid instance type", e));
        }
        ok(format!(
            "## Deployment configuration\n\n\
             | field | value |\n|---|---|\n\
             | instance type | `{it}` |\n\
             | GPUs | {gpus} |\n\
             | host RAM | {ram} GiB |\n\
             | swapfile required | {swap} |\n\
             | tensor-parallel-size | {tps} |\n\
             | subnet | `{subnet}` |\n\
             | security group | `{sg}` |\n\
             | instance profile | `{prof}` |\n\
             | AMI | resolved at launch (arm64 + NVIDIA driver) |\n\n\
             Serve flags:\n\n```\n{flags}\n```\n",
            gpus = gpu_count(&it),
            ram = host_memory_gb(&it),
            swap = needs_swap(&it),
            tps = tensor_parallel_size(&it),
            subnet = args.subnet_id,
            sg = args.security_group_id,
            prof = args.iam_instance_profile,
            flags = serve_flags(&model_name(), &it),
        ))
    }

    /// List instances this rig manages.
    #[tool(description = "List EC2 instances tagged ManagedBy=gpu-vllm-g5g-2b.")]
    async fn list_g5g_instances(&self) -> Result<CallToolResult, ErrorData> {
        let conf = aws_config().await;
        let ec2 = aws_sdk_ec2::Client::new(&conf);
        let resp = match ec2
            .describe_instances()
            .filters(managed_filter())
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return ok(err_text("describe_instances failed", e)),
        };

        let mut rows = Vec::new();
        for res in resp.reservations() {
            for inst in res.instances() {
                let state = inst
                    .state()
                    .and_then(|s| s.name())
                    .map(|n| n.as_str().to_string())
                    .unwrap_or_else(|| "unknown".into());
                if state == "terminated" {
                    continue;
                }
                rows.push(format!(
                    "| `{}` | {} | {} | {} | {} |",
                    inst.instance_id().unwrap_or("?"),
                    inst.instance_type()
                        .map(|t| t.as_str())
                        .unwrap_or("?"),
                    state,
                    inst.public_ip_address().unwrap_or("-"),
                    inst.placement()
                        .and_then(|p| p.availability_zone())
                        .unwrap_or("-"),
                ));
            }
        }

        if rows.is_empty() {
            return ok(format!("📡 No instances tagged `ManagedBy={RIG_NAME}`."));
        }
        ok(format!(
            "📡 Instances managed by `{RIG_NAME}`\n\n\
             | id | type | state | public ip | az |\n|---|---|---|---|---|\n{}\n",
            rows.join("\n")
        ))
    }

    /// Start a stopped instance.
    #[tool(description = "Start a stopped G5g instance.")]
    async fn start_g5g_instance(
        &self,
        Parameters(args): Parameters<InstanceId>,
    ) -> Result<CallToolResult, ErrorData> {
        let conf = aws_config().await;
        let ec2 = aws_sdk_ec2::Client::new(&conf);
        match ec2
            .start_instances()
            .instance_ids(&args.instance_id)
            .send()
            .await
        {
            Ok(_) => ok(format!("✅ Starting `{}`.", args.instance_id)),
            Err(e) => ok(err_text("start_instances failed", e)),
        }
    }

    /// Stop an instance. Stop preserves the root volume; terminate does not.
    #[tool(description = "Stop a running G5g instance (preserves the root volume).")]
    async fn stop_g5g_instance(
        &self,
        Parameters(args): Parameters<InstanceId>,
    ) -> Result<CallToolResult, ErrorData> {
        let conf = aws_config().await;
        let ec2 = aws_sdk_ec2::Client::new(&conf);
        match ec2
            .stop_instances()
            .instance_ids(&args.instance_id)
            .send()
            .await
        {
            Ok(_) => ok(format!(
                "✅ Stopping `{}`. The root volume survives; an idle 80 GiB volume \
                 is about $6.40/month, so weigh this against terminating.",
                args.instance_id
            )),
            Err(e) => ok(err_text("stop_instances failed", e)),
        }
    }

    /// Terminate an instance. Permanent, and destroys the root volume with it.
    #[tool(description = "Terminate a G5g instance. Permanent — destroys the root volume.")]
    async fn terminate_g5g_instance(
        &self,
        Parameters(args): Parameters<InstanceId>,
    ) -> Result<CallToolResult, ErrorData> {
        let conf = aws_config().await;
        let ec2 = aws_sdk_ec2::Client::new(&conf);
        match ec2
            .terminate_instances()
            .instance_ids(&args.instance_id)
            .send()
            .await
        {
            Ok(_) => ok(format!(
                "✅ Terminating `{}`. This is permanent.",
                args.instance_id
            )),
            Err(e) => ok(err_text("terminate_instances failed", e)),
        }
    }

    /// Resolve the serving endpoint from the instance. Never hardcoded.
    #[tool(description = "Resolve this rig's vLLM endpoint from the instance's current address.")]
    async fn get_endpoint(
        &self,
        Parameters(args): Parameters<InstanceId>,
    ) -> Result<CallToolResult, ErrorData> {
        let conf = aws_config().await;
        let ec2 = aws_sdk_ec2::Client::new(&conf);
        let resp = match ec2
            .describe_instances()
            .instance_ids(&args.instance_id)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return ok(err_text("describe_instances failed", e)),
        };
        for res in resp.reservations() {
            for inst in res.instances() {
                if let Some(ip) = inst.public_ip_address() {
                    return ok(format!(
                        "📡 `http://{ip}:{port}` (chat: `/v1/chat/completions`)",
                        port = vllm_port()
                    ));
                }
            }
        }
        ok(format!(
            "❌ No public address on `{}` yet.",
            args.instance_id
        ))
    }

    /// Run a command on the instance through SSM. No inbound SSH anywhere.
    #[tool(description = "Run a shell command on the instance via SSM Run Command (no SSH).")]
    async fn run_remote(
        &self,
        Parameters(args): Parameters<RunRemoteArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let conf = aws_config().await;
        let ssm = aws_sdk_ssm::Client::new(&conf);
        let sent = ssm
            .send_command()
            .instance_ids(&args.instance_id)
            .document_name("AWS-RunShellScript")
            .parameters("commands", vec![args.command.clone()])
            .send()
            .await;
        let command_id = match sent {
            Ok(r) => match r.command().and_then(|c| c.command_id().map(str::to_string)) {
                Some(id) => id,
                None => return ok("❌ SSM returned no command id.".to_string()),
            },
            Err(e) => return ok(err_text("send_command failed", e)),
        };

        // Poll to a terminal state.
        for _ in 0..120 {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            let inv = ssm
                .get_command_invocation()
                .command_id(&command_id)
                .instance_id(&args.instance_id)
                .send()
                .await;
            let inv = match inv {
                Ok(i) => i,
                Err(_) => continue, // invocation may not exist yet
            };
            let status = inv
                .status()
                .map(|s| s.as_str().to_string())
                .unwrap_or_default();
            if matches!(
                status.as_str(),
                "Success" | "Failed" | "Cancelled" | "TimedOut"
            ) {
                let out = inv.standard_output_content().unwrap_or("");
                let errout = inv.standard_error_content().unwrap_or("");
                let mark = if status == "Success" { "✅" } else { "❌" };
                let mut body = format!("{mark} SSM `{status}`\n\n```\n{out}\n```");
                if !errout.is_empty() {
                    body.push_str(&format!("\nstderr:\n```\n{errout}\n```"));
                }
                return ok(body);
            }
        }
        ok(format!(
            "❌ SSM command `{command_id}` did not reach a terminal state in time."
        ))
    }

    /// Health-check the model through the chat endpoint.
    #[tool(description = "Verify model health via /v1/chat/completions on the instance itself.")]
    async fn verify_model_health(
        &self,
        Parameters(args): Parameters<InstanceId>,
    ) -> Result<CallToolResult, ErrorData> {
        // Deliberately chat, not raw completions. Measured on this rig
        // 2026-08-12: raw /v1/completions returns degenerate repetition
        // (': ok: ok: ok…'), so an emptiness check would pass on garbage.
        let probe = format!(
            "curl -s -m 60 localhost:{port}/v1/chat/completions \
             -H 'Content-Type: application/json' \
             -d '{{\"model\":\"{model}\",\"messages\":[{{\"role\":\"user\",\
             \"content\":\"Reply with the single word: healthy\"}}],\
             \"max_tokens\":16,\"temperature\":0}}'",
            port = vllm_port(),
            model = model_name(),
        );
        self.run_remote(Parameters(RunRemoteArgs {
            instance_id: args.instance_id,
            command: probe,
        }))
        .await
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for G5gServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::new(ServerCapabilities::builder().enable_tools().build());
        info.server_info = Implementation::new(RIG_NAME, env!("CARGO_PKG_VERSION"));
        info.instructions = Some(
            "Devops agent for the gpu-vllm-g5g-2b rig: AWS EC2 G5g (Graviton2 + NVIDIA \
             T4G, SM 7.5) serving Gemma 4 E2B under vLLM. Provisioning requires explicit \
             subnet, security-group and instance-profile ids. Remote administration goes \
             through SSM Run Command; there is no inbound SSH."
                .to_string(),
        );
        info
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let service = G5gServer::new().serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_size_has_at_least_one_gpu() {
        for (name, gpus, _) in G5G_SIZES {
            assert!(*gpus >= 1, "{name} has no GPU");
        }
    }

    #[test]
    fn tensor_parallel_follows_gpu_count() {
        assert_eq!(tensor_parallel_size("g5g.xlarge"), 1);
        assert_eq!(tensor_parallel_size("g5g.2xlarge"), 1);
        assert_eq!(tensor_parallel_size("g5g.4xlarge"), 1);
        assert_eq!(tensor_parallel_size("g5g.8xlarge"), 1);
        assert_eq!(tensor_parallel_size("g5g.16xlarge"), 2);
        assert_eq!(tensor_parallel_size("g5g.metal"), 2);
    }

    #[test]
    fn only_xlarge_needs_swap() {
        // 8 GiB cannot mmap the 10.2 GB checkpoint; 16 GiB and up can.
        assert!(needs_swap("g5g.xlarge"));
        for t in ["g5g.2xlarge", "g5g.4xlarge", "g5g.8xlarge", "g5g.16xlarge"] {
            assert!(!needs_swap(t), "{t} should not need swap");
        }
    }

    #[test]
    fn unknown_types_are_rejected_and_never_need_swap() {
        assert!(validate_instance_type("t4g.2xlarge").is_err());
        assert!(validate_instance_type("g5.2xlarge").is_err());
        assert!(validate_instance_type("g5g.2xlarge").is_ok());
        // An unknown type reports 0 GiB; needs_swap must not read that as "tiny".
        assert!(!needs_swap("t4g.2xlarge"));
    }

    #[test]
    fn serve_flags_are_turing_shaped() {
        let f = serve_flags("google/gemma-4-E2B-it", "g5g.2xlarge");
        assert!(f.contains("--dtype float16"), "Turing has no bf16 datapath");
        assert!(
            f.contains("--kv-cache-dtype auto"),
            "Turing has no fp8 datapath"
        );
        assert!(f.contains("--tensor-parallel-size 1"));
        // vLLM v0.27 does not recognise this variable and forces TRITON_ATTN.
        assert!(!f.contains("attention-backend"));
    }
}
