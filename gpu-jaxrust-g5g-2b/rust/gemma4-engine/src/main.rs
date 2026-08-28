//! `gemma4-engine` — Gemma 4 E2B served from Rust on an NVIDIA T4G, no Python in
//! the serving path.
//!
//! The graph is built in rlx's JAX-shaped IR and executed through rlx's CUDA backend
//! (cuBLAS/cuDNN/NVRTC). What that replaces is `jax_openai_server.py` +
//! `jax_engine.py` + `ports/gemma4/`, which stay in the tree as the **parity oracle**:
//! they are the only implementation here whose numbers have been checked, so a
//! disagreement between the two is evidence about this engine, not about the model.
//!
//! Two things about this hardware that the TPU sibling does not have to think about:
//!
//! - **Turing has no bf16 datapath.** float16 is the only real 16-bit choice, and a
//!   bf16 mismatch does not error — it emulates through fp32. On the Python rig that
//!   cost 86.8% of decode and hid for weeks.
//! - **NVRTC compiles at runtime**, so SM 7.5 is a JIT target rather than a
//!   cubin-arch-table question. That is the failure that breaks the vLLM sibling and
//!   it cannot happen this way.

mod api;
mod engine;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use engine::{Device, Engine, EngineConfig, Metrics, T4G_MEMORY_BUDGET_GB};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "gemma4-engine",
    about = "OpenAI-compatible Gemma 4 E2B server on a T4G, driven from Rust"
)]
struct Args {
    /// Directory or file holding the checkpoint (safetensors or GGUF).
    #[arg(long, env = "JAXRUST_WEIGHTS")]
    weights: PathBuf,

    #[arg(long, env = "JAXRUST_TOKENIZER")]
    tokenizer: Option<PathBuf>,

    /// config.json. Required for safetensors; embedded for GGUF. Also what /health
    /// derives the KV footprint from.
    #[arg(long, env = "JAXRUST_CONFIG")]
    config: Option<PathBuf>,

    #[arg(long, env = "RIG_NAME", default_value = "gpu-jaxrust-g5g-2b")]
    rig_name: String,

    /// Cosmetic; the weights decide what actually runs.
    #[arg(long, env = "MODEL_NAME", default_value = "google/gemma-4-E2B-it")]
    model_name: String,

    /// Context length to compile for. Graphs are statically shaped, so this is a
    /// compile-time commitment, not a runtime cap.
    ///
    /// **4096, not 8192.** MEASURED 2026-08-26 on the Python engine: 4,105 tokens
    /// serve and 5,120 is infeasible, so a little over half of 8192 was reachable.
    /// The prefill transient has a flat term and a linear one (~0.9 MiB/token above
    /// ~4K); KV is not what binds — it is 30 MiB at this length.
    #[arg(long, env = "MAX_MODEL_LEN", default_value_t = 4096)]
    max_model_len: usize,

    #[arg(long, value_enum, env = "JAXRUST_DEVICE", default_value_t = Device::Cuda)]
    device: Device,

    /// Device memory budget. Defaults to the T4G's measured 14.07 GB. Quote the
    /// largest CONTIGUOUS block, not free bytes: peak fragmentation measured 0.661
    /// and two of three quantization bugs failed to allocate with GBs free.
    #[arg(long, env = "JAXRUST_MAX_MEMORY_GB", default_value_t = T4G_MEMORY_BUDGET_GB)]
    max_memory_gb: f32,

    #[arg(long, env = "JAXRUST_PACKED_WEIGHTS")]
    packed_weights: Option<bool>,

    #[arg(long, default_value_t = false)]
    greedy: bool,
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 64)]
    top_k: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[arg(long, env = "JAXRUST_HOST", default_value = "0.0.0.0")]
    host: String,
    #[arg(long, env = "JAXRUST_PORT", default_value_t = 8000)]
    port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Configure the root subscriber BEFORE anything else logs. On the Python rig the
    // device-policy banner was dropped for the life of the rig because uvicorn only
    // configures its own loggers and never adds a root handler — on the one rig whose
    // entire premise is which dtype the device picked.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    if !args.weights.exists() {
        anyhow::bail!("weights path {:?} does not exist", args.weights);
    }

    tracing::info!(
        rig = %args.rig_name, weights = ?args.weights, device = %args.device,
        max_model_len = args.max_model_len, max_memory_gb = args.max_memory_gb,
        "loading — the process does not bind a port until the graph is compiled"
    );

    let engine = Engine::start(EngineConfig {
        rig_name: args.rig_name.clone(),
        weights: args.weights.clone(),
        tokenizer: args.tokenizer,
        config_json: args.config,
        max_seq: args.max_model_len,
        device: args.device,
        max_memory_gb: args.max_memory_gb,
        packed_weights: args.packed_weights,
        model_label: args.model_name,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        seed: args.seed,
        greedy: args.greedy,
    })
    .context("engine failed to start")?;

    let info = engine.info().clone();
    if !info.can_generate {
        tracing::warn!(
            "built without the `gemma` feature: /health reports no-model and generation is \
             refused. Rebuild with --no-default-features --features cuda,gemma."
        );
    }

    let state = api::AppState {
        engine: Arc::new(engine),
        metrics: Arc::new(Metrics::default()),
    };
    let app = api::router(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    // The rig's readiness check scans for this exact marker. Keep it if you reword.
    println!("JAXRUST-SERVER: listening on {addr}");
    tracing::info!(%addr, "serving");

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
