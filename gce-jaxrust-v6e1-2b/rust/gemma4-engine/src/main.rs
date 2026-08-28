//! `gemma4-engine` — Gemma 4 served from Rust on a TPU v6e-1, no Python in the
//! serving path.
//!
//! The graph is built in rlx's JAX-shaped IR, lowered to HLO and executed through
//! libtpu's PJRT plugin. What that replaces is `jax_openai_server.py` +
//! `jax_engine.py` + `ports/gemma4/`, which stay in the tree as the parity oracle:
//! they are the only implementation here whose numbers have been checked against
//! the reference, so a disagreement between the two is evidence about this engine,
//! not about the model.
//!
//! Run `xla-probe` first. It answers a smaller question — can this host compile and
//! execute StableHLO on the chip at all — and if the answer is no, nothing here can
//! tell you why.

mod api;
mod engine;

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use engine::{Device, Engine, EngineConfig};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "gemma4-engine",
    about = "OpenAI-compatible Gemma 4 server on TPU, driven from Rust"
)]
struct Args {
    /// Directory or file holding the checkpoint (safetensors or GGUF).
    #[arg(long, env = "JAXRUST_WEIGHTS")]
    weights: PathBuf,

    /// tokenizer.json. Defaults to one found beside the weights.
    #[arg(long, env = "JAXRUST_TOKENIZER")]
    tokenizer: Option<PathBuf>,

    /// config.json. Defaults to one found beside the weights; required for
    /// safetensors, embedded in the file for GGUF.
    #[arg(long, env = "JAXRUST_CONFIG")]
    config: Option<PathBuf>,

    /// Name reported by /v1/models. Cosmetic; the weights decide what actually runs.
    #[arg(long, env = "MODEL_NAME", default_value = "google/gemma-4-E2B-it")]
    model_name: String,

    /// Context length to compile for. Graphs here are statically shaped, so this
    /// is a compile-time commitment, not a runtime cap: raising it costs HBM for
    /// the KV cache whether or not a request ever uses the length.
    #[arg(long, env = "MAX_MODEL_LEN", default_value_t = 8192)]
    max_model_len: usize,

    #[arg(long, value_enum, env = "JAXRUST_DEVICE", default_value_t = Device::Tpu)]
    device: Device,

    /// Keep K-quant GGUF weights packed in the arena instead of dequantising to
    /// F32 at load. Unset = rlx decides by file size (packed at ≥ 256 MiB).
    #[arg(long, env = "JAXRUST_PACKED_WEIGHTS")]
    packed_weights: Option<bool>,

    /// Sampling, fixed for the life of the process — see EngineConfig::temperature.
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
        weights = ?args.weights,
        device = %args.device,
        max_model_len = args.max_model_len,
        "loading — the process does not bind a port until the graph is compiled"
    );

    let engine = Engine::start(EngineConfig {
        weights: args.weights.clone(),
        tokenizer: args.tokenizer,
        config_json: args.config,
        max_seq: args.max_model_len,
        device: args.device,
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
    tracing::info!(?info, "engine ready");
    if !info.can_generate {
        tracing::warn!(
            "built without the `gemma` feature: /health will report no-model and generation \
             will be refused. Rebuild with --features tpu,gemma."
        );
    }

    let state = api::AppState {
        engine: std::sync::Arc::new(engine),
    };
    let app = api::router(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    // This exact line is what `wait_for_jaxrust_ready` scans the serial log for.
    // Keep the marker if you change the wording.
    println!("JAXRUST-SERVER: listening on {addr}");
    tracing::info!(%addr, "serving");

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
