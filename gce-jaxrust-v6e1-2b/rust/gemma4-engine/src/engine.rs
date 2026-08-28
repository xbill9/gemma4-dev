//! The model actor.
//!
//! PJRT's `Client` is reference-counted with `Rc`, so everything rlx hands back
//! from a TPU session — the client, the compiled graph, the runner — is `!Send`.
//! That is not a wart to work around with a mutex: a `Mutex<GemmaRunner>` will not
//! even compile across an await point, and forcing it would be wrong anyway, because
//! one v6e-1 chip runs one graph at a time.
//!
//! So the runner lives on exactly one thread for the life of the process and the HTTP
//! layer talks to it over a channel. Requests queue instead of racing, which is the
//! honest shape of a single-chip rig: concurrency here buys nothing and costs the
//! recompiles that a second in-flight sequence length would trigger.

use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use anyhow::{anyhow, Result};
use tokio::sync::{mpsc as tokio_mpsc, oneshot};

/// What the engine reports about itself once it is up. Filled in on the model
/// thread after the weights load, so every field is observed rather than assumed.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineInfo {
    pub model: String,
    pub device: String,
    pub max_seq: usize,
    /// False on a build without the `gemma` feature: the server runs, reports the
    /// device and refuses generation. See the licensing note in Cargo.toml.
    pub can_generate: bool,
    pub vocab_size: Option<usize>,
}

pub struct GenerateRequest {
    pub system: Option<String>,
    pub user: String,
    pub max_new_tokens: usize,
    /// When set, each decoded fragment is pushed here as it is produced and the
    /// final `reply` carries the full text as well.
    pub tokens: Option<tokio_mpsc::UnboundedSender<String>>,
}

enum Job {
    Info(oneshot::Sender<EngineInfo>),
    Generate(Box<GenerateRequest>, oneshot::Sender<Result<Generated>>),
}

#[derive(Debug, Clone)]
pub struct Generated {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Clone)]
pub struct Engine {
    jobs: mpsc::Sender<Job>,
    info: EngineInfo,
}

pub struct EngineConfig {
    pub weights: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub config_json: Option<PathBuf>,
    pub max_seq: usize,
    pub device: Device,
    pub packed_weights: Option<bool>,
    pub model_label: String,
    /// Sampling is a PROCESS-wide setting here, not a per-request one.
    /// `GemmaRunner` takes its `SampleOpts` at build time and keeps the field
    /// private, so a per-request temperature would mean rebuilding the runner —
    /// which on this rig means recompiling the graph. The HTTP layer therefore
    /// accepts OpenAI's `temperature` / `top_p` and ignores them, and says so
    /// rather than pretending they took effect.
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub greedy: bool,
}

/// Which backend to compile for. Kept as this crate's own enum rather than
/// re-exporting rlx's: the `gemma` feature can be off, and the CLI still has to
/// parse `--device`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Device {
    Tpu,
    Cpu,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Tpu => write!(f, "tpu"),
            Device::Cpu => write!(f, "cpu"),
        }
    }
}

impl Engine {
    /// Spawns the model thread and blocks until the weights are loaded and the
    /// graph is compiled — or until that fails, which is reported here rather
    /// than on the first request. A rig that answers /health before it can
    /// generate is the same "RUNNING is not ready" trap the provisioning side
    /// of this rig spends most of its documentation on.
    pub fn start(cfg: EngineConfig) -> Result<Self> {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<EngineInfo>>();

        thread::Builder::new()
            .name("gemma4-model".to_string())
            .spawn(move || model_thread(cfg, job_rx, ready_tx))?;

        let info = ready_rx
            .recv()
            .map_err(|_| anyhow!("model thread died during startup without reporting why"))??;

        Ok(Engine { jobs: job_tx, info })
    }

    pub fn info(&self) -> &EngineInfo {
        &self.info
    }

    /// Re-reads the info from the model thread. Cheap, and it doubles as a
    /// liveness check: if the thread has died this returns an error instead of
    /// the cached struct, which is what /health wants to know.
    pub async fn live_info(&self) -> Result<EngineInfo> {
        let (tx, rx) = oneshot::channel();
        self.jobs
            .send(Job::Info(tx))
            .map_err(|_| anyhow!("model thread is gone"))?;
        rx.await
            .map_err(|_| anyhow!("model thread dropped the reply"))
    }

    pub async fn generate(&self, req: GenerateRequest) -> Result<Generated> {
        let (tx, rx) = oneshot::channel();
        self.jobs
            .send(Job::Generate(Box::new(req), tx))
            .map_err(|_| anyhow!("model thread is gone"))?;
        rx.await
            .map_err(|_| anyhow!("model thread dropped the reply"))?
    }
}

#[cfg(feature = "gemma")]
fn model_thread(
    cfg: EngineConfig,
    jobs: mpsc::Receiver<Job>,
    ready: mpsc::Sender<Result<EngineInfo>>,
) {
    use rlx_gemma::prelude::{Device as RlxDevice, GemmaRunner, SampleOpts};
    use rlx_gemma::{decode_token_auto, encode_chat_prompt_auto, GemmaConfigSource};

    let device = match cfg.device {
        Device::Tpu => RlxDevice::Tpu,
        Device::Cpu => RlxDevice::Cpu,
    };

    let sample = if cfg.greedy {
        SampleOpts::greedy()
    } else {
        SampleOpts::temperature(cfg.temperature, cfg.seed)
            .with_top_k(cfg.top_k)
            .with_top_p(cfg.top_p)
    };

    let build = || -> Result<GemmaRunner> {
        let mut builder = GemmaRunner::builder()
            .weights(cfg.weights.clone())
            .device(device)
            .max_seq(cfg.max_seq)
            .sample(sample)
            // Streaming on: the generator then calls back per token instead of
            // handing over the whole sequence at the end, which is what the SSE
            // path needs and costs nothing when nobody is listening.
            .stream(true);
        if let Some(path) = cfg.config_json.clone() {
            builder = builder.config(GemmaConfigSource::JsonFile(path));
        }
        if let Some(packed) = cfg.packed_weights {
            builder = builder.packed_weights(packed);
        }
        builder.build()
    };

    let mut runner = match build() {
        Ok(r) => r,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };

    let info = EngineInfo {
        model: cfg.model_label.clone(),
        device: cfg.device.to_string(),
        max_seq: cfg.max_seq,
        can_generate: true,
        vocab_size: Some(runner.config().vocab_size),
    };
    if ready.send(Ok(info.clone())).is_err() {
        return;
    }

    while let Ok(job) = jobs.recv() {
        match job {
            Job::Info(reply) => {
                let _ = reply.send(info.clone());
            }
            Job::Generate(req, reply) => {
                let out = (|| -> Result<Generated> {
                    let prompt_ids = encode_chat_prompt_auto(
                        &cfg.weights,
                        cfg.tokenizer.as_deref(),
                        req.system.as_deref(),
                        &req.user,
                        true,
                    )?;

                    let mut text = String::new();
                    let mut emitted = 0usize;
                    let tokenizer = cfg.tokenizer.clone();
                    let weights = cfg.weights.clone();
                    let sink = req.tokens.clone();

                    let tokens = runner.generate(&prompt_ids, req.max_new_tokens, |tok| {
                        emitted += 1;
                        // A token that fails to decode is dropped from the text
                        // rather than aborting the stream: partial UTF-8 across a
                        // byte-pair boundary is normal, not an error.
                        if let Ok(fragment) = decode_token_auto(&weights, tokenizer.as_deref(), tok)
                        {
                            if let Some(sink) = sink.as_ref() {
                                let _ = sink.send(fragment.clone());
                            }
                            text.push_str(&fragment);
                        }
                    })?;

                    Ok(Generated {
                        text,
                        prompt_tokens: prompt_ids.len(),
                        completion_tokens: tokens.len().max(emitted),
                    })
                })();
                let _ = reply.send(out);
            }
        }
    }
}

#[cfg(not(feature = "gemma"))]
fn model_thread(
    cfg: EngineConfig,
    jobs: mpsc::Receiver<Job>,
    ready: mpsc::Sender<Result<EngineInfo>>,
) {
    let info = EngineInfo {
        model: cfg.model_label.clone(),
        device: cfg.device.to_string(),
        max_seq: cfg.max_seq,
        can_generate: false,
        vocab_size: None,
    };
    if ready.send(Ok(info.clone())).is_err() {
        return;
    }
    while let Ok(job) = jobs.recv() {
        match job {
            Job::Info(reply) => {
                let _ = reply.send(info.clone());
            }
            Job::Generate(_, reply) => {
                let _ = reply.send(Err(anyhow!(
                    "this binary was built without the `gemma` feature, so it carries no model. \
                     Rebuild with `cargo build --release --features tpu,gemma` — and read the \
                     licensing note in Cargo.toml first: rlx-gemma is GPL-3.0-only."
                )));
            }
        }
    }
}
