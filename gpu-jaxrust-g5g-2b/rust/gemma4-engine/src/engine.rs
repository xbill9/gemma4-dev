//! The model actor.
//!
//! rlx hands back a session whose backend handles are `Rc`-based, so the runner is
//! `!Send`. That is not a wart to paper over with a mutex — a `Mutex<GemmaRunner>`
//! will not compile across an await point, and forcing it would be wrong anyway:
//! **one T4G runs one graph at a time**, and this rig serves `MAX_NUM_SEQS=1`
//! because the Python engine raises `NotImplementedError` for `B > 1` and the
//! decode step donates its KV buffers.
//!
//! So the runner lives on exactly one thread for the life of the process and the
//! HTTP layer talks to it over a channel. Requests queue rather than race, which is
//! the honest shape here: concurrency buys nothing on one chip and costs recompiles,
//! because `max_new_tokens` is part of the compiled shape.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use anyhow::{anyhow, Result};
use tokio::sync::{mpsc as tokio_mpsc, oneshot};

/// Measured usable device memory on a T4G: 15360 MiB total, 14.07 GB budget after
/// the CUDA context. `HARDWARE.md` and this rig's own OOM table are the source.
/// Quoted as a budget, not as free bytes — peak fragmentation measured 0.661, so
/// the largest *contiguous* block is what actually binds.
pub const T4G_MEMORY_BUDGET_GB: f32 = 14.07;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Device {
    /// NVIDIA T4G via native CUDA (cuBLAS/cuDNN/NVRTC). The rig's real target.
    Cuda,
    /// CPU. Not a serving mode — it is how parity against the Python port is done
    /// off the chip, and it is why this crate builds on a box with no driver.
    Cpu,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cuda => write!(f, "cuda"),
            Device::Cpu => write!(f, "cpu"),
        }
    }
}

/// What the engine reports once it is up. Filled in on the model thread *after* the
/// weights load, so every field is observed rather than assumed — the same reason
/// the Python rig prints a resolved-configuration banner instead of echoing config.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineInfo {
    pub rig: String,
    pub model: String,
    pub device: String,
    pub max_seq: usize,
    pub compute_dtype: String,
    /// False on a build without the `gemma` feature: the server runs, reports the
    /// device, and refuses generation. See the licensing note in Cargo.toml.
    pub can_generate: bool,
    pub vocab_size: Option<usize>,
    /// KV bytes at `max_seq` with the sliding ring on, computed from the geometry
    /// crate. Present so an operator can see that KV is *not* the binding
    /// constraint here — prefill transients are ~174x larger at 4K.
    pub kv_bytes_at_max_seq: Option<usize>,
}

pub struct GenerateRequest {
    pub system: Option<String>,
    pub user: String,
    pub max_new_tokens: usize,
    /// When set, each decoded fragment is pushed here as produced; the final reply
    /// still carries the full text.
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
    pub decode_seconds: f64,
    /// True when this (bucket, max_new_tokens) shape had not been compiled before.
    /// The Python rig measured a 25x cold/warm gap on an identical request, so a
    /// cumulative rate that includes a cold request is not a result.
    pub cold_shape: bool,
}

/// Process-wide counters, named to match the Python rig's series **exactly**.
/// `CLAUDE.md` is explicit that both benchmark reports compare on
/// `tpu_jax_decode_tokens_per_second` *by name*, so renaming the prefix would break
/// continuity with every existing report. The `rig` label is what distinguishes the
/// two engines, not the series name.
#[derive(Default)]
pub struct Metrics {
    pub requests: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub completion_tokens: AtomicU64,
    pub decode_micros: AtomicU64,
    pub cold_requests: AtomicU64,
    pub errors: AtomicU64,
}

impl Metrics {
    pub fn render(&self, info: &EngineInfo) -> String {
        let rig = &info.rig;
        let comp = self.completion_tokens.load(Ordering::Relaxed);
        let micros = self.decode_micros.load(Ordering::Relaxed);
        let cold = self.cold_requests.load(Ordering::Relaxed);
        let secs = micros as f64 / 1e6;
        let rate = if secs > 0.0 { comp as f64 / secs } else { 0.0 };
        let mut s = String::new();
        s.push_str("# HELP tpu_jax_decode_tokens_per_second Cumulative decode rate.\n");
        s.push_str("# TYPE tpu_jax_decode_tokens_per_second gauge\n");
        s.push_str(&format!(
            "tpu_jax_decode_tokens_per_second{{rig=\"{rig}\",device=\"{}\"}} {rate:.4}\n",
            info.device
        ));
        s.push_str("# TYPE tpu_jax_decode_seconds_total counter\n");
        s.push_str(&format!(
            "tpu_jax_decode_seconds_total{{rig=\"{rig}\"}} {secs:.6}\n"
        ));
        s.push_str("# TYPE tpu_jax_requests_total counter\n");
        s.push_str(&format!(
            "tpu_jax_requests_total{{rig=\"{rig}\"}} {}\n",
            self.requests.load(Ordering::Relaxed)
        ));
        s.push_str("# TYPE tpu_jax_prompt_tokens_total counter\n");
        s.push_str(&format!(
            "tpu_jax_prompt_tokens_total{{rig=\"{rig}\"}} {}\n",
            self.prompt_tokens.load(Ordering::Relaxed)
        ));
        s.push_str("# TYPE tpu_jax_completion_tokens_total counter\n");
        s.push_str(&format!(
            "tpu_jax_completion_tokens_total{{rig=\"{rig}\"}} {comp}\n"
        ));
        s.push_str("# HELP tpu_jax_cold_requests_total Requests that compiled a new shape.\n");
        s.push_str("# TYPE tpu_jax_cold_requests_total counter\n");
        s.push_str(&format!(
            "tpu_jax_cold_requests_total{{rig=\"{rig}\"}} {cold}\n"
        ));
        s.push_str("# TYPE tpu_jax_errors_total counter\n");
        s.push_str(&format!(
            "tpu_jax_errors_total{{rig=\"{rig}\"}} {}\n",
            self.errors.load(Ordering::Relaxed)
        ));
        // Reported so a reader can see the dtype the device actually picked, which
        // is the single most consequential fact about this hardware.
        s.push_str("# TYPE tpu_jax_precision_info gauge\n");
        s.push_str(&format!(
            "tpu_jax_precision_info{{rig=\"{rig}\",compute_dtype=\"{}\"}} 1\n",
            info.compute_dtype
        ));
        s
    }
}

pub struct EngineConfig {
    pub rig_name: String,
    pub weights: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub config_json: Option<PathBuf>,
    pub max_seq: usize,
    pub device: Device,
    pub max_memory_gb: f32,
    pub packed_weights: Option<bool>,
    pub model_label: String,
    /// Sampling is a PROCESS-wide setting, not per-request: `GemmaRunner` takes its
    /// `SampleOpts` at build time, so a per-request temperature would mean rebuilding
    /// the runner — which means recompiling the graph. The HTTP layer accepts
    /// OpenAI's `temperature`/`top_p` and **says** it ignored them rather than
    /// pretending they took effect.
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub greedy: bool,
}

#[derive(Clone)]
pub struct Engine {
    jobs: mpsc::Sender<Job>,
    info: EngineInfo,
}

impl Engine {
    /// Spawns the model thread and blocks until the weights are loaded and the graph
    /// compiled — or until that fails, reported here rather than on the first
    /// request. A rig that answers /health before it can generate is the same
    /// "RUNNING is not READY" trap the provisioning half of this rig documents.
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

    /// Re-reads info from the model thread. Doubles as a liveness check: if the
    /// thread has died this errors instead of returning the cached struct, which is
    /// what /health actually wants to know.
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

/// KV footprint at `max_seq`, straight from the geometry crate, so /health can state
/// it rather than an operator deriving it. E2B at ctx 4096 with the ring on is
/// 30 MiB — 0.22% of the budget.
fn kv_bytes(config_json: Option<&PathBuf>, max_seq: usize) -> Option<usize> {
    let path = config_json?;
    let text = std::fs::read_to_string(path).ok()?;
    let cfg = gemma4_geometry::config::Gemma4EConfig::from_json(&text).ok()?;
    Some(gemma4_geometry::kv::total_kv_bytes(&cfg, max_seq, true))
}

#[cfg(feature = "gemma")]
fn model_thread(
    cfg: EngineConfig,
    jobs: mpsc::Receiver<Job>,
    ready: mpsc::Sender<Result<EngineInfo>>,
) {
    use rlx_gemma::prelude::{Device as RlxDevice, GemmaRunner, SampleOpts};
    use rlx_gemma::{decode_token_auto, encode_chat_prompt_auto, GemmaConfigSource};
    use std::collections::HashSet;

    let device = match cfg.device {
        Device::Cuda => RlxDevice::Cuda,
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
        let mut b = GemmaRunner::builder()
            .weights(cfg.weights.clone())
            .device(device)
            .max_seq(cfg.max_seq)
            .max_memory_gb(cfg.max_memory_gb)
            .sample(sample)
            // Streaming on: the generator calls back per token rather than handing
            // over the whole sequence, which is what SSE needs and costs nothing
            // when nobody is listening.
            .stream(true);
        if let Some(p) = cfg.config_json.clone() {
            b = b.config(GemmaConfigSource::JsonFile(p));
        }
        if let Some(packed) = cfg.packed_weights {
            b = b.packed_weights(packed);
        }
        b.build()
    };

    let mut runner = match build() {
        Ok(r) => r,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };

    let info = EngineInfo {
        rig: cfg.rig_name.clone(),
        model: cfg.model_label.clone(),
        device: cfg.device.to_string(),
        // Turing has no bf16 datapath, so float16 is the only real 16-bit choice.
        // Stated rather than inferred: on the Python rig a bf16/f16 mismatch cost
        // 86.8% of decode and never raised, because bf16 emulates through fp32.
        compute_dtype: "float16".to_string(),
        max_seq: cfg.max_seq,
        can_generate: true,
        vocab_size: Some(runner.config().vocab_size),
        kv_bytes_at_max_seq: kv_bytes(cfg.config_json.as_ref(), cfg.max_seq),
    };
    if ready.send(Ok(info.clone())).is_err() {
        return;
    }
    tracing::info!(
        rig = %info.rig, device = %info.device, compute_dtype = %info.compute_dtype,
        max_seq = info.max_seq, vocab = ?info.vocab_size, kv_bytes = ?info.kv_bytes_at_max_seq,
        "READY"
    );

    // Shapes this process has compiled. `max_new_tokens` is part of the compiled
    // shape, so (bucket, max_new) is the key — warming at a different max_tokens
    // than you measure leaves the measured request cold, a 4x error on the Python rig.
    let mut seen: HashSet<(usize, usize)> = HashSet::new();

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
                    let bucket = gemma4_geometry::kv::bucket_for(prompt_ids.len());
                    let cold = seen.insert((bucket, req.max_new_tokens));

                    let mut text = String::new();
                    let mut emitted = 0usize;
                    let weights = cfg.weights.clone();
                    let tokenizer = cfg.tokenizer.clone();
                    let sink = req.tokens.clone();

                    let started = Instant::now();
                    let toks = runner.generate(&prompt_ids, req.max_new_tokens, |tok| {
                        emitted += 1;
                        // A token that fails to decode is dropped rather than
                        // aborting the stream: partial UTF-8 across a byte-pair
                        // boundary is normal, not an error.
                        if let Ok(frag) = decode_token_auto(&weights, tokenizer.as_deref(), tok) {
                            if let Some(s) = sink.as_ref() {
                                let _ = s.send(frag.clone());
                            }
                            text.push_str(&frag);
                        }
                    })?;
                    let decode_seconds = started.elapsed().as_secs_f64();

                    Ok(Generated {
                        text,
                        prompt_tokens: prompt_ids.len(),
                        completion_tokens: toks.len().max(emitted),
                        decode_seconds,
                        cold_shape: cold,
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
        rig: cfg.rig_name.clone(),
        model: cfg.model_label.clone(),
        device: cfg.device.to_string(),
        compute_dtype: "float16".to_string(),
        max_seq: cfg.max_seq,
        can_generate: false,
        vocab_size: None,
        kv_bytes_at_max_seq: kv_bytes(cfg.config_json.as_ref(), cfg.max_seq),
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
                     Rebuild with `--features cuda,gemma` — and read the licensing note in \
                     Cargo.toml first: rlx-gemma is GPL-3.0-only and did not follow rlx's \
                     0.2.14 relicense."
                )));
            }
        }
    }
}
