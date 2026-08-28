//! OpenAI-compatible HTTP surface.
//!
//! The endpoint set matches what `jax_openai_server.py` served, because the
//! clients pointed at this rig — `query_gemma4`, `verify_model_health`, the
//! benchmark harness — do not know or care that the engine underneath is now
//! Rust. Swapping the engine is only a real A/B if the wire protocol is
//! unchanged.
//!
//! One deliberate difference from the vLLM rigs: `/v1/completions` here goes
//! through the same chat template as `/v1/chat/completions`. On the vLLM path a
//! raw completion against an `-it` checkpoint returns an empty string, which has
//! cost this repo enough debugging time to be written into the root CLAUDE.md.
//! There is no reason to reproduce that.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::engine::{Engine, GenerateRequest};

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state)
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn request_id(prefix: &str) -> String {
    format!("{prefix}-{:x}", now().wrapping_mul(2_654_435_761))
}

// ── /health ─────────────────────────────────────────────────────────────────
//
// Answers only once the model thread is alive and has reported. That is the
// point: this rig's whole provisioning story is that RUNNING is not ready, and
// a health endpoint that returns 200 before the weights are resident would
// reintroduce exactly that gap one layer up.

async fn health(State(state): State<AppState>) -> Response {
    match state.engine.live_info().await {
        Ok(info) if info.can_generate => (
            StatusCode::OK,
            Json(json!({"status": "ok", "engine": info})),
        )
            .into_response(),
        Ok(info) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "no-model", "engine": info})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "model-thread-down", "error": e.to_string()})),
        )
            .into_response(),
    }
}

async fn models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let info = state.engine.info();
    Json(json!({
        "object": "list",
        "data": [{
            "id": info.model,
            "object": "model",
            "created": now(),
            "owned_by": "gce-jaxrust-v6e1-2b",
        }],
    }))
}

// ── chat completions ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    // Accepted and ignored — see EngineConfig::temperature for why.
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

/// Folds an OpenAI message list into the (system, user) pair the Gemma chat
/// template helper takes.
///
/// `encode_chat_prompt_auto` accepts one system string and one user string, so a
/// multi-turn conversation has to be flattened. Earlier turns are rendered with
/// explicit role labels and prepended to the final user turn rather than being
/// dropped — lossier than the real template's turn markers, and the reason the
/// first place to look for a multi-turn quality regression is here, not the model.
fn flatten(messages: &[ChatMessage]) -> (Option<String>, String) {
    let system = messages
        .iter()
        .filter(|m| m.role == "system")
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    let turns: Vec<&ChatMessage> = messages.iter().filter(|m| m.role != "system").collect();
    let user = match turns.split_last() {
        None => String::new(),
        Some((last, [])) => last.content.clone(),
        Some((last, prior)) => {
            let mut buf = String::new();
            for m in prior {
                buf.push_str(&format!("{}: {}\n", m.role, m.content));
            }
            buf.push_str(&last.content);
            buf
        }
    };

    (
        if system.is_empty() {
            None
        } else {
            Some(system)
        },
        user,
    )
}

async fn chat_completions(State(state): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    let (system, user) = flatten(&req.messages);
    let max_new = req.max_tokens.unwrap_or(256);
    if req.temperature.is_some() || req.top_p.is_some() {
        // Not an error and not silently honoured either: the runner fixes its
        // SampleOpts at build time, so the only truthful thing to do with a
        // per-request override is say it was dropped.
        tracing::debug!(
            temperature = ?req.temperature,
            top_p = ?req.top_p,
            "per-request sampling overrides ignored; sampling is set by the server's CLI flags"
        );
    }
    let model = req
        .model
        .unwrap_or_else(|| state.engine.info().model.clone());

    if req.stream {
        return stream_response(state, system, user, max_new, model, "chat.completion.chunk").await;
    }

    let result = state
        .engine
        .generate(GenerateRequest {
            system,
            user,
            max_new_tokens: max_new,
            tokens: None,
        })
        .await;

    match result {
        Ok(out) => Json(json!({
            "id": request_id("chatcmpl"),
            "object": "chat.completion",
            "created": now(),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": out.text},
                "finish_reason": if out.completion_tokens >= max_new { "length" } else { "stop" },
            }],
            "usage": Usage {
                prompt_tokens: out.prompt_tokens,
                completion_tokens: out.completion_tokens,
                total_tokens: out.prompt_tokens + out.completion_tokens,
            },
        }))
        .into_response(),
        Err(e) => error_response(e),
    }
}

// ── completions ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let max_new = req.max_tokens.unwrap_or(256);
    let model = req
        .model
        .unwrap_or_else(|| state.engine.info().model.clone());

    if req.stream {
        return stream_response(state, None, req.prompt, max_new, model, "text_completion").await;
    }

    match state
        .engine
        .generate(GenerateRequest {
            system: None,
            user: req.prompt,
            max_new_tokens: max_new,
            tokens: None,
        })
        .await
    {
        Ok(out) => Json(json!({
            "id": request_id("cmpl"),
            "object": "text_completion",
            "created": now(),
            "model": model,
            "choices": [{
                "index": 0,
                "text": out.text,
                "finish_reason": if out.completion_tokens >= max_new { "length" } else { "stop" },
            }],
            "usage": Usage {
                prompt_tokens: out.prompt_tokens,
                completion_tokens: out.completion_tokens,
                total_tokens: out.prompt_tokens + out.completion_tokens,
            },
        }))
        .into_response(),
        Err(e) => error_response(e),
    }
}

// ── streaming ───────────────────────────────────────────────────────────────

/// SSE in the OpenAI shape. The model thread pushes decoded fragments into an
/// unbounded channel; this turns each into one `data:` frame and closes with
/// `data: [DONE]`.
///
/// The generation itself is spawned rather than awaited so the first frame can
/// leave before the last token is produced — otherwise "streaming" would only
/// mean re-chunking a finished string, which is the failure mode worth naming
/// because it looks identical from the client until you time it.
async fn stream_response(
    state: AppState,
    system: Option<String>,
    user: String,
    max_new: usize,
    model: String,
    object: &'static str,
) -> Response {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let engine = state.engine.clone();
    let id = request_id(if object == "text_completion" {
        "cmpl"
    } else {
        "chatcmpl"
    });

    let handle = tokio::spawn(async move {
        engine
            .generate(GenerateRequest {
                system,
                user,
                max_new_tokens: max_new,
                tokens: Some(tx),
            })
            .await
    });

    let created = now();
    let id_for_stream = id.clone();
    let stream = async_stream::stream! {
        while let Some(fragment) = rx.recv().await {
            let chunk = if object == "text_completion" {
                json!({
                    "id": id_for_stream, "object": object, "created": created, "model": model,
                    "choices": [{"index": 0, "text": fragment, "finish_reason": null}],
                })
            } else {
                json!({
                    "id": id_for_stream, "object": object, "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"content": fragment}, "finish_reason": null}],
                })
            };
            yield Ok::<Event, std::convert::Infallible>(Event::default().data(chunk.to_string()));
        }
        // The join tells us whether the run ended or failed; a failure after the
        // first frame cannot change the status code, so it goes into the stream.
        if let Ok(Err(e)) = handle.await {
            yield Ok(Event::default().data(json!({"error": e.to_string()}).to_string()));
        }
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn error_response(e: anyhow::Error) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error": {"message": e.to_string(), "type": "engine_error"}})),
    )
        .into_response()
}
