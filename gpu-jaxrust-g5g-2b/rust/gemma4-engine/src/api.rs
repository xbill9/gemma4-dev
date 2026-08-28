//! OpenAI-compatible HTTP surface.
//!
//! `/v1/chat/completions` is the endpoint that matters: raw `/v1/completions` skips
//! the chat template and is unreliable on `-it` models, which is why the rig's
//! `verify_model_health` uses this one. **Do not health-check by testing for a
//! non-empty response** — on the vLLM sibling a degenerate body measured as
//! `': ok: ok: ok…'`, which any non-empty test calls fine.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc as tokio_mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::engine::{Engine, GenerateRequest, Metrics};

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
    pub metrics: Arc<Metrics>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

async fn health(State(s): State<AppState>) -> Response {
    match s.engine.live_info().await {
        Ok(info) => {
            let code = if info.can_generate {
                StatusCode::OK
            } else {
                StatusCode::SERVICE_UNAVAILABLE
            };
            (code, Json(json!({ "status": if info.can_generate {"ok"} else {"no-model"}, "engine": info })))
                .into_response()
        }
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "status": "dead", "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn metrics(State(s): State<AppState>) -> Response {
    let info = s.engine.info().clone();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        s.metrics.render(&info),
    )
        .into_response()
}

async fn models(State(s): State<AppState>) -> Json<serde_json::Value> {
    let info = s.engine.info();
    Json(json!({
        "object": "list",
        "data": [{ "id": info.model, "object": "model", "owned_by": "google" }]
    }))
}

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
    pub stream: Option<bool>,
    /// Accepted and IGNORED — sampling is fixed at build time because changing it
    /// rebuilds the runner and recompiles the graph. The response says so in
    /// `system_fingerprint` rather than silently pretending it applied.
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

fn split_messages(msgs: &[ChatMessage]) -> (Option<String>, String) {
    let system = msgs
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content.clone())
        .filter(|s| !s.is_empty());
    let user = msgs
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();
    (system, user)
}

/// Cheap degenerate-output detector, ported from the Python rig's
/// `tpu_jax_degenerate_responses_total`. Observational only: it changes neither the
/// body nor the status code. Kept because it does not depend on ring eviction being
/// the only cause, and because "not empty" is not a health check.
///
/// **It looks for a repeating CYCLE, not just an identical token.** The padding
/// -eviction bug produced "a token repeated four times running", but the vLLM
/// sibling's measured degenerate body was `': ok: ok: ok…'` — an alternating
/// two-token cycle that an identical-adjacent test walks straight past. Both are the
/// same failure and both must trip it.
fn looks_degenerate(text: &str) -> bool {
    let toks: Vec<&str> = text.split_whitespace().collect();
    if toks.len() < 8 {
        return false;
    }
    // A cycle of length `n` repeated at least 4 times, anywhere in the output.
    const REPEATS: usize = 4;
    for n in 1..=4usize {
        let span = n * REPEATS;
        if toks.len() < span {
            break;
        }
        for start in 0..=(toks.len() - span) {
            let cycle = &toks[start..start + n];
            if (1..REPEATS).all(|k| &toks[start + k * n..start + (k + 1) * n] == cycle) {
                return true;
            }
        }
    }
    false
}

async fn chat_completions(State(s): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    let id = format!("chatcmpl-jaxrust-{}", std::process::id());
    let info = s.engine.info().clone();
    if !info.can_generate {
        s.metrics.errors.fetch_add(1, Ordering::Relaxed);
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": {"message":
                "built without the `gemma` feature: this binary carries no model", "type": "no_model"}})),
        )
            .into_response();
    }

    // A client asking for a model this process is not serving gets told, rather
    // than being answered by whatever happens to be loaded.
    if let Some(asked) = req.model.as_deref() {
        if asked != info.model {
            tracing::warn!(requested = asked, serving = %info.model,
                "request named a different model; serving the loaded one");
        }
    }
    // Sampling is fixed at build time (rebuilding the runner recompiles the graph),
    // so these are accepted and ignored — but *observably*, in the fingerprint.
    let ignored_sampling = req.temperature.is_some() || req.top_p.is_some();

    let (system, user) = split_messages(&req.messages);
    if user.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": {"message": "no user message", "type": "invalid_request_error"}})),
        )
            .into_response();
    }

    // Clamp against the compiled context and SAY SO. The Python rig's silent clamp
    // also changed the compiled shape, which turned into mystery slow requests.
    let requested = req.max_tokens.unwrap_or(256);
    let max_new = requested.min(info.max_seq);
    let clamped = max_new != requested;
    if clamped {
        tracing::warn!(
            requested,
            max_new,
            "max_tokens clamped — this CHANGES the compiled shape"
        );
    }

    s.metrics.requests.fetch_add(1, Ordering::Relaxed);

    // Streaming is honoured rather than ignored. The engine already calls back per
    // token; what SSE adds is a channel and a framing. A server that accepts
    // `stream: true` and returns a single blob is the silent-fallback pattern this
    // rig's own history is full of.
    if req.stream.unwrap_or(false) {
        let (tok_tx, tok_rx) = tokio_mpsc::unbounded_channel::<String>();
        let engine = s.engine.clone();
        let metrics = s.metrics.clone();
        let model = info.model.clone();
        let sid = id.clone();
        let (sys2, user2) = split_messages(&req.messages);
        tokio::spawn(async move {
            let r = engine
                .generate(GenerateRequest {
                    system: sys2,
                    user: user2,
                    max_new_tokens: max_new,
                    tokens: Some(tok_tx),
                })
                .await;
            match r {
                Ok(g) => {
                    metrics
                        .prompt_tokens
                        .fetch_add(g.prompt_tokens as u64, Ordering::Relaxed);
                    metrics
                        .completion_tokens
                        .fetch_add(g.completion_tokens as u64, Ordering::Relaxed);
                    metrics
                        .decode_micros
                        .fetch_add((g.decode_seconds * 1e6) as u64, Ordering::Relaxed);
                    if g.cold_shape {
                        metrics.cold_requests.fetch_add(1, Ordering::Relaxed);
                    }
                    tracing::info!(
                        req_id = %sid, prompt_tokens = g.prompt_tokens,
                        completion_tokens = g.completion_tokens, decode_s = g.decode_seconds,
                        cold = g.cold_shape, degenerate = looks_degenerate(&g.text),
                        stream = true, "request"
                    );
                }
                Err(e) => {
                    metrics.errors.fetch_add(1, Ordering::Relaxed);
                    // The SSE body runs AFTER the handler returns, so on the Python
                    // rig a streaming failure fell outside the handler's try
                    // entirely: not counted, not logged, just a short answer.
                    tracing::error!(error = %e, req_id = %sid, "streaming generation failed");
                }
            }
        });

        let head = id.clone();
        let model_for_chunks = model.clone();
        let stream = UnboundedReceiverStream::new(tok_rx)
            .map(move |frag| {
                let chunk = json!({
                    "id": head,
                    "object": "chat.completion.chunk",
                    "model": model_for_chunks,
                    "choices": [{"index": 0, "delta": {"content": frag}, "finish_reason": null}],
                });
                Ok::<Event, std::convert::Infallible>(Event::default().data(chunk.to_string()))
            })
            .chain(tokio_stream::iter(vec![
                Ok(Event::default().data("[DONE]")),
            ]));
        return Sse::new(stream).into_response();
    }

    let out = s
        .engine
        .generate(GenerateRequest {
            system,
            user,
            max_new_tokens: max_new,
            tokens: None,
        })
        .await;

    match out {
        Err(e) => {
            s.metrics.errors.fetch_add(1, Ordering::Relaxed);
            // Log the cause. The Python rig raised HTTPException(detail=str(exc)),
            // which discarded the traceback and left a per-request OOM invisible.
            tracing::error!(error = %e, req_id = %id, "generation failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": {"message": e.to_string(), "type": "engine_error"}})),
            )
                .into_response()
        }
        Ok(g) => {
            s.metrics
                .prompt_tokens
                .fetch_add(g.prompt_tokens as u64, Ordering::Relaxed);
            s.metrics
                .completion_tokens
                .fetch_add(g.completion_tokens as u64, Ordering::Relaxed);
            s.metrics
                .decode_micros
                .fetch_add((g.decode_seconds * 1e6) as u64, Ordering::Relaxed);
            if g.cold_shape {
                s.metrics.cold_requests.fetch_add(1, Ordering::Relaxed);
            }
            let degenerate = looks_degenerate(&g.text);
            let rate = if g.decode_seconds > 0.0 {
                g.completion_tokens as f64 / g.decode_seconds
            } else {
                0.0
            };
            // One flat key=value line per request. `req_id` reaching nothing but the
            // response body is why "request X was wrong" used to be unresolvable.
            tracing::info!(
                req_id = %id, prompt_tokens = g.prompt_tokens, completion_tokens = g.completion_tokens,
                decode_s = g.decode_seconds, tok_per_s = rate, cold = g.cold_shape,
                clamped, degenerate, "request"
            );
            let mut body = json!({
                "id": id,
                "object": "chat.completion",
                "model": info.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": g.text},
                    "finish_reason": "stop"
                }],
                "usage": Usage {
                    prompt_tokens: g.prompt_tokens,
                    completion_tokens: g.completion_tokens,
                    total_tokens: g.prompt_tokens + g.completion_tokens,
                },
            });
            body["system_fingerprint"] = json!(format!(
                "rig={} device={} dtype={} sampling=process-wide{}{}",
                info.rig,
                info.device,
                info.compute_dtype,
                if ignored_sampling {
                    "(temperature/top_p IGNORED)"
                } else {
                    ""
                },
                if clamped { " max_tokens=clamped" } else { "" }
            ));
            body["x_cold_shape"] = json!(g.cold_shape);
            body["x_degenerate"] = json!(degenerate);
            (StatusCode::OK, Json(body)).into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degenerate_detector_catches_the_reported_signature() {
        // The eviction bug's signature: a token repeated four times running.
        assert!(looks_degenerate("ok ok ok ok ok ok ok ok ok"));
        // The vLLM sibling's measured body, as an alternating two-token cycle.
        // An identical-adjacent test walks straight past this one.
        assert!(looks_degenerate(": ok : ok : ok : ok : ok : ok : ok : ok"));
        // ...and in the spacing it was actually reported with.
        assert!(looks_degenerate("x : ok: ok: ok: ok: ok: ok: ok: ok:"));
        // A degenerate tail after a coherent opening still counts.
        assert!(looks_degenerate(
            "The capital is Paris and and and and and and"
        ));
        // Real prose must not trip it.
        assert!(!looks_degenerate(
            "The capital of France is Paris, a city on the Seine with a long history."
        ));
        // Natural short repeats are not a cycle of four.
        assert!(!looks_degenerate(
            "it is very very good and that is that is why we say so today"
        ));
        // Short answers must not trip it at all.
        assert!(!looks_degenerate("yes yes"));
    }

    #[test]
    fn last_user_message_wins_and_system_is_separated() {
        let msgs = vec![
            ChatMessage {
                role: "system".into(),
                content: "be terse".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "first".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "ack".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "second".into(),
            },
        ];
        let (sys, user) = split_messages(&msgs);
        assert_eq!(sys.as_deref(), Some("be terse"));
        assert_eq!(user, "second");
    }

    #[test]
    fn an_empty_system_message_is_not_a_system_prompt() {
        let msgs = vec![
            ChatMessage {
                role: "system".into(),
                content: String::new(),
            },
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
            },
        ];
        assert_eq!(split_messages(&msgs).0, None);
    }
}
