#!/usr/bin/env python3
"""OpenAI-Compatible FastAPI Server for pure JAX Gemma 4.

Generation runs entirely on the pure-JAX engine in ``jax_engine.py`` (no
PyTorch, no torch_xla) against a static KV cache, so every streamed token
attends to the full history.

This file is shared with the TPU rig, so it does NOT hardcode a chip, a
checkpoint, or a precision: the engine resolves those from the device and the
weights, and reports what it actually got via ``ENGINE.precision_info()``. The
header here used to claim "TPU v6e-1", a "-qat-w4a16-ct" checkpoint and "BF16
activations" -- all three wrong on the G5g rig, whose T4G has no bf16 datapath
at all and which serves the dense reference build.

- Precision: reported at runtime by GET /health and the
  ``tpu_jax_precision_info`` series on GET /metrics. Read it there, not here.
- Endpoints:
  - GET  /health
  - GET  /metrics  (Prometheus format metrics)
  - GET  /v1/models
  - POST /v1/chat/completions
  - POST /v1/completions
"""

import argparse
import json
import os
import time

import jax
from pydantic import BaseModel

# Matmul precision is a per-platform decision and jax_e_model makes it from the
# live device. This file used to set "bfloat16" unconditionally, which is right
# on a TPU MXU and wrong on Turing — it tells XLA it may demote fp32 matmul
# inputs to a format the chip has no unit for. Importing the model module below
# applies the correct policy; do not re-set it here.

# Persistent XLA compilation disk cache (skips ~17s of compilation on restarts,
# measured on TPU v6e-1; unmeasured on this rig).
_cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.path.expanduser(
    "~/.cache/jax_compilation_cache"
)
os.makedirs(_cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# ruff: noqa: E402 — these must follow the jax.config calls above.
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse

from jax_engine import GenerationStats, JaxGemmaEngine

# Global state
ENGINE: JaxGemmaEngine | None = None
TOKENIZER = None
# Defaults for this rig: the *reference* instruction-tuned checkpoint, not the
# QAT export the TPU rig serves. E2B is 9.5 GiB of weights and the T4G has
# 15360 MiB, so the dense model fits with room for a real KV pool, and the fused
# W4A16 kernel does not fit Turing shared memory anyway (jax_e_model).
# Every default here is overridable from tpu.env via the launcher.
MODEL_ID = os.environ.get("MODEL_NAME", "google/gemma-4-E2B-it")
KV_CACHE_DTYPE = os.environ.get("KV_CACHE_DTYPE", "auto")

METRICS = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "prompt_tokens_total": 0,
    "completion_tokens_total": 0,
    "total_latency_seconds": 0.0,
    "last_tokens_per_second": 0.0,
    "last_prefill_ms": 0.0,
    "degenerate_responses": 0,
}

app = FastAPI(title="Pure JAX Gemma 4 Server")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = 128
    temperature: float | None = 0.7
    top_k: int | None = 40
    stream: bool | None = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    max_tokens: int | None = 128
    temperature: float | None = 0.0
    top_k: int | None = 40
    stream: bool | None = False


def fetch_hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        import base64
        import urllib.request

        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            project = res.read().decode()
        token_req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(token_req, timeout=5) as res:
            access_token = json.loads(res.read().decode())["access_token"]
        secret_url = (
            f"https://secretmanager.googleapis.com/v1/projects/{project}"
            "/secrets/hf-token/versions/latest:access"
        )
        sec_req = urllib.request.Request(
            secret_url, headers={"Authorization": f"Bearer {access_token}"}
        )
        with urllib.request.urlopen(sec_req, timeout=5) as res:
            data = json.load(res)["payload"]["data"]
            token = base64.b64decode(data).decode()
            os.environ["HF_TOKEN"] = token
            print("Successfully fetched HF_TOKEN from GCP Secret Manager.")
            return token
    except Exception:
        return None


def load_engine(
    model_id: str,
    kv_dtype: str = "auto",
    quant_mode: str = "auto",
    max_model_len: int = 4096,
    local_dir: str | None = None,
    ple_bits: int = 0,
    int8_lm_head: bool = False,
    prefill_chunk_size: int | None = None,
    window_kv: bool | None = None,
):
    """Load the engine. Defaults are "auto", NOT the TPU rig's bf16/w4a16.

    Those literals were inherited from the TPU rig and are wrong on any chip
    without a bf16 datapath: kv_dtype="bf16" reaches resolve_cache_dtype, which
    raises on pre-Ampere, and quant_mode="w4a16" against a dense checkpoint
    loads garbage rather than failing. Latent so far only because the CLI always
    passes both explicitly -- so a second caller was one keyword away from the
    bug. "auto" resolves each from the device and the checkpoint.

    prefill_chunk_size bounds prefill temporaries, which JaxGemmaEngine documents
    as LINEAR in the prompt tokens admitted in one pass at ~2.13 MB/token -- a
    figure measured on v6e-1, NOT on GPU. One-shot prefill does OOM: measured
    2026-08-23 on a T4G, dense served 115 tokens and failed at 2,015, quantised
    served 3,515 and failed at 5,015. Those are brackets, not thresholds, and both
    failures land BELOW what the formula predicts (2,262 and 5,174), so it
    over-predicts capacity. Do not size a deployment from it. None keeps the old
    one-shot behaviour.
    """
    global ENGINE, TOKENIZER, MODEL_ID, KV_CACHE_DTYPE
    MODEL_ID, KV_CACHE_DTYPE = model_id, kv_dtype
    fetch_hf_token()

    print(f"JAX devices: {jax.devices()}")

    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {model_id}")
    TOKENIZER = AutoTokenizer.from_pretrained(model_id)

    # Say what is actually being loaded. This line used to read "Loading W4A16 QAT
    # weights" unconditionally — inherited from the TPU rig, which serves the
    # -qat-w4a16-ct export. On this rig the checkpoint is dense fp16, so that line
    # told an operator the box had done the one thing this rig refuses to do.
    print(f"Loading weights into JAX: {model_id} (quant_mode={quant_mode})")
    t0 = time.perf_counter()
    engine = JaxGemmaEngine(
        model_id=model_id,
        kv_cache_dtype=kv_dtype,
        quant_mode=quant_mode,
        max_model_len=max_model_len,
        ple_bits=ple_bits,
        int8_lm_head=int8_lm_head,
        prefill_chunk_size=prefill_chunk_size,
        window_kv=window_kv,
    )
    engine.load(local_dir=local_dir)
    engine.bos_token_id = getattr(TOKENIZER, "bos_token_id", None)
    ENGINE = engine
    load_s = time.perf_counter() - t0
    print(
        f"Loaded {engine.weight_bytes / 1e9:.2f} GB of parameters on {engine.device} "
        f"in {load_s:.1f}s (KV cache: {kv_dtype})"
    )


def _eos_ids() -> list[int]:
    ids = []
    for attr in ("eos_token_id", "pad_token_id"):
        val = getattr(TOKENIZER, attr, None)
        if isinstance(val, int):
            ids.append(val)
        elif isinstance(val, list):
            ids.extend(v for v in val if isinstance(v, int))
    # Gemma chat turns terminate on the turn-end marker, but its spelling differs
    # by checkpoint: <end_of_turn> on some, <turn|> on the QAT E2B ones. A name
    # absent from the vocab does not raise -- convert_tokens_to_ids returns
    # unk_token_id, which is >= 0 and so passed the old guard. That put <unk> in
    # the stop set while leaving the REAL terminator out of it.
    unk = getattr(TOKENIZER, "unk_token_id", None)
    for name in ("<end_of_turn>", "<turn|>"):
        try:
            turn_end = TOKENIZER.convert_tokens_to_ids(name)
        except Exception:
            continue
        if isinstance(turn_end, int) and turn_end >= 0 and turn_end != unk:
            ids.append(turn_end)
    return sorted(set(ids))


def _require_ready():
    if ENGINE is None or not ENGINE.is_ready or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="JAX engine is loading")


def looks_degenerate(text: str) -> bool:
    """Heuristic: did the model emit a token loop rather than an answer?

    MEASURED 2026-08-23: a prompt whose bucket padding reaches 512 makes the model
    emit "TheTheThe..." while the server records status="success" -- worse than a
    500, because nothing in the metrics, the health check or the benchmark harness
    could tell it from a good answer. (The mechanism, KV-ring eviction starving
    the 28 sliding layers, is inferred; see docs/padding-window-eviction.md. The
    counter here does not depend on that being the right explanation.)

    Deliberately conservative: it fires on a whole response collapsing to one or
    two distinct tokens, not on merely repetitive prose, because a false positive
    here would discredit a real result. It is a smoke alarm, not a quality score,
    and it does NOT change the response or the status code.
    """
    body = (text or "").strip()
    if len(body) <= 40:
        return False
    words = body.split()
    if len(words) >= 8 and len(set(words)) <= 2:
        return True
    # Catches runs with no whitespace at all, e.g. "TheTheThe..."
    return len(set(body.replace(" ", ""))) <= 6


def _record(stats: GenerationStats, elapsed: float, text: str | None = None):
    METRICS["successful_requests"] += 1
    if text is not None and looks_degenerate(text):
        METRICS["degenerate_responses"] += 1
    METRICS["prompt_tokens_total"] += stats.prompt_tokens
    METRICS["completion_tokens_total"] += stats.completion_tokens
    METRICS["total_latency_seconds"] += elapsed
    METRICS["last_tokens_per_second"] = stats.decode_tok_per_s
    METRICS["last_prefill_ms"] = stats.prefill_ms


def _sse_stream(prompt_ids, req, req_id: str, object_name: str, t0: float):
    """Shared SSE generator for chat and text completions."""
    created = int(time.time())

    def emit(delta_field: dict, finish=None):
        chunk = {
            "id": req_id,
            "object": object_name,
            "created": created,
            "model": req.model or MODEL_ID,
            "choices": [{"index": 0, **delta_field, "finish_reason": finish}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    is_chat = object_name == "chat.completion.chunk"
    stats: GenerationStats | None = None
    pieces: list[str] = []
    for item in ENGINE.generate_stream(
        prompt_ids,
        max_new_tokens=req.max_tokens or 128,
        temperature=req.temperature if req.temperature is not None else 0.0,
        top_k=req.top_k or 40,
        eos_token_ids=_eos_ids(),
    ):
        if isinstance(item, GenerationStats):
            stats = item
            break
        text = TOKENIZER.decode([item], skip_special_tokens=True)
        pieces.append(text)
        yield emit({"delta": {"content": text}} if is_chat else {"text": text})

    if stats is not None:
        _record(stats, time.time() - t0, "".join(pieces))
        finish = stats.finish_reason
    else:
        finish = "stop"
    yield emit({"delta": {}} if is_chat else {"text": ""}, finish=finish)
    yield "data: [DONE]\n\n"


@app.get("/health")
def health():
    ready = ENGINE is not None and ENGINE.is_ready
    payload = {
        "status": "ok" if ready else "loading",
        "backend": "jax",
        "device": str(ENGINE.device) if ready else None,
        "model": MODEL_ID,
    }
    if not ready:
        return payload
    # Read precision back off the engine. This block used to hardcode
    # activations="bfloat16" and weights="bf16" — both inherited from the TPU
    # rig and both impossible on Turing, which has no bf16 datapath at all. It
    # also reported the REQUESTED kv dtype, hiding what "auto" resolved to.
    info = ENGINE.precision_info()
    payload["precision"] = {
        "weights": "w4_int4" if info["quant_mode"] == "w4a16" else info["compute_dtype"],
        "activations": info["compute_dtype"],
        "kv_cache": info["kv_cache_dtype"],
        "kv_cache_requested": info["kv_cache_requested"],
        "quant_mode": info["quant_mode"],
        "ple_bits": info["ple_bits"],
        "int8_lm_head": info["int8_lm_head"],
        "pre_ampere": info["pre_ampere"],
    }
    return payload


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    ready = ENGINE is not None and ENGINE.is_ready
    mem = ENGINE.memory_stats() if ready else {}
    device = str(ENGINE.device) if ready else "unknown"
    info = ENGINE.precision_info() if ready else {}
    # Prometheus "info" convention: the labels carry the payload, the value is
    # always 1. Emitted only when the engine is up, because before that every
    # dtype would be a guess about config rather than a fact about the device.
    precision = (
        [
            "# HELP tpu_jax_precision_info Dtypes and quantisation resolved on device",
            "# TYPE tpu_jax_precision_info gauge",
            "tpu_jax_precision_info{"
            + ",".join(
                [
                    f'model="{MODEL_ID}"',
                    f'compute_dtype="{info["compute_dtype"]}"',
                    f'quant_mode="{info["quant_mode"]}"',
                    f'kv_cache_dtype="{info["kv_cache_dtype"]}"',
                    f'kv_cache_requested="{info["kv_cache_requested"]}"',
                    f'ple_bits="{info["ple_bits"]}"',
                    f'int8_lm_head="{str(info["int8_lm_head"]).lower()}"',
                    f'pre_ampere="{str(info["pre_ampere"]).lower()}"',
                ]
            )
            + "} 1",
            "",
        ]
        if ready
        else []
    )
    lines = [
        *precision,
        "# HELP tpu_jax_requests_total Total HTTP requests processed by JAX TPU server",
        "# TYPE tpu_jax_requests_total counter",
        f'tpu_jax_requests_total{{model="{MODEL_ID}",status="success"}} {METRICS["successful_requests"]}',
        f'tpu_jax_requests_total{{model="{MODEL_ID}",status="failed"}} {METRICS["failed_requests"]}',
        "",
        "# HELP tpu_jax_degenerate_responses_total Responses that collapsed to a token loop",
        "# TYPE tpu_jax_degenerate_responses_total counter",
        f'tpu_jax_degenerate_responses_total{{model="{MODEL_ID}"}} {METRICS["degenerate_responses"]}',
        "",
        "# HELP tpu_jax_prompt_tokens_total Total prompt tokens processed",
        "# TYPE tpu_jax_prompt_tokens_total counter",
        f'tpu_jax_prompt_tokens_total{{model="{MODEL_ID}"}} {METRICS["prompt_tokens_total"]}',
        "",
        "# HELP tpu_jax_completion_tokens_total Total completion tokens generated",
        "# TYPE tpu_jax_completion_tokens_total counter",
        f'tpu_jax_completion_tokens_total{{model="{MODEL_ID}"}} {METRICS["completion_tokens_total"]}',
        "",
        "# HELP tpu_jax_latency_seconds_sum Total generation latency sum",
        "# TYPE tpu_jax_latency_seconds_sum counter",
        f'tpu_jax_latency_seconds_sum{{model="{MODEL_ID}"}} {METRICS["total_latency_seconds"]:.3f}',
        "",
        "# HELP tpu_jax_decode_tokens_per_second Decode throughput of the last request",
        "# TYPE tpu_jax_decode_tokens_per_second gauge",
        f'tpu_jax_decode_tokens_per_second{{model="{MODEL_ID}"}} {METRICS["last_tokens_per_second"]:.1f}',
        "",
        "# HELP tpu_jax_prefill_milliseconds Prefill (TTFT) of the last request",
        "# TYPE tpu_jax_prefill_milliseconds gauge",
        f'tpu_jax_prefill_milliseconds{{model="{MODEL_ID}"}} {METRICS["last_prefill_ms"]:.1f}',
        "",
        "# HELP tpu_jax_weight_bytes Parameter footprint resident on device",
        "# TYPE tpu_jax_weight_bytes gauge",
        f'tpu_jax_weight_bytes{{model="{MODEL_ID}"}} {mem.get("weight_bytes", 0)}',
        "",
        "# HELP tpu_jax_hbm_used_bytes High Bandwidth Memory used in bytes",
        "# TYPE tpu_jax_hbm_used_bytes gauge",
        f'tpu_jax_hbm_used_bytes{{device="{device}"}} {mem.get("hbm_bytes_in_use", 0)}',
        "",
        "# HELP tpu_jax_hbm_limit_bytes High Bandwidth Memory total limit in bytes",
        "# TYPE tpu_jax_hbm_limit_bytes gauge",
        f'tpu_jax_hbm_limit_bytes{{device="{device}"}} {mem.get("hbm_bytes_limit", 0)}',
        "",
    ]
    return "\n".join(lines)


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "jax-tpu"}
        ],
    }


def _chat_prompt_ids(messages) -> list[int]:
    formatted = [{"role": m.role, "content": m.content} for m in messages]
    if hasattr(TOKENIZER, "apply_chat_template"):
        encoded = TOKENIZER.apply_chat_template(
            formatted, tokenize=True, add_generation_prompt=True)
        # Transformers 5 may return a BatchEncoding/dict here instead of the
        # bare token list returned by earlier versions.
        if hasattr(encoded, "keys") and "input_ids" in encoded:
            encoded = encoded["input_ids"]
        return list(encoded)
    text = "\n".join(f"{m.role}: {m.content}" for m in messages)
    return TOKENIZER(text)["input_ids"]


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    _require_ready()
    METRICS["total_requests"] += 1
    t0 = time.time()
    req_id = f"chatcmpl-jax-{int(t0 * 1000)}"
    try:
        prompt_ids = _chat_prompt_ids(req.messages)

        if req.stream:
            return StreamingResponse(
                _sse_stream(prompt_ids, req, req_id, "chat.completion.chunk", t0),
                media_type="text/event-stream",
            )

        tokens, stats = ENGINE.generate(
            prompt_ids,
            max_new_tokens=req.max_tokens or 128,
            temperature=req.temperature if req.temperature is not None else 0.0,
            top_k=req.top_k or 40,
            eos_token_ids=_eos_ids(),
        )
        elapsed = time.time() - t0
        text = TOKENIZER.decode(tokens, skip_special_tokens=True)
        _record(stats, elapsed, text)

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": int(t0),
            "model": req.model or MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text.strip()},
                    "finish_reason": stats.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.completion_tokens,
                "total_tokens": stats.prompt_tokens + stats.completion_tokens,
                "latency_seconds": round(elapsed, 3),
                "prefill_ms": round(stats.prefill_ms, 1),
                "decode_tokens_per_second": round(stats.decode_tok_per_s, 1),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        METRICS["failed_requests"] += 1
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/completions")
def text_completions(req: CompletionRequest):
    _require_ready()
    METRICS["total_requests"] += 1
    t0 = time.time()
    req_id = f"cmpl-jax-{int(t0 * 1000)}"
    try:
        prompt_text = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        prompt_ids = TOKENIZER(prompt_text)["input_ids"]

        if req.stream:
            return StreamingResponse(
                _sse_stream(prompt_ids, req, req_id, "text_completion", t0),
                media_type="text/event-stream",
            )

        tokens, stats = ENGINE.generate(
            prompt_ids,
            max_new_tokens=req.max_tokens or 128,
            temperature=req.temperature if req.temperature is not None else 0.0,
            top_k=req.top_k or 40,
            eos_token_ids=_eos_ids(),
        )
        elapsed = time.time() - t0
        text = TOKENIZER.decode(tokens, skip_special_tokens=True)
        _record(stats, elapsed, text)

        return {
            "id": req_id,
            "object": "text_completion",
            "created": int(t0),
            "model": req.model or MODEL_ID,
            "choices": [{"text": text.strip(), "index": 0, "finish_reason": stats.finish_reason}],
            "usage": {
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.completion_tokens,
                "total_tokens": stats.prompt_tokens + stats.completion_tokens,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        METRICS["failed_requests"] += 1
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--kv-cache-dtype", default=KV_CACHE_DTYPE)
    parser.add_argument("--quant-mode", default=os.environ.get("QUANT_MODE", "auto"),
                        choices=["auto", "w4a16", "fp16"],
                        help="auto reads it off the checkpoint name")
    parser.add_argument("--max-model-len", type=int,
                        default=int(os.environ.get("MAX_MODEL_LEN", "4096")))
    parser.add_argument("--local-dir", default=None,
                        help="Load from a local checkpoint dir instead of the Hub")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("JAX_PORT", "8000")))
    parser.add_argument(
        "--ple-bits", type=int, default=0, choices=[0, 4, 8],
        help="Quantize the per-layer-embedding table. 0 = off.")
    parser.add_argument(
        "--int8-lm-head", action="store_true",
        help="Quantize the LM head to int8. NOT numerics-preserving.")
    parser.add_argument(
        "--window-kv", dest="window_kv", default=None,
        choices=["auto", "on", "off"],
        help="Ring-buffer KV for sliding layers. Default auto = on whenever "
             "max_model_len > sliding_window (verified: True at 8192 vs 512). "
             "MEASURED 2026-08-23 with it on: a prompt whose bucket padding reaches "
             "512 makes the model emit a token loop. Ring eviction is the inferred "
             "mechanism; 'off' is untested and exists so it can be compared.")
    parser.add_argument(
        "--prefill-chunk-size", type=int,
        default=(int(os.environ["PREFILL_CHUNK_SIZE"])
                 if os.environ.get("PREFILL_CHUNK_SIZE") else None),
        help="Split prefill into chunks of this many tokens. One-shot prefill OOMs "
             "on long prompts (measured: dense failed at 2,015 tokens, quantised at "
             "5,015). Unset = one-shot (previous behaviour).")
    args = parser.parse_args()

    load_engine(
        args.model, args.kv_cache_dtype, args.quant_mode,
        args.max_model_len, args.local_dir, args.ple_bits, args.int8_lm_head,
        args.prefill_chunk_size,
        {"auto": None, "on": True, "off": False}[args.window_kv or "auto"],
    )
    uvicorn.run(app, host=args.host, port=args.port)
