"""Option A: OpenAI-Compatible HTTP / SSE Server for the PyTorch TPU backend.

Provides OpenAI API routes (/v1/chat/completions, /v1/completions, /health, /metrics)
powered by the Option A TPUEngine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from tpu_engine import TPUEngine

logger = logging.getLogger("tpu_server")


# Request/Response schemas
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "google/gemma-4-E2B-it"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=128, ge=1)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "google/gemma-4-E2B-it"
    prompt: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=128, ge=1)
    stream: bool = False


def create_app(engine: Any = None) -> FastAPI:
    """Create FastAPI app with Option A TPUEngine dependency."""
    app = FastAPI(title="the PyTorch TPU backend Option A Server", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.engine = engine

    @app.get("/health")
    async def health():
        return {"status": "healthy", "backend": "Option A the PyTorch TPU backend"}

    @app.get("/metrics")
    async def metrics():
        return {"uptime_seconds": time.monotonic(), "status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "google/gemma-4-E2B-it",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "tpu",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        engine_inst = app.state.engine
        if engine_inst is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        # Process prompt text
        prompt_text = "\n".join([f"{m.role}: {m.content}" for m in request.messages])
        if hasattr(engine_inst, "tokenizer") and engine_inst.tokenizer is not None:
            prompt_tokens = engine_inst.tokenizer.encode(prompt_text)
        else:
            prompt_tokens = [ord(c) for c in prompt_text[:128]]  # Fallback for testing

        request_id = f"chatcmpl-{int(time.time()*1000)}"

        if request.stream:

            async def event_generator() -> AsyncGenerator[str, None]:
                gen = engine_inst.generate(
                    prompt_tokens,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                )
                for token_id in gen:
                    chunk_text = (
                        engine_inst.tokenizer.decode([token_id])
                        if hasattr(engine_inst, "tokenizer") and engine_inst.tokenizer
                        else chr(token_id % 256)
                    )
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0)

                # Final DONE frame
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Non-streaming
        generated_tokens = list(
            engine_inst.generate(
                prompt_tokens,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        )
        if hasattr(engine_inst, "tokenizer") and engine_inst.tokenizer:
            completion_text = engine_inst.tokenizer.decode(generated_tokens)
        else:
            completion_text = "".join([chr(t % 256) for t in generated_tokens])

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(generated_tokens),
                "total_tokens": len(prompt_tokens) + len(generated_tokens),
            },
        }

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
