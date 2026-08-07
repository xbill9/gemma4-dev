#!/usr/bin/env python3.12
"""OpenAI-compatible HTTP/SSE server backed by a compiled static-shape TPU decode loop.

Routes: /v1/models, /v1/chat/completions, /v1/completions, /health.

Why not HF generate(): compiled mode requires every graph argument on-device, and
generate() feeds CPU scalars in. Why not a growing [1, cur_len] view: compile is
dynamic=False, so a changing sequence dimension recompiles on every token. This
server keeps ONE [B, SEQ] buffer for the life of the process, writes each new
token with index_copy_ at a TENSOR position, and compiles a single step function.

Run on the TPU VM with the dedicated interpreter:

    python3.12 tpu_openai_server.py --model google/gemma-4-E2B-it --port 8000

Gated checkpoints need a Hugging Face token: set HF_TOKEN, or leave it unset and
the VM service account fetches the `hf-token` Secret Manager secret.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import time
import urllib.request
import uuid

METADATA = "http://metadata.google.internal/computeMetadata/v1"
# Secret Manager secret holding the Hugging Face token. The startup script fetches it by
# id at boot, so a rotated or per-project secret only needs this to change.
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "hf-token")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tpu_openai_server")

DEFAULT_MODEL = "google/gemma-4-E2B-it"

# Request models live at module scope on purpose: with `from __future__ import
# annotations` every annotation is a string, and FastAPI resolves them against
# module globals. Nested inside create_app they are invisible to that lookup and
# each body param silently degrades to a query param ("field required: query.req").
from pydantic import BaseModel, Field  # noqa: E402


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: list[ChatMessage]
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=128, ge=1)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    prompt: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=128, ge=1)
    stream: bool = False


def _metadata(path: str) -> str:
    req = urllib.request.Request(f"{METADATA}/{path}", headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=10) as res:
        return res.read().decode()


def load_hf_token() -> None:
    """Best-effort: populate HF_TOKEN from Secret Manager via the VM service account."""
    if os.environ.get("HF_TOKEN"):
        return
    try:
        project = _metadata("project/project-id")
        access = json.loads(_metadata("instance/service-accounts/default/token"))["access_token"]
        url = (
            f"https://secretmanager.googleapis.com/v1/projects/{project}"
            f"/secrets/{HF_SECRET_ID}/versions/latest:access"
        )
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access}"})
        with urllib.request.urlopen(req, timeout=10) as res:
            payload = json.load(res)["payload"]["data"]
        os.environ["HF_TOKEN"] = base64.b64decode(payload).decode()
        logger.info("HF token loaded from Secret Manager.")
    except Exception:
        logger.info("No HF token from Secret Manager; relying on cache or public weights.")


class StaticDecodeEngine:
    """Greedy/top-p decode over a fixed [1, SEQ] device buffer, single compiled step.

    Shapes never change after warmup, so torch.compile(backend="tpu",
    dynamic=False) compiles exactly once and every later request reuses it.
    """

    def __init__(self, model_id: str, seq: int = 256) -> None:
        import torch
        import transformers

        self.torch = torch
        self.model_id = model_id
        self.seq = seq
        self.device = torch.device("tpu")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        t0 = time.monotonic()
        self.model = (
            transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
            .to(self.device)
            .eval()
        )
        logger.info("Loaded %s in %.1fs", model_id, time.monotonic() - t0)

        self.pad_id = self.tokenizer.pad_token_id or 0
        self.stop_ids = {self.tokenizer.eos_token_id}
        for tok in ("<end_of_turn>", "<eos>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                self.stop_ids.add(tid)
        self.stop_ids.discard(None)

        def step(tokens, mask, last_idx):
            logits = self.model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
            return logits.index_select(1, last_idx)  # [1, 1, vocab]

        self.step = torch.compile(step, backend="tpu")
        self._one = torch.ones((1, 1), dtype=torch.long).to(self.device)
        self.lock = asyncio.Lock()  # one TPU process -> serialize requests
        self.warm = False

    def warmup(self) -> None:
        """Pay the compile cost once at boot, not on the first user request."""
        torch = self.torch
        t0 = time.monotonic()
        tokens = torch.full((1, self.seq), self.pad_id, dtype=torch.long).to(self.device)
        mask = torch.zeros((1, self.seq), dtype=torch.long).to(self.device)
        mask[:, :8] = 1
        pos = torch.tensor([8], dtype=torch.long).to(self.device)
        with torch.no_grad():
            for _ in range(2):
                logits = self.step(tokens, mask, pos - 1)
                nxt = logits.argmax(-1)
                tokens = tokens.index_copy(1, pos, nxt)
                mask = mask.index_copy(1, pos, self._one)
                pos = pos + 1
            tokens[0, :1].cpu()
        self.warm = True
        logger.info("Warmup (compile) done in %.1fs", time.monotonic() - t0)

    def encode_text(self, prompt: str) -> list[int]:
        """Tokenize a raw prompt, forcing BOS.

        Gemma's tokenizer does not prepend <bos> for a plain __call__ even with
        add_special_tokens=True, and the model is trained with it — without BOS
        greedy decoding degenerates into a repetition loop. apply_chat_template
        inserts it already, so only this raw path needs the fixup.
        """
        ids = self.tokenizer(prompt)["input_ids"]
        bos = self.tokenizer.bos_token_id
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
        return ids

    def encode_chat(self, messages: list[dict]) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )["input_ids"]
        return ids[0].tolist()

    def generate(self, prompt_ids: list[int], max_tokens: int, temperature: float, top_p: float):
        """Yields (token_id, text_piece) per step. Caller must hold self.lock."""
        torch = self.torch
        n_prompt = len(prompt_ids)
        budget = min(max_tokens, self.seq - n_prompt - 1)
        if budget <= 0:
            raise ValueError(
                f"prompt is {n_prompt} tokens; buffer is {self.seq} with no room to decode. "
                f"Start the server with a larger --seq."
            )

        tokens = torch.full((1, self.seq), self.pad_id, dtype=torch.long)
        tokens[0, :n_prompt] = torch.tensor(prompt_ids, dtype=torch.long)
        tokens = tokens.to(self.device)
        mask = torch.zeros((1, self.seq), dtype=torch.long)
        mask[0, :n_prompt] = 1
        mask = mask.to(self.device)
        pos = torch.tensor([n_prompt], dtype=torch.long).to(self.device)

        with torch.no_grad():
            for _ in range(budget):
                logits = self.step(tokens, mask, pos - 1)
                if temperature and temperature > 0.0:
                    probs = torch.softmax(logits[:, 0, :].float() / max(temperature, 1e-5), dim=-1)
                    if 0.0 < top_p < 1.0:
                        srt, idx = torch.sort(probs, dim=-1, descending=True)
                        keep = (srt.cumsum(-1) - srt) < top_p
                        srt = srt * keep
                        srt = srt / srt.sum(-1, keepdim=True).clamp_min(1e-9)
                        pick = torch.multinomial(srt, num_samples=1)
                        nxt = idx.gather(-1, pick)
                    else:
                        nxt = torch.multinomial(probs, num_samples=1)
                else:
                    nxt = logits.argmax(-1)[:, 0:1]

                tid = int(nxt.cpu().item())  # one sync per token: needed to stream and stop
                if tid in self.stop_ids:
                    break
                yield tid, self.tokenizer.decode([tid], skip_special_tokens=True)

                tokens = tokens.index_copy(1, pos, nxt.view(1, 1))
                mask = mask.index_copy(1, pos, self._one)
                pos = pos + 1


def create_app(engine: StaticDecodeEngine):
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    app = FastAPI(title="TPU OpenAI-compatible server", version="1.0.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": engine.model_id, "warm": engine.warm, "seq": engine.seq}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": engine.model_id, "object": "model", "owned_by": "local"}],
        }

    def _run(prompt_ids, req):
        pieces, ids = [], []
        for tid, piece in engine.generate(prompt_ids, req.max_tokens, req.temperature, req.top_p):
            ids.append(tid)
            pieces.append(piece)
        return "".join(pieces), len(ids)

    async def _stream_chat(prompt_ids, req, cid, created):
        head = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(head)}\n\n"
        async with engine.lock:
            for _tid, piece in engine.generate(
                prompt_ids, req.max_tokens, req.temperature, req.top_p
            ):
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0)
        tail = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(tail)}\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        prompt_ids = engine.encode_chat([m.model_dump() for m in req.messages])
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        if req.stream:
            return StreamingResponse(
                _stream_chat(prompt_ids, req, cid, created), media_type="text/event-stream"
            )
        async with engine.lock:
            try:
                text, n_out = await asyncio.to_thread(_run, prompt_ids, req)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": n_out,
                "total_tokens": len(prompt_ids) + n_out,
            },
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        prompt_ids = engine.encode_text(req.prompt)
        async with engine.lock:
            try:
                text, n_out = await asyncio.to_thread(_run, prompt_ids, req)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": n_out,
                "total_tokens": len(prompt_ids) + n_out,
            },
        }

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seq", type=int, default=256, help="static buffer length")
    args = parser.parse_args()

    load_hf_token()
    import uvicorn

    engine = StaticDecodeEngine(args.model, seq=args.seq)
    engine.warmup()
    app = create_app(engine)
    logger.info("Serving %s on %s:%d (SEQ=%d)", args.model, args.host, args.port, args.seq)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
