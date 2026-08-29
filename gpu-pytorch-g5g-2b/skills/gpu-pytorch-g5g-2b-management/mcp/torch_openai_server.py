#!/usr/bin/env python3.12
"""OpenAI-compatible HTTP/SSE server for Gemma 4 on an NVIDIA T4G, under PyTorch.

Routes: /v1/models, /v1/chat/completions, /v1/completions, /health, /metrics.

Decode keeps an HF KV cache and feeds one token per step -- the ordinary CUDA
path. The docstring here used to argue for a compiled static-shape buffer with
`use_cache=False`, which is the correct discipline on XLA and pure loss on CUDA:
nothing recompiles here, so a static buffer only means every decoded token pays a
full-context forward. The `backend="tpu"` that went with it does not exist in a
CUDA build and raised during warmup, before the port was ever bound.

/metrics exposes prefill and decode SEPARATELY. Quote
`tpu_jax_decode_tokens_per_second`, not an end-to-end rate: the latter carries
prefill and the HTTP round trip, and the sibling rigs' reports all compare on the
gauge. The `tpu_jax_` prefix is an identifier, not a description -- see Metrics.

Run on the G5g instance with the interpreter that holds the DLAMI's torch (the
bootstrap records it in /opt/torch-g5g/PYTHON_BIN):

    python3.12 torch_openai_server.py --model google/gemma-4-E2B-it --port 8000

Gated checkpoints need a Hugging Face token: set HF_TOKEN, or leave it unset and
cloud-init puts the Hugging Face token in a root-only systemd EnvironmentFile.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    # torch is a SERVING dependency, imported lazily inside the functions
    # that need it so the control plane can import this module with no CUDA.
    import torch

# Secret Manager secret holding the Hugging Face token. The startup script fetches it by
# id at boot, so a rotated or per-project secret only needs this to change.
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "hf-token")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("torch_openai_server")

DEFAULT_MODEL = "google/gemma-4-E2B-it"

# Request models live at module scope on purpose: with `from __future__ import
# annotations` every annotation is a string, and FastAPI resolves them against
# module globals. Nested inside create_app they are invisible to that lookup and
# each body param silently degrades to a query param ("field required: query.req").


def resolve_compute_dtype(device: torch.device) -> torch.dtype:
    """Pick the 16-bit dtype the DEVICE actually has, not the one in a config.

    This is the single most expensive lesson in the sibling JAX rig, ported here
    because PyTorch fails the same way: **bfloat16 on a pre-Ampere GPU does not
    raise.** CUDA accepts it and emulates through fp32, and the only symptom is
    that most of decode disappears into conversion. Turing (SM 7.5, the T4G in
    this box) has no bf16 datapath and no fp8; float16 is its only real 16-bit
    path.

    So the dtype is read off the live device rather than taken on trust, and the
    resolved value is logged at startup so a misconfiguration is one grep away
    instead of a mystery in the throughput.
    """
    import torch

    if device.type != "cuda":
        return torch.float32
    major, minor = torch.cuda.get_device_capability(device)
    pre_ampere = (major, minor) < (8, 0)
    dtype = torch.float16 if pre_ampere else torch.bfloat16
    logging.info(
        "torch device policy: name=%s compute_capability=%d.%d pre_ampere=%s "
        "compute_dtype=%s",
        torch.cuda.get_device_name(device), major, minor, pre_ampere,
        str(dtype).replace("torch.", ""),
    )
    return dtype


# Deferred like torch: pydantic is a SERVING dependency and the control plane
# imports this module without it. E402 is expected and suppressed on purpose.
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


def load_hf_token() -> None:
    """Read HF_TOKEN from the environment. Nothing is fetched at request time.

    REWRITTEN FOR EC2. The TPU sibling this file came from called GCE metadata
    and Google Secret Manager -- `metadata.google.internal` does not resolve on
    a G5g, so that path could only ever have failed here, silently, and left
    every gated download unauthenticated.

    On this rig the token arrives the way the JAX sibling established: cloud-init
    fetches it from AWS Secrets Manager at boot into a ROOT-ONLY systemd
    EnvironmentFile, which is why it is already in the process environment by the
    time this runs. It is deliberately never put in user data -- instance
    metadata is readable by anything on the box.
    """
    if os.environ.get("HF_TOKEN"):
        logger.info("HF token present in the environment.")
        return
    logger.info(
        "No HF_TOKEN in the environment; relying on the local cache or public "
        "weights. If a gated download 401s, check the systemd EnvironmentFile."
    )


class KVCacheDecodeEngine:
    """Greedy/top-p decode over a growing HF KV cache. One request at a time.

    REWRITTEN 2026-08-29. The forked version was a TPU design carried over
    literally, and two lines of it could not run on this box at all:

      * `torch.compile(step, backend="tpu")` -- there is no "tpu" backend in a
        CUDA build, so this raised at __init__ and killed the process during
        warmup, before uvicorn ever bound the port.
      * `use_cache=False` over a static [1, SEQ] buffer. That is the right shape
        discipline for XLA, where a changing sequence dimension recompiles. On
        CUDA nothing recompiles, and paying a full SEQ-length forward for every
        single token is pure loss -- at --seq 4096 each decoded token would have
        re-read the whole 4,096-token context.

    So this uses the ordinary CUDA decode: prefill the prompt once, keep
    `past_key_values`, and feed one token per step. Cost per step is then the
    weights plus one token of KV, which is what makes the throughput here
    comparable to the JAX sibling's `tpu_jax_decode_tokens_per_second`.

    `seq` survives as the context BOUND (it caps prompt+output) rather than as a
    padded buffer length, so --seq no longer costs anything to raise.
    """

    def __init__(self, model_id: str, seq: int = 4096) -> None:
        import torch
        import transformers

        self.torch = torch
        self.model_id = model_id
        self.seq = seq
        self.device = torch.device("cuda")
        self.dtype = resolve_compute_dtype(self.device)
        major, minor = torch.cuda.get_device_capability(self.device)
        self.capability = f"{major}.{minor}"
        self.pre_ampere = (major, minor) < (8, 0)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        t0 = time.monotonic()
        self.model = (
            transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        self.load_seconds = time.monotonic() - t0
        self.weight_bytes = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )
        logger.info(
            "Loaded %s in %.1fs: weights=%.3f GB dtype=%s",
            model_id, self.load_seconds, self.weight_bytes / 1e9,
            str(self.dtype).replace("torch.", ""),
        )

        self.pad_id = self.tokenizer.pad_token_id or 0
        self.stop_ids = {self.tokenizer.eos_token_id}
        for tok in ("<end_of_turn>", "<eos>"):
            tid = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                self.stop_ids.add(tid)
        self.stop_ids.discard(None)

        self.lock = asyncio.Lock()  # one GPU, one process -> serialize requests
        self.warm = False
        self.metrics = Metrics(model_id, self)
        self._seen_shapes: set[tuple[int, int]] = set()

    def warmup(self) -> None:
        """Pay the first-call costs once at boot, not on the first user request.

        Nothing is compiled here -- there is no torch.compile on this path any
        more -- but the first CUDA matmul still pays cuBLAS handle creation,
        kernel autotuning and allocator growth. MEASURED on the JAX sibling as
        18.77s cold against 4.35s warm for the same request; the mechanism
        differs, the size of the trap does not.
        """
        torch = self.torch
        t0 = time.monotonic()
        ids = self.tokenizer("warmup", return_tensors="pt").to(self.device)
        with torch.inference_mode():
            self.model.generate(**ids, max_new_tokens=4, do_sample=False)
            torch.cuda.synchronize()
        self.warm = True
        logger.info("Warmup done in %.1fs", time.monotonic() - t0)

    def encode_text(self, prompt: str) -> list[int]:
        """Tokenize a raw prompt, forcing BOS.

        Gemma's tokenizer does not prepend <bos> for a plain __call__ even with
        add_special_tokens=True, and the model is trained with it -- without BOS
        greedy decoding degenerates into a repetition loop. apply_chat_template
        inserts it already, so only this raw path needs the fixup.
        """
        ids = self.tokenizer(prompt)["input_ids"]
        bos = self.tokenizer.bos_token_id
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos, *ids]
        return ids

    def encode_chat(self, messages: list[dict]) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )["input_ids"]
        return ids[0].tolist()

    def _sample(self, logits, temperature: float, top_p: float):
        torch = self.torch
        if not temperature or temperature <= 0.0:
            return logits.argmax(-1, keepdim=True)
        probs = torch.softmax(logits.float() / max(temperature, 1e-5), dim=-1)
        if 0.0 < top_p < 1.0:
            srt, idx = torch.sort(probs, dim=-1, descending=True)
            keep = (srt.cumsum(-1) - srt) < top_p
            srt = srt * keep
            srt = srt / srt.sum(-1, keepdim=True).clamp_min(1e-9)
            return idx.gather(-1, torch.multinomial(srt, num_samples=1))
        return torch.multinomial(probs, num_samples=1)

    def generate(self, prompt_ids: list[int], max_tokens: int, temperature: float, top_p: float):
        """Yield (token_id, text_piece) per step. Caller must hold self.lock.

        Prefill and decode are timed SEPARATELY and the split is what /metrics
        exposes. An end-to-end tok/s carries prefill and the HTTP round trip and
        is not comparable to the sibling's decode gauge; this is.
        """
        torch = self.torch
        n_prompt = len(prompt_ids)
        budget = min(max_tokens, self.seq - n_prompt - 1)
        if budget <= 0:
            raise ValueError(
                f"prompt is {n_prompt} tokens and the context bound is {self.seq}, "
                f"leaving no room to decode. Start the server with a larger --seq."
            )

        # Cold is tracked per (prompt bucket, max_tokens) rather than per process:
        # a new shape still pays autotune and allocator growth. Quoting a mean
        # that includes a cold request is how a 4x error gets into a report.
        shape = (1 << max(0, n_prompt - 1).bit_length(), budget)
        cold = shape not in self._seen_shapes
        self._seen_shapes.add(shape)

        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        out_ids: list[int] = []

        with torch.inference_mode():
            t0 = time.monotonic()
            out = self.model(input_ids=ids, use_cache=True)
            torch.cuda.synchronize()
            prefill_s = time.monotonic() - t0

            past = out.past_key_values
            nxt = self._sample(out.logits[:, -1, :], temperature, top_p)

            t0 = time.monotonic()
            decoded = 0
            for _ in range(budget):
                tid = int(nxt.item())  # one sync per token: needed to stream and stop
                if tid in self.stop_ids:
                    break
                out_ids.append(tid)
                decoded += 1
                yield tid, self.tokenizer.decode([tid], skip_special_tokens=True)

                out = self.model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = self._sample(out.logits[:, -1, :], temperature, top_p)
            torch.cuda.synchronize()
            decode_s = time.monotonic() - t0

        self.metrics.record(
            prompt_tokens=n_prompt, completion_tokens=decoded,
            prefill_s=prefill_s, decode_s=decode_s, cold=cold,
            degenerate=_is_degenerate(out_ids),
        )


def _is_degenerate(ids: list[int]) -> bool:
    """True when the tail of the output is a short repeating cycle.

    This exists because a NON-EMPTY reply is not evidence of health -- the rule
    the engineering notes call out by name. The vLLM sibling once answered
    `': ok: ok: ok...'` and the JAX sibling's KV-ring eviction returned a token
    loop, both with status="success". `verify_model_health` reads the counter
    this feeds either side of its own probe, so the verdict on the text is the
    server's rather than the caller's.

    Observational only: it changes neither the response nor the status code.
    """
    if len(ids) < 12:
        return False
    tail = ids[-12:]
    for period in (1, 2, 3, 4):
        unit = tail[:period]
        if len(tail) // period >= 4 and all(
            tail[i] == unit[i % period] for i in range(period * (len(tail) // period))
        ):
            return True
    return False


class Metrics:
    """Prometheus exposition, with the series names the control plane parses.

    THE NAMES ARE DELIBERATELY `tpu_jax_*` ON A PYTORCH RIG. They are wrong as
    description and right as an identifier: both of this family's benchmark
    reports compare on `tpu_jax_decode_tokens_per_second` BY NAME, and
    `server.py`'s `_parse_prom` / `get_metrics` / `verify_model_health` look up
    these exact strings. Renaming the prefix would break continuity with every
    prior measurement, which is the whole reason this rig exists. The `rig` label
    is what separates the runtimes; carry that, not a rename.

    Only `model` labels the numeric series, because _parse_prom pops `model` and
    folds every REMAINING label into the sample key -- so an extra label here
    turns `tpu_jax_decode_seconds_total` into a key get_metrics cannot find, and
    the cumulative decode line silently disappears.
    """

    def __init__(self, model_id: str, engine: KVCacheDecodeEngine) -> None:
        self.model_id = model_id
        self.engine = engine
        self.rig = os.getenv("RIG_NAME", "gpu-pytorch-g5g-2b")
        self.build_id = _read_build_id()
        self.requests = {"success": 0.0, "failed": 0.0}
        self.prompt_tokens = 0.0
        self.completion_tokens = 0.0
        self.prefill_seconds = 0.0
        self.decode_seconds = 0.0
        self.latency_seconds = 0.0
        self.cold_requests = 0.0
        self.degenerate = 0.0
        self.last_decode_tps = 0.0
        self.last_cold = False

    def record(self, *, prompt_tokens, completion_tokens, prefill_s, decode_s,
               cold, degenerate) -> None:
        self.requests["success"] += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.prefill_seconds += prefill_s
        self.decode_seconds += decode_s
        self.latency_seconds += prefill_s + decode_s
        if cold:
            self.cold_requests += 1
        if degenerate:
            self.degenerate += 1
        self.last_cold = cold
        self.last_decode_tps = completion_tokens / decode_s if decode_s > 0 else 0.0
        logger.info(
            "request done prompt_tokens=%d completion_tokens=%d prefill_s=%.3f "
            "decode_s=%.3f decode_tps=%.2f cold=%s degenerate=%s",
            prompt_tokens, completion_tokens, prefill_s, decode_s,
            self.last_decode_tps, cold, degenerate,
        )
        if degenerate:
            logger.warning("degenerate (repeating) output detected; see /metrics")

    def failed(self) -> None:
        self.requests["failed"] += 1

    def render(self) -> str:
        m, rig = self.model_id, self.rig
        e = self.engine
        lines = [
            "# HELP tpu_jax_precision_info Dtypes and quantisation resolved on device",
            "# TYPE tpu_jax_precision_info gauge",
            f'tpu_jax_precision_info{{model="{m}",rig="{rig}",build_id="{self.build_id}",'
            f'compute_dtype="{str(e.dtype).replace("torch.", "")}",quant_mode="fp16",'
            f'kv_cache_dtype="{str(e.dtype).replace("torch.", "")}",kv_cache_requested="auto",'
            f'ple_bits="0",int8_lm_head="false",'
            f'pre_ampere="{str(e.pre_ampere).lower()}"}} 1',
            "# HELP tpu_jax_requests_total Requests by terminal status",
            "# TYPE tpu_jax_requests_total counter",
        ]
        for status, count in self.requests.items():
            lines.append(f'tpu_jax_requests_total{{model="{m}",status="{status}"}} {count}')
        for name, value, kind, help_text in (
            ("tpu_jax_prompt_tokens_total", self.prompt_tokens, "counter", "Prompt tokens"),
            ("tpu_jax_completion_tokens_total", self.completion_tokens, "counter",
             "Completion tokens"),
            ("tpu_jax_prefill_seconds_total", self.prefill_seconds, "counter",
             "Seconds spent in prefill"),
            ("tpu_jax_decode_seconds_total", self.decode_seconds, "counter",
             "Seconds spent in decode alone"),
            ("tpu_jax_latency_seconds_sum", self.latency_seconds, "counter",
             "Prefill plus decode seconds"),
            ("tpu_jax_cold_requests_total", self.cold_requests, "counter",
             "Requests that were the first of their shape"),
            ("tpu_jax_degenerate_responses_total", self.degenerate, "counter",
             "Replies whose tail was a repeating token cycle"),
            ("tpu_jax_decode_tokens_per_second", self.last_decode_tps, "gauge",
             "Decode throughput of the LAST request, excluding prefill and HTTP"),
            ("tpu_jax_weight_bytes", float(e.weight_bytes), "gauge",
             "Resident parameter bytes"),
            ("tpu_jax_model_load_seconds", e.load_seconds, "gauge", "Checkpoint load time"),
        ):
            lines += [
                f"# HELP {name} {help_text}",
                f"# TYPE {name} {kind}",
                f'{name}{{model="{m}"}} {value}',
            ]
        return "\n".join(lines) + "\n"


def _read_build_id() -> str:
    """The payload digest the deploy stamped next to this file.

    `deploy_torch_server` ships a PAYLOAD_SHA in the tarball so a stale deploy is
    one line of output rather than a hand comparison of md5s -- which is what it
    cost on the sibling on 2026-08-24. Absent when running from a checkout.
    """
    for cand in ("PAYLOAD_SHA", os.path.join(os.path.dirname(__file__), "PAYLOAD_SHA")):
        try:
            with open(cand) as fh:
                return fh.read().strip()
        except OSError:
            continue
    return "unknown"


def create_app(engine: KVCacheDecodeEngine):
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse, StreamingResponse

    app = FastAPI(title="Gemma 4 on T4G via PyTorch", version="1.0.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": engine.model_id,
            "warm": engine.warm,
            "seq": engine.seq,
            "build_id": engine.metrics.build_id,
            "rig": engine.metrics.rig,
            "compute_dtype": str(engine.dtype).replace("torch.", ""),
            "compute_capability": engine.capability,
            "pre_ampere": engine.pre_ampere,
            "weight_bytes": engine.weight_bytes,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return engine.metrics.render()

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
        # The SSE body runs AFTER the handler returns, so a failure in here is
        # outside the handler's try entirely -- on the sibling that made a broken
        # stream look like a short answer: not counted, not logged. Hence the
        # explicit guard.
        try:
            async with engine.lock:
                for _tid, piece in engine.generate(
                    prompt_ids, req.max_tokens, req.temperature, req.top_p
                ):
                    chunk = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [
                            {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0)
        except Exception:
            engine.metrics.failed()
            logger.exception("streaming generation failed for %s", cid)
            raise
        tail = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(tail)}\n\n"
        yield "data: [DONE]\n\n"

    async def _complete(prompt_ids, req, cid):
        async with engine.lock:
            try:
                return await asyncio.to_thread(_run, prompt_ids, req)
            except ValueError as e:
                engine.metrics.failed()
                raise HTTPException(status_code=400, detail=str(e)) from e
            except Exception as e:
                # A per-request CUDA OOM used to vanish here: HTTPException(str(exc))
                # discards the traceback and nothing was logged, so the failure the
                # rig most needs to diagnose was invisible to get_torch_logs.
                engine.metrics.failed()
                logger.exception("generation failed for %s", cid)
                raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        prompt_ids = engine.encode_chat([m.model_dump() for m in req.messages])
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        if req.stream:
            return StreamingResponse(
                _stream_chat(prompt_ids, req, cid, created), media_type="text/event-stream"
            )
        text, n_out = await _complete(prompt_ids, req, cid)
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
                "decode_tokens_per_second": round(engine.metrics.last_decode_tps, 3),
                "cold_shape": engine.metrics.last_cold,
            },
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        prompt_ids = engine.encode_text(req.prompt)
        cid = f"cmpl-{uuid.uuid4().hex[:24]}"
        text, n_out = await _complete(prompt_ids, req, cid)
        return {
            "id": cid,
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": n_out,
                "total_tokens": len(prompt_ids) + n_out,
                "decode_tokens_per_second": round(engine.metrics.last_decode_tps, 3),
                "cold_shape": engine.metrics.last_cold,
            },
        }

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--seq", type=int, default=int(os.getenv("MAX_MODEL_LEN", "4096")),
        help="context bound: prompt + output. No longer a padded buffer, so raising it is free.",
    )
    args = parser.parse_args()

    load_hf_token()
    import uvicorn

    engine = KVCacheDecodeEngine(args.model, seq=args.seq)
    engine.warmup()
    app = create_app(engine)
    logger.info("Serving %s on %s:%d (SEQ=%d)", args.model, args.host, args.port, args.seq)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
