#!/usr/bin/env python3
"""OpenAI-compatible HTTP/SSE server over static-shape Neuron graphs.

Routes: /v1/models, /v1/chat/completions, /v1/completions, /health, /metrics.

    python3 torch_openai_server.py --model google/gemma-4-E2B-it --port 8000
    python3 torch_openai_server.py --batch 8 --neff-dir /opt/gemma4/neff

The engine lives in `torch_generate.py` and is imported, not copied. The KV
blend, the per-stream masks and the host-side embedding gather are the parts of
this rig that fail *silently* when they are wrong -- a second copy that drifted
would produce plausible text rather than an error.

WHY NOT transformers' generate(): a traced Neuron graph has exactly one shape.
generate() grows the sequence dimension per token and feeds CPU scalars in, so
every step would be a new graph and a fresh multi-minute compile. This server
keeps one [B, MAX] KV buffer for the life of the process and calls a single
decode graph per step.

WHY ONE ENGINE THREAD: the device is not shareable. One thread owns both graphs
and steps every slot in lockstep, so the weights are read once per step no matter
how many streams are live -- concurrency is close to free (29.1 ms/step at B=8
against 21.1 at B=1, measured on the QAT graphs,
`benchmarks/runs/2026-07-31-inf2-serving-perf/`). FastAPI handlers submit a
Stream and wait on its queue; they never touch the device.

WHAT IS NOT MEASURED HERE. The dense reference build has no served result of its
own on this rig yet. The numbers above came off the QAT graphs, and the 2026-08-02
parity run drove the engine in-process rather than through HTTP. Do not quote
them as this server's throughput.

The default `--batch 1` is the conservative one: a graph traced at one B cannot
run at another, so raising it is a retrace, not a flag.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import threading
import time
import uuid
from typing import Any

from torch_generate import (
    DEFAULT_BATCH,
    DEFAULT_MAX_TOTAL,
    DEFAULT_MODEL,
    DEFAULT_PROMPT_BUCKET,
    NeuronGemmaEngine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("torch_openai_server")

# Secrets Manager id holding the Hugging Face token. cloud-init fetches it by id
# at boot into a root-only systemd EnvironmentFile; it is deliberately never put
# in user data, which anything on the instance can read.
HF_SECRET_ID = os.getenv("HF_SECRET_ID", "vllm/hf-token")

GEN_TIMEOUT = float(os.getenv("GEN_TIMEOUT", "120"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "16"))
GRACE_SECONDS = float(os.getenv("GRACE_SECONDS", "25"))

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
    top_k: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=128, ge=1)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    prompt: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=128, ge=1)
    stream: bool = False


def load_hf_token() -> None:
    """Confirm the token is already in the environment. Nothing is fetched here.

    cloud-init reads Secrets Manager at boot and writes a root-only systemd
    EnvironmentFile, so by the time this process starts the token is either in
    the environment or it is not coming. Fetching at request time would put an
    AWS call on the serving path for no benefit.
    """
    if os.environ.get("HF_TOKEN"):
        logger.info("HF token present in the environment.")
        return
    logger.info(
        "No HF_TOKEN in the environment; relying on the local cache or public weights. "
        "If a gated download 401s, check the systemd EnvironmentFile and secret %s.",
        HF_SECRET_ID,
    )


def sample(torch, logits, temperature: float, top_k: int, top_p: float) -> int:
    """Pick one token id from a [vocab] logits row.

    Sampling runs on the HOST, deliberately. It is a few hundred microseconds of
    work on a vector the device has already produced, and putting it in the graph
    would make temperature and top_p trace-time constants -- a recompile per
    distinct request parameter, which is exactly the availability hazard the JAX
    sibling hit with `max_new_tokens` as a static argname (quirk 6).
    """
    if not temperature or temperature <= 0.0:
        return int(torch.argmax(logits))
    scaled = logits.float() / float(temperature)
    if top_k and top_k > 0:
        k = min(int(top_k), scaled.numel())
        kth = torch.topk(scaled, k).values[-1]
        scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)
    probs = torch.softmax(scaled, dim=-1)
    if top_p and 0.0 < top_p < 1.0:
        sorted_p, order = torch.sort(probs, descending=True)
        keep = (torch.cumsum(sorted_p, dim=-1) - sorted_p) <= top_p
        sorted_p = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
        probs = torch.zeros_like(probs).scatter(0, order, sorted_p)
    total = probs.sum()
    if total <= 0:
        return int(torch.argmax(scaled))
    return int(torch.multinomial(probs / total, 1))


class Stream:
    """One request's lifecycle across the engine."""

    __slots__ = (
        "cur",
        "deadline",
        "done",
        "finish",
        "ids",
        "last",
        "max_new",
        "n0",
        "out_q",
        "prompt_ids",
        "stop_ids",
        "t_submit",
        "temperature",
        "top_k",
        "top_p",
    )

    def __init__(self, prompt_ids, max_new, temperature, top_k, top_p, stop_ids,
                 timeout_s, ceiling):
        self.prompt_ids = prompt_ids
        self.n0 = len(prompt_ids)
        # Cap so `cur` can never reach the park row an idle slot writes to.
        self.max_new = max(1, min(max_new, ceiling - self.n0))
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.stop_ids = stop_ids
        self.deadline = time.time() + (timeout_s or GEN_TIMEOUT)
        self.ids: list[int] = []
        self.cur = self.n0
        self.last: int | None = None
        self.out_q: queue.Queue = queue.Queue()   # (token_id, None) per token; (None, reason) ends
        self.done = threading.Event()
        self.finish = "length"
        self.t_submit = time.time()


class Metrics:
    """Prometheus text exposition, kept deliberately small.

    `requests_total` is the counter to check before concluding a request hung:
    curl gives up at its 120 s default while the server completes the work and
    records it.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.start = time.time()
        self.values = {
            "requests_total": 0, "errors_total": 0, "timeouts_total": 0,
            "prompt_tokens_total": 0, "completion_tokens_total": 0,
            "generation_seconds_total": 0.0, "last_tok_per_s": 0.0,
            "engine_steps_total": 0, "prefills_total": 0,
        }

    def bump(self, **kwargs) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if key == "last_tok_per_s":
                    self.values[key] = value
                else:
                    self.values[key] += value

    def render(self, engine: ServingEngine | None, ready: bool, draining: bool) -> str:
        with self.lock:
            v = dict(self.values)
        secs = v["generation_seconds_total"]
        avg = (v["completion_tokens_total"] / secs) if secs > 0 else 0.0
        # Zeroed rather than omitted while the engine is still compiling: a
        # series that disappears and reappears reads as a scrape failure.
        slots = engine.model.batch if engine else 0
        active = engine.active_count() if engine else 0
        depth = engine.pending.qsize() if engine else 0
        max_total = engine.model.max_total if engine else 0
        max_prompt = engine.model.prompt_bucket if engine else 0
        rows = [
            ("gemma_up", "gauge", 1 if ready else 0),
            ("gemma_uptime_seconds", "gauge", f"{time.time() - self.start:.1f}"),
            ("gemma_requests_total", "counter", v["requests_total"]),
            ("gemma_errors_total", "counter", v["errors_total"]),
            ("gemma_timeouts_total", "counter", v["timeouts_total"]),
            ("gemma_prompt_tokens_total", "counter", v["prompt_tokens_total"]),
            ("gemma_completion_tokens_total", "counter", v["completion_tokens_total"]),
            ("gemma_generation_seconds_total", "counter", f"{secs:.3f}"),
            ("gemma_tokens_per_second_last", "gauge", f"{v['last_tok_per_s']:.2f}"),
            ("gemma_tokens_per_second_avg", "gauge", f"{avg:.2f}"),
            ("gemma_engine_steps_total", "counter", v["engine_steps_total"]),
            ("gemma_prefills_total", "counter", v["prefills_total"]),
            ("gemma_batch_slots", "gauge", slots),
            ("gemma_active_slots", "gauge", active),
            ("gemma_queue_depth", "gauge", depth),
            ("gemma_max_total_tokens", "gauge", max_total),
            ("gemma_max_prompt_tokens", "gauge", max_prompt),
            ("gemma_draining", "gauge", 1 if draining else 0),
        ]
        return "\n".join(f"# TYPE {n} {t}\n{n} {val}" for n, t, val in rows) + "\n"


class ServingEngine:
    """Single thread owning the device: admits joiners by prefill, steps in lockstep.

    Slot lifecycle is the whole design. A joiner is admitted only at a step
    boundary, and admission runs the prefill graph for the FULL batch -- empty
    slots included, since the graph has one shape. Their rows are discarded; only
    a joiner's KV is copied into the persistent buffers.
    """

    def __init__(self, model: NeuronGemmaEngine, metrics: Metrics) -> None:
        self.model = model
        self.metrics = metrics
        self.pending: queue.Queue = queue.Queue()
        self.slots: list[Stream | None] = [None] * model.batch
        self.key_bufs, self.val_bufs = model.zero_kv()
        self.wake = threading.Event()
        self.torch = None

    def active_count(self) -> int:
        return sum(1 for s in self.slots if s is not None)

    def submit(self, stream: Stream) -> None:
        if self.pending.qsize() >= MAX_QUEUE:
            raise queue.Full
        self.pending.put(stream)
        self.wake.set()

    def _admit(self) -> None:
        free = [b for b in range(self.model.batch) if self.slots[b] is None]
        joiners: dict[int, Stream] = {}
        while free and not self.pending.empty():
            try:
                joiners[free.pop(0)] = self.pending.get_nowait()
            except queue.Empty:
                break
        if not joiners:
            return

        prompts = []
        for b in range(self.model.batch):
            stream = joiners.get(b)
            # An unoccupied slot gets a single pad token, not an empty row: an
            # all-masked row makes softmax NaN, and NaN does not stay in its slot.
            prompts.append(stream.prompt_ids if stream else [self.model.tokenizer.bos_token_id or 0])
        pads, masks, _lengths = self.model.pad_prompts(prompts)
        logits, keys, values = self.model.prefill(pads, masks)

        bucket = self.model.prompt_bucket
        for b, stream in joiners.items():
            for j in range(len(self.model.nonshared)):
                self.key_bufs[j][b, :, :bucket, :] = keys[j][b, :, :bucket, :]
                self.val_bufs[j][b, :, :bucket, :] = values[j][b, :, :bucket, :]
            stream.last = sample(self.torch, logits[b, stream.n0 - 1],
                                 stream.temperature, stream.top_k, stream.top_p)
            self.slots[b] = stream
            self._emit_or_finish(b, stream)   # the first token may already be EOS
        self.metrics.bump(prefills_total=1)

    def _emit_or_finish(self, b: int, stream: Stream) -> None:
        token = stream.last
        if token in self.model.eos_ids or token in stream.stop_ids:
            self._finish(b, stream, "stop")
            return
        stream.ids.append(token)
        stream.out_q.put((token, None))
        if len(stream.ids) >= stream.max_new:
            self._finish(b, stream, "length")
        elif time.time() > stream.deadline:
            self._finish(b, stream, "timeout")

    def _finish(self, b: int, stream: Stream, reason: str) -> None:
        stream.finish = reason
        self.slots[b] = None
        stream.out_q.put((None, reason))
        stream.done.set()

    def _step(self) -> None:
        park = self.model.park
        tokens = [(s.last if s else 0) for s in self.slots]
        positions = [(s.cur if s else park) for s in self.slots]
        logits, self.key_bufs, self.val_bufs = self.model.decode_step(
            tokens, positions, self.key_bufs, self.val_bufs
        )
        for b, stream in enumerate(self.slots):
            if stream is None:
                continue
            stream.cur += 1
            stream.last = sample(self.torch, logits[b, 0],
                                 stream.temperature, stream.top_k, stream.top_p)
            self._emit_or_finish(b, stream)
        self.metrics.bump(engine_steps_total=1)

    def run(self) -> None:
        import torch

        self.torch = torch
        while True:
            if self.active_count() == 0 and self.pending.empty():
                self.wake.wait(timeout=1.0)
                self.wake.clear()
                continue
            if not self.pending.empty() and any(s is None for s in self.slots):
                self._admit()
            if self.active_count():
                self._step()


class Server:
    """Process-wide state: the engine, readiness, drain, metrics.

    Loading is deliberately NOT done in the constructor. A cold trace of both
    graphs takes minutes -- tens of minutes on a first compile -- and doing it
    before uvicorn binds means every health probe in that window gets connection
    refused, which reads as a crashed service rather than a warming one. The port
    comes up first and /health answers 503 "loading" until the model is real.
    """

    def __init__(self, args: Any) -> None:
        self.args = args
        self.metrics = Metrics()
        self.model: NeuronGemmaEngine | None = None
        self.engine: ServingEngine | None = None
        self.ready = threading.Event()
        self.draining = threading.Event()
        self.failed: str | None = None

    def start(self) -> None:
        threading.Thread(target=self._watch_spot, daemon=True).start()
        threading.Thread(target=self._bring_up, daemon=True).start()

    def _bring_up(self) -> None:
        args = self.args
        try:
            model = NeuronGemmaEngine(
                model_id=args.model, batch=args.batch, max_total=args.max_total,
                prompt_bucket=args.prompt_bucket, device=args.device,
                neff_dir=args.neff_dir, local_dir=args.local_dir,
            )
            model.load()
            model.compile()
        except Exception as exc:
            # Recorded rather than raised: the thread dying silently would leave
            # /health saying "loading" forever, which is the least debuggable
            # possible outcome of a failed compile.
            self.failed = repr(exc)
            logger.exception("bring-up failed")
            return
        self.model = model
        self.engine = ServingEngine(model, self.metrics)
        threading.Thread(target=self.engine.run, daemon=True).start()
        self._warm()

    def _warm(self) -> None:
        """Run one real request before declaring ready.

        Graph load and the first allocation are paid here rather than by the
        first caller. It also proves the round trip produces TEXT: a warmup that
        only counted tokens would pass against the zero-gather fault, which
        returns a clean stop with an empty string.
        """
        t0 = time.monotonic()
        prompt_ids = self.model.encode_chat([{"role": "user", "content": "Hi"}])
        # A generous deadline on purpose: Stream.deadline is wall clock from
        # submit, and the first call through a freshly loaded graph is the
        # slowest one this process will ever make. A tight cap here would report
        # the warmup as a timeout and hide whatever it actually produced.
        stream = Stream(prompt_ids, 4, 0.0, 0, 1.0, set(), 600.0, self.model.park)
        self.engine.submit(stream)
        if not stream.done.wait(timeout=600):
            logger.error("warmup did not complete in 600s; serving anyway, expect slow first calls")
        elif not stream.ids:
            logger.error(
                "WARMUP PRODUCED ZERO TOKENS. This is the signature Neuron fault, not a "
                "cold start: an oversized device gather returns zeros, which decode to the "
                "pad id, which is an EOS. See docs/neuron-jax-quirks.md quirk 1."
            )
        self.ready.set()
        logger.info("ready in %.1fs", time.monotonic() - t0)

    def _watch_spot(self) -> None:
        """Flip to draining on a spot interruption notice.

        Launches default to spot on this rig, so a two-minute notice is the
        normal end of an instance's life, not an exception. Draining turns
        /health 503 so a load balancer stops sending work while in-flight
        streams finish.
        """
        import urllib.request as request

        base = "http://169.254.169.254"
        while not self.draining.is_set():
            token = None
            try:
                token = request.urlopen(
                    request.Request(base + "/latest/api/token", method="PUT",
                                    headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}),
                    timeout=2,
                ).read().decode()
            except Exception:
                pass
            try:
                req = request.Request(base + "/latest/meta-data/spot/instance-action")
                if token:
                    req.add_header("X-aws-ec2-metadata-token", token)
                if request.urlopen(req, timeout=2).status == 200:
                    self.draining.set()
                    logger.warning("spot interruption notice -- draining")
                    return
            except Exception:
                pass
            time.sleep(5)

    # -- request plumbing -----------------------------------------------------

    def submit(self, prompt_ids: list[int], req) -> Stream:
        bucket = self.model.prompt_bucket
        if len(prompt_ids) > bucket:
            raise ValueError(
                f"prompt is {len(prompt_ids)} tokens; the graph is traced at "
                f"prompt_bucket={bucket}. Retrace with a larger --prompt-bucket."
            )
        stream = Stream(prompt_ids, req.max_tokens, req.temperature, req.top_k, req.top_p,
                        set(), None, self.model.park)
        self.engine.submit(stream)
        return stream

    def record(self, stream: Stream, n_out: int) -> None:
        elapsed = time.time() - stream.t_submit
        self.metrics.bump(
            requests_total=1, prompt_tokens_total=stream.n0, completion_tokens_total=n_out,
            generation_seconds_total=elapsed,
            timeouts_total=1 if stream.finish == "timeout" else 0,
            last_tok_per_s=(n_out / elapsed if elapsed > 0 else 0.0),
        )
        logger.info("pt=%d ct=%d %.1f tok/s %.2fs finish=%s active=%d",
                    stream.n0, n_out, n_out / elapsed if elapsed > 0 else 0.0,
                    elapsed, stream.finish, self.engine.active_count())

    def collect(self, stream: Stream) -> tuple[str, int]:
        stream.done.wait(timeout=GEN_TIMEOUT + 30)
        text = self.model.tokenizer.decode(stream.ids, skip_special_tokens=True)
        self.record(stream, len(stream.ids))
        return text, len(stream.ids)


def create_app(server: Server):
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse, StreamingResponse

    app = FastAPI(title="Inferentia2 OpenAI-compatible server", version="1.0.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    # The model is resolved per request, never captured here: create_app runs
    # before the checkpoint is loaded so that the port binds first.

    @app.get("/health")
    async def health():
        if server.failed:
            raise HTTPException(status_code=500, detail=f"bring-up failed: {server.failed}")
        if server.draining.is_set():
            raise HTTPException(status_code=503, detail="draining")
        if not server.ready.is_set():
            raise HTTPException(status_code=503, detail="loading")
        model = server.model
        return {
            "status": "ok", "model": model.model_id, "device": "Inferentia2",
            "batch_slots": model.batch, "active": server.engine.active_count(),
            "max_total_tokens": model.max_total, "max_prompt_tokens": model.prompt_bucket,
        }

    @app.get("/v1/models")
    async def models():
        model_id = server.model.model_id if server.model else server.args.model
        return {"object": "list",
                "data": [{"id": model_id, "object": "model", "owned_by": "local-inferentia2"}]}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        # Served while loading too. gemma_up=0 with a rising uptime is how a long
        # compile is told apart from a hung process from the outside.
        return server.metrics.render(server.engine, server.ready.is_set(),
                                     server.draining.is_set())

    def _guard() -> None:
        if server.failed:
            raise HTTPException(status_code=500, detail=f"bring-up failed: {server.failed}")
        if not server.ready.is_set():
            raise HTTPException(status_code=503, detail="server loading; retry shortly")
        if server.draining.is_set():
            raise HTTPException(status_code=503, detail="server draining; retry elsewhere")

    def _submit(prompt_ids, req) -> Stream:
        try:
            return server.submit(prompt_ids, req)
        except queue.Full as e:
            server.metrics.bump(errors_total=1)
            raise HTTPException(status_code=429,
                                detail=f"server busy (queue > {MAX_QUEUE})") from e
        except ValueError as e:
            server.metrics.bump(errors_total=1)
            raise HTTPException(status_code=400, detail=str(e)) from e

    async def _sse(stream: Stream, req, cid: str, created: int, kind: str):
        def envelope(delta=None, finish=None):
            if kind == "text":
                return {"id": cid, "object": "text_completion", "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "text": delta or "", "finish_reason": finish}]}
            content = {} if delta is None else {"content": delta}
            return {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": content, "finish_reason": finish}]}

        if kind == "chat":
            head = envelope()
            head["choices"][0]["delta"] = {"role": "assistant"}
            yield f"data: {json.dumps(head)}\n\n"

        ids: list[int] = []
        previous = ""
        finish = "length"
        while True:
            # get() blocks a worker thread, so it runs off the event loop. The
            # tokenizer is re-run over the whole prefix each step because Gemma's
            # SentencePiece decoding is not a concatenation of per-token strings.
            try:
                token, reason = await asyncio.to_thread(
                    stream.out_q.get, True, GEN_TIMEOUT + 30
                )
            except queue.Empty:
                # The engine stopped feeding this stream. Close the SSE cleanly:
                # an exception here would truncate the body with no terminator
                # and the client would report a network error rather than a
                # timeout it can act on.
                finish = "timeout"
                server.metrics.bump(timeouts_total=1)
                break
            if token is None:
                finish = reason
                break
            ids.append(token)
            text = server.model.tokenizer.decode(ids, skip_special_tokens=True)
            delta, previous = text[len(previous):], text
            if delta:
                yield f"data: {json.dumps(envelope(delta=delta))}\n\n"
        yield f"data: {json.dumps(envelope(finish=finish))}\n\n"
        yield "data: [DONE]\n\n"
        server.record(stream, len(ids))

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        _guard()
        prompt_ids = server.model.encode_chat([m.model_dump() for m in req.messages])
        stream = _submit(prompt_ids, req)
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        if req.stream:
            return StreamingResponse(_sse(stream, req, cid, created, "chat"),
                                     media_type="text/event-stream")
        text, n_out = await asyncio.to_thread(server.collect, stream)
        return {
            "id": cid, "object": "chat.completion", "created": created, "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": stream.finish}],
            "usage": {"prompt_tokens": stream.n0, "completion_tokens": n_out,
                      "total_tokens": stream.n0 + n_out},
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        _guard()
        prompt_ids = server.model.encode_text(req.prompt)
        stream = _submit(prompt_ids, req)
        cid, created = f"cmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        if req.stream:
            return StreamingResponse(_sse(stream, req, cid, created, "text"),
                                     media_type="text/event-stream")
        text, n_out = await asyncio.to_thread(server.collect, stream)
        return {
            "id": cid, "object": "text_completion", "created": created, "model": req.model,
            "choices": [{"index": 0, "text": text, "logprobs": None,
                         "finish_reason": stream.finish}],
            "usage": {"prompt_tokens": stream.n0, "completion_tokens": n_out,
                      "total_tokens": stream.n0 + n_out},
        }

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-dir", default=None,
                        help="load from this path instead of the Hub")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help="lockstep slots; changing this requires a retrace")
    parser.add_argument("--max-total", type=int, default=DEFAULT_MAX_TOTAL,
                        help="KV rows the graph is traced for (prompt + generated)")
    parser.add_argument("--prompt-bucket", type=int, default=DEFAULT_PROMPT_BUCKET,
                        help="padded prefill length the graph is traced for")
    parser.add_argument("--neff-dir", default=os.getenv("NEFF_DIR"),
                        help="reuse/save traced graphs here; a cold trace takes minutes")
    parser.add_argument("--device", default="neuron", choices=("neuron", "cpu"),
                        help="'cpu' runs the same graphs eagerly, for testing without a device")
    args = parser.parse_args()

    load_hf_token()
    import uvicorn

    server = Server(args)
    app = create_app(server)
    server.start()

    # No custom SIGTERM handler: uvicorn installs its own during run() and would
    # replace one set here. Draining on shutdown is uvicorn's
    # timeout_graceful_shutdown; draining on a spot notice is _watch_spot.
    logger.info("serving %s on %s:%d (batch=%d, max_total=%d, prompt_bucket=%d)",
                args.model, args.host, args.port, args.batch, args.max_total,
                args.prompt_bucket)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                timeout_graceful_shutdown=int(GRACE_SECONDS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
