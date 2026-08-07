"""Gemma-4-E2B continuous-batching inference server on Inferentia2 (Option B, batch-8 graphs).

Same SLIM-HOST loading + HTTP surface as optb_server_qat.py, but the generation core is a
lockstep continuous-batching engine: one engine thread owns the device and multiplexes up to
B concurrent streams over batch-B prefill/decode graphs with PER-STREAM positions
(qat_e2b_cb_trace.py neffs — position_ids [B,1], onehot [B,1,MAX,1], masks [B,1,1,MAX]).

- Requests join at the next step boundary (one prefill call admits all waiting joiners).
- Each stream has its own position/sampling/EOS; finished streams free their slot immediately.
- Weights are read once per step regardless of active streams, so concurrency is ~free:
  measured 29.1 ms/step at B=8 vs 21.1 at B=1 (274.9 tok/s aggregate ceiling).

Env: B (default 8), KV_MAX, KV_BUCKET, KV_PRE_OUT, KV_DEC_OUT, HOST_DTYPE, SELFTEST,
MAX_QUEUE (extra requests beyond B slots allowed to wait), GEN_TIMEOUT, GRACE_SECONDS, PORT.
NOTE: no authentication (per request). Bound to 0.0.0.0."""
import os
os.environ["NEURON_RT_VISIBLE_CORES"] = "0,1"
import sys, json, time, threading, glob, queue, resource, torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
MP = "/workspace/real-gemma4-E2B-it"
B = int(os.environ.get("B", "8"))
MAX = int(os.environ.get("KV_MAX", "128"))
BUCKET = int(os.environ.get("KV_BUCKET", "32"))
PRE_NEFF = os.environ.get("KV_PRE_OUT", "/workspace/cb8_pre.pt")
DEC_NEFF = os.environ.get("KV_DEC_OUT", "/workspace/cb8_dec.pt")
HOST_DTYPE = torch.bfloat16 if os.environ.get("HOST_DTYPE", "bf16") == "bf16" else torch.float32
SELFTEST = os.environ.get("SELFTEST", "0") == "1"
NEG = torch.finfo(torch.float32).min
NEG_INF = float("-inf")
PORT = int(os.environ.get("PORT", "8080"))
MODEL_NAME = "gemma-4-E2B-it"
PARK = MAX - 1              # parked slots write KV to this row; generation is capped below it

def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 2)

from transformers import AutoTokenizer, AutoConfig, Gemma4ForConditionalGeneration
from safetensors import safe_open
import torch_neuronx
print(f"loading tokenizer + SLIM host embeddings (dtype={HOST_DTYPE}, B={B})...", flush=True)
tok = AutoTokenizer.from_pretrained(MP)
cfg_full = AutoConfig.from_pretrained(MP)
with torch.device("meta"):
    m = Gemma4ForConditionalGeneration(cfg_full)
m.eval()
shards = sorted(glob.glob(os.path.join(MP, "*.safetensors")))
if not shards:
    raise RuntimeError(f"no *.safetensors in {MP}")
sd, nbytes = {}, 0
for st in shards:
    with safe_open(st, framework="pt") as f:
        for k in f.keys():
            if ".language_model." in k and ".layers." not in k:
                t = f.get_tensor(k).to(HOST_DTYPE)
                sd[k] = t; nbytes += t.numel() * t.element_size()
missing, unexpected = m.load_state_dict(sd, strict=False, assign=True)
lang = m.model.language_model; cfg = lang.config; SW = cfg.sliding_window
for _name, _buf in list(lang.named_buffers()):
    if "layers." in _name or not _buf.is_meta:
        continue
    _mod = lang; *_parents, _leaf = _name.split(".")
    for _p in _parents: _mod = getattr(_mod, _p)
    if _leaf == "embed_scale":
        _mod.register_buffer(_leaf, torch.tensor(float(cfg.hidden_size) ** 0.5), persistent=False)
    else:
        print(f"WARN: unmaterialized host buffer {_name} -> zeros", flush=True)
        _mod.register_buffer(_leaf, torch.zeros(tuple(_buf.shape)), persistent=False)
print(f"slim-loaded {len(sd)} tensors ({round(nbytes/1e9,2)} GB); peak RSS {rss_gb()} GB", flush=True)
NONSHARED, LINFO = [], {}
for i, lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
    a = lyr.self_attn
    if not a.is_kv_shared_layer:
        LINFO[i] = (a.k_proj.out_features // a.head_dim, a.head_dim); NONSHARED.append(i)
ec = m.generation_config.eos_token_id
EOS = set(ec) if isinstance(ec, (list, tuple)) else {ec}

PRE = DEC = None
_READY = threading.Event()
_counter = [0]

# ---- spot drain + queue accounting (queue = requests waiting for a slot) ----
import urllib.request as _urlreq
import signal
_DRAINING = threading.Event()
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "16"))            # waiting beyond the B active slots
GEN_TIMEOUT = float(os.environ.get("GEN_TIMEOUT", "120"))
GRACE_SECONDS = float(os.environ.get("GRACE_SECONDS", "25"))

def _watch_spot():
    base = "http://169.254.169.254"
    while not _DRAINING.is_set():
        tokn = None
        try:
            tokn = _urlreq.urlopen(_urlreq.Request(base + "/latest/api/token", method="PUT",
                    headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}), timeout=2).read().decode()
        except Exception:
            pass
        try:
            req = _urlreq.Request(base + "/latest/meta-data/spot/instance-action")
            if tokn:
                req.add_header("X-aws-ec2-metadata-token", tokn)
            if _urlreq.urlopen(req, timeout=2).status == 200:
                _DRAINING.set()
                print("SPOT INTERRUPTION NOTICE — draining", flush=True)
                return
        except Exception:
            pass
        time.sleep(5)

# ---- metrics ----
_START = time.time()
_MLOCK = threading.Lock()
_METRICS = {"requests_total": 0, "errors_total": 0, "timeouts_total": 0, "prompt_tokens_total": 0,
            "completion_tokens_total": 0, "generation_seconds_total": 0.0, "last_tok_per_s": 0.0,
            "engine_steps_total": 0, "prefills_total": 0}

def _bump(reqs=0, ptoks=0, ctoks=0, secs=0.0, err=0, to=0, tps=None, steps=0, prefills=0):
    with _MLOCK:
        _METRICS["requests_total"] += reqs
        _METRICS["prompt_tokens_total"] += ptoks
        _METRICS["completion_tokens_total"] += ctoks
        _METRICS["generation_seconds_total"] += secs
        _METRICS["errors_total"] += err
        _METRICS["timeouts_total"] += to
        _METRICS["engine_steps_total"] += steps
        _METRICS["prefills_total"] += prefills
        if tps is not None:
            _METRICS["last_tok_per_s"] = tps

def _cur_rss_bytes():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0

def _render_metrics():
    with _MLOCK:
        mm = dict(_METRICS)
    up = time.time() - _START
    avg = (mm["completion_tokens_total"] / mm["generation_seconds_total"]) if mm["generation_seconds_total"] > 0 else 0.0
    def g(name, help_, typ, val):
        return f"# HELP {name} {help_}\n# TYPE {name} {typ}\n{name} {val}"
    return "\n".join([
        g("gemma_up", "1 if serving (ready)", "gauge", 1 if _READY.is_set() else 0),
        g("gemma_uptime_seconds", "seconds since process start", "gauge", f"{up:.1f}"),
        g("gemma_requests_total", "total generation requests", "counter", mm["requests_total"]),
        g("gemma_errors_total", "total request errors", "counter", mm["errors_total"]),
        g("gemma_timeouts_total", "generations cut short by the wall-clock cap", "counter", mm["timeouts_total"]),
        g("gemma_prompt_tokens_total", "total prompt tokens", "counter", mm["prompt_tokens_total"]),
        g("gemma_completion_tokens_total", "total generated tokens", "counter", mm["completion_tokens_total"]),
        g("gemma_generation_seconds_total", "total request wall seconds", "counter", f"{mm['generation_seconds_total']:.3f}"),
        g("gemma_tokens_per_second_last", "tok/s of the most recent request", "gauge", f"{mm['last_tok_per_s']:.2f}"),
        g("gemma_tokens_per_second_avg", "avg per-request tok/s", "gauge", f"{avg:.2f}"),
        g("gemma_engine_steps_total", "lockstep decode steps executed", "counter", mm["engine_steps_total"]),
        g("gemma_prefills_total", "prefill graph calls", "counter", mm["prefills_total"]),
        g("gemma_batch_slots", "configured lockstep slots", "gauge", B),
        g("gemma_active_slots", "streams currently decoding", "gauge", ENGINE.active_count()),
        g("gemma_queue_depth", "requests waiting for a slot", "gauge", ENGINE.pending.qsize()),
        g("gemma_max_total_tokens", "configured KV_MAX", "gauge", MAX),
        g("gemma_max_prompt_tokens", "configured KV_BUCKET", "gauge", BUCKET),
        g("gemma_draining", "1 if draining", "gauge", 1 if _DRAINING.is_set() else 0),
        g("process_resident_memory_bytes", "resident set size", "gauge", _cur_rss_bytes()),
    ]) + "\n"

def embed_ids(batch_ids):
    ids = torch.tensor(batch_ids)
    with torch.no_grad():
        ie = lang.embed_tokens(ids); ple = lang.get_per_layer_inputs(ids, ie)
    return ie, ple

def host_pos(pos_list):
    pos = torch.tensor(pos_list)
    ar = torch.arange(MAX)
    oh = (ar[None, :] == pos[:, None]).view(B, 1, MAX, 1).to(HOST_DTYPE)
    valid = ar[None, :] <= pos[:, None]
    f = torch.where(valid, 0.0, NEG).view(B, 1, 1, MAX)
    s = torch.where(valid & (ar[None, :] > (pos[:, None] - SW)), 0.0, NEG).view(B, 1, 1, MAX)
    return pos.view(B, 1).long(), oh, f, s

def pick(logits, temperature, top_k, top_p):
    if temperature is None or temperature <= 0.0:
        return int(torch.argmax(logits))
    logits = logits.float() / float(temperature)
    if top_k and top_k > 0:
        k = min(int(top_k), logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, NEG_INF), logits)
    probs = torch.softmax(logits, dim=-1)
    if top_p and 0.0 < top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        keep = cum - sp <= top_p
        sp = torch.where(keep, sp, torch.zeros_like(sp))
        probs = torch.zeros_like(probs).scatter(0, si, sp)
    total = probs.sum()
    if total <= 0:
        return int(torch.argmax(logits))
    probs = probs / total
    return int(torch.multinomial(probs, 1))


class Stream:
    """One request's lifecycle across the engine."""
    __slots__ = ("prompt_ids", "n0", "max_new", "temperature", "top_k", "top_p", "stop_ids",
                 "deadline", "ids", "cur", "last", "out_q", "done", "finish", "t_submit", "t_first")

    def __init__(self, prompt_ids, max_new, temperature, top_k, top_p, stop_ids, timeout_s):
        self.prompt_ids = prompt_ids
        self.n0 = len(prompt_ids)
        # cap so cur never reaches PARK row
        self.max_new = max(1, min(max_new, MAX - 1 - self.n0 - 1))
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.stop_ids = stop_ids
        self.deadline = time.time() + (timeout_s or GEN_TIMEOUT)
        self.ids = []
        self.cur = self.n0
        self.last = None
        self.out_q = queue.Queue()      # per-token for streaming; (None, finish) terminates
        self.done = threading.Event()
        self.finish = "length"
        self.t_submit = time.time()
        self.t_first = None


class Engine:
    """Single thread owning the device: admits joiners via batched prefill, steps all
    active slots in lockstep, releases slots on EOS/length/timeout."""

    def __init__(self):
        self.pending = queue.Queue()
        self.slots = [None] * B
        self.kb = [torch.zeros(B, LINFO[i][0], MAX, LINFO[i][1], dtype=HOST_DTYPE) for i in NONSHARED]
        self.vb = [torch.zeros(B, LINFO[i][0], MAX, LINFO[i][1], dtype=HOST_DTYPE) for i in NONSHARED]
        self.wake = threading.Event()

    def active_count(self):
        return sum(1 for s in self.slots if s is not None)

    def submit(self, stream):
        if self.pending.qsize() >= MAX_QUEUE:
            raise queue.Full
        self.pending.put(stream)
        self.wake.set()

    def _admit(self):
        free = [b for b in range(B) if self.slots[b] is None]
        joiners = []
        while free and not self.pending.empty():
            try:
                st = self.pending.get_nowait()
            except queue.Empty:
                break
            joiners.append((free.pop(0), st))
        if not joiners:
            return
        pads, ams = [], []
        for b in range(B):
            st = dict(joiners).get(b)
            if st is not None:
                pads.append(st.prompt_ids + [0] * (BUCKET - st.n0))
                ams.append([1] * st.n0 + [0] * (BUCKET - st.n0))
            else:
                pads.append([0] * BUCKET)
                ams.append([1] + [0] * (BUCKET - 1))   # 1 valid token: keeps softmax finite
        ie, ple = embed_ids(pads)
        am = torch.tensor(ams)
        with torch.no_grad():
            lg, ks, vs = PRE(ie, am, ple)
        for b, st in joiners:
            for j in range(len(NONSHARED)):
                self.kb[j][b, :, :BUCKET, :] = ks[j][b, :, :BUCKET, :]
                self.vb[j][b, :, :BUCKET, :] = vs[j][b, :, :BUCKET, :]
            st.last = pick(lg[b, st.n0 - 1], st.temperature, st.top_k, st.top_p)
            st.t_first = time.time()
            self.slots[b] = st
            self._emit_or_finish(b, st)     # first token may already be EOS
        _bump(prefills=1)

    def _emit_or_finish(self, b, st):
        tok_id = st.last
        if tok_id in EOS or tok_id in st.stop_ids:
            self._finish(b, st, "stop")
            return
        st.ids.append(tok_id)
        st.out_q.put((tok_id, None))
        if len(st.ids) >= st.max_new:
            self._finish(b, st, "length")
        elif time.time() > st.deadline:
            self._finish(b, st, "timeout")

    def _finish(self, b, st, reason):
        st.finish = reason
        self.slots[b] = None
        st.out_q.put((None, reason))
        st.done.set()

    def _step(self):
        toks = [(self.slots[b].last if self.slots[b] else 0) for b in range(B)]
        poss = [(self.slots[b].cur if self.slots[b] else PARK) for b in range(B)]
        ie1, ple1 = embed_ids([[t] for t in toks])
        pid, oh, f, s = host_pos(poss)
        with torch.no_grad():
            lg1, self.kb, self.vb = DEC(ie1, ple1, pid, oh, f, s, self.kb, self.vb)
        for b in range(B):
            st = self.slots[b]
            if st is None:
                continue
            st.cur += 1
            st.last = pick(lg1[b, 0], st.temperature, st.top_k, st.top_p)
            self._emit_or_finish(b, st)
        _bump(steps=1)

    def run(self):
        while True:
            if self.active_count() == 0 and self.pending.empty():
                self.wake.wait(timeout=1.0)
                self.wake.clear()
                continue
            if not self.pending.empty() and any(sl is None for sl in self.slots):
                self._admit()
            if self.active_count():
                self._step()


ENGINE = Engine()

def _prep_chat(messages, stop):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    prompt_ids = enc["input_ids"][0].tolist()
    stop_ids = set()
    if isinstance(stop, str): stop = [stop]
    for s in (stop or []):
        try: stop_ids.update(tok.encode(s, add_special_tokens=False))
        except Exception: pass
    return prompt_ids, stop_ids

def run_chat(messages, max_new, temperature, top_k, top_p, stop, timeout_s=None):
    prompt_ids, stop_ids = _prep_chat(messages, stop)
    if len(prompt_ids) > BUCKET:
        raise ValueError(f"prompt {len(prompt_ids)} tokens > BUCKET {BUCKET}")
    st = Stream(prompt_ids, max_new, temperature, top_k, top_p, stop_ids, timeout_s)
    ENGINE.submit(st)
    st.done.wait(timeout=(timeout_s or GEN_TIMEOUT) + 30)
    text = tok.decode(st.ids, skip_special_tokens=True)
    dt = time.time() - st.t_submit; ct = len(st.ids)
    _bump(reqs=1, ptoks=st.n0, ctoks=ct, secs=dt, to=(1 if st.finish == "timeout" else 0),
          tps=(ct / dt if dt > 0 else 0.0))
    print(f"[req] pt={st.n0} ct={ct} {round(ct/dt,1) if dt>0 else 0}tok/s "
          f"{round(dt,2)}s finish={st.finish} active={ENGINE.active_count()}", flush=True)
    return text, st.n0, ct, st.finish

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def _send_text(self, code, text):
        b = text.encode()
        self.send_response(code); self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")
    def do_GET(self):
        p = self.path.rstrip("/")
        if p == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local-inferentia2"}]})
        elif p in ("/health", "/ping"):
            if _DRAINING.is_set():
                self._send(503, {"status": "draining"})
            elif not _READY.is_set():
                self._send(503, {"status": "loading"})
            else:
                self._send(200, {"status": "ok", "active": ENGINE.active_count(), "slots": B})
        elif p == "/metrics":
            self._send_text(200, _render_metrics())
        else:
            self._send(200, {"status": "ok", "model": f"{MODEL_NAME} (Option B batch-{B} continuous batching)",
                             "device": "Inferentia2", "max_total_tokens": MAX, "max_prompt_tokens": BUCKET,
                             "batch_slots": B,
                             "routes": ["/generate", "/v1/chat/completions", "/v1/completions", "/v1/models", "/health", "/metrics"]})
    def do_POST(self):
        path = self.path.rstrip("/")
        if path not in ("/v1/chat/completions", "/v1/completions", "/generate"):
            return self._send(404, {"error": {"message": f"unknown route {path}"}})
        if not _READY.is_set():
            return self._send(503, {"error": {"message": "server loading (warming up); retry shortly"}})
        if _DRAINING.is_set():
            return self._send(503, {"error": {"message": "server draining (spot interruption); retry"}})
        try:
            body = self._body()
            if path == "/v1/chat/completions":
                msgs = body.get("messages")
                if not msgs: return self._send(400, {"error": {"message": "missing 'messages'"}})
                otype = "chat"
            elif path == "/v1/completions":
                prompt = body.get("prompt", "")
                if isinstance(prompt, list): prompt = prompt[0] if prompt else ""
                if not prompt: return self._send(400, {"error": {"message": "missing 'prompt'"}})
                msgs = [{"role": "user", "content": prompt}]; otype = "text"
            else:
                prompt = body.get("prompt", "")
                if not prompt: return self._send(400, {"error": "missing 'prompt'"})
                msgs = [{"role": "user", "content": prompt}]; otype = "generate"
            gd = otype == "generate"
            max_new = int(body.get("max_tokens", body.get("max_completion_tokens", body.get("max_new_tokens", 110 if gd else 256))))
            temp = float(body.get("temperature", 0.0 if gd else 0.7))
            top_k = int(body.get("top_k", 0))
            top_p = float(body.get("top_p", 1.0 if gd else 0.95))
            max_new = max(1, min(max_new, MAX - 1)); temp = max(0.0, min(temp, 5.0))
            top_p = max(0.0, min(top_p, 1.0)); top_k = max(0, top_k)
            stop = body.get("stop"); model = body.get("model", MODEL_NAME)
            timeout_s = float(body["timeout"]) if body.get("timeout") else None
            if bool(body.get("stream")) and otype in ("chat", "text"):
                self._stream(msgs, max_new, temp, top_k, top_p, stop, otype, model, timeout_s)
            else:
                text, pt, ct, finish = run_chat(msgs, max_new, temp, top_k, top_p, stop, timeout_s)
                self._completion(otype, text, pt, ct, finish, model, body.get("prompt", ""))
        except queue.Full:
            _bump(err=1)
            try: self._send(429, {"error": {"message": f"server busy (queue > {MAX_QUEUE}); retry later"}})
            except Exception: pass
        except ValueError as e:
            _bump(err=1)
            try: self._send(400, {"error": {"message": str(e)}})
            except Exception: pass
        except Exception as e:
            _bump(err=1)
            try: self._send(500, {"error": {"message": repr(e)}})
            except Exception: pass

    def _completion(self, otype, text, pt, ct, finish, model, prompt):
        _counter[0] += 1; now = int(time.time())
        if otype == "chat":
            self._send(200, {"id": f"chatcmpl-{now}-{_counter[0]}", "object": "chat.completion",
                "created": now, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
                "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}})
        elif otype == "text":
            self._send(200, {"id": f"cmpl-{now}-{_counter[0]}", "object": "text_completion",
                "created": now, "model": model,
                "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}],
                "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}})
        else:
            self._send(200, {"prompt": prompt, "response": text, "prompt_tokens": pt,
                             "gen_tokens": ct, "finish_reason": finish})

    def _stream(self, msgs, max_new, temp, top_k, top_p, stop, otype, model, timeout_s=None):
        prompt_ids, stop_ids = _prep_chat(msgs, stop)
        if len(prompt_ids) > BUCKET:
            return self._send(400, {"error": {"message": f"prompt {len(prompt_ids)} tokens > BUCKET {BUCKET}"}})
        st = Stream(prompt_ids, max_new, temp, top_k, top_p, stop_ids, timeout_s)
        try:
            ENGINE.submit(st)
        except queue.Full:
            _bump(err=1)
            return self._send(429, {"error": {"message": f"server busy (queue > {MAX_QUEUE}); retry later"}})
        _counter[0] += 1; now = int(time.time()); cid = f"chatcmpl-{now}-{_counter[0]}"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        def chunk(delta=None, finish=None):
            if otype == "text":
                return {"id": cid, "object": "text_completion", "created": now, "model": model,
                        "choices": [{"index": 0, "text": delta or "", "finish_reason": finish}]}
            d = {} if delta is None else {"content": delta}
            return {"id": cid, "object": "chat.completion.chunk", "created": now, "model": model,
                    "choices": [{"index": 0, "delta": d, "finish_reason": finish}]}
        def sse(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n"); self.wfile.flush()
        ids = []; prev = ""; finish = "length"
        try:
            while True:
                tok_id, fin = st.out_q.get(timeout=(timeout_s or GEN_TIMEOUT) + 30)
                if tok_id is None:
                    finish = fin
                    break
                ids.append(tok_id)
                text = tok.decode(ids, skip_special_tokens=True)
                delta = text[len(prev):]; prev = text
                if delta: sse(chunk(delta=delta))
            sse(chunk(finish=finish))
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            dt = time.time() - st.t_submit; ct = len(ids)
            _bump(reqs=1, ptoks=st.n0, ctoks=ct, secs=dt, to=(1 if finish == "timeout" else 0),
                  tps=(ct / dt if dt > 0 else 0.0))
            print(f"[req] stream pt={st.n0} ct={ct} {round(ct/dt,1) if dt>0 else 0}tok/s "
                  f"{round(dt,2)}s finish={finish}", flush=True)
        except (BrokenPipeError, ConnectionResetError):
            _bump(reqs=1, ptoks=st.n0, ctoks=len(ids), secs=time.time() - st.t_submit)
    def log_message(self, *a): pass

threading.Thread(target=_watch_spot, daemon=True).start()
_HTTPD = ThreadingHTTPServer(("0.0.0.0", PORT), H)

def _graceful_term(signum, frame):
    print(f"signal {signum} — draining (up to {GRACE_SECONDS}s), then shutting down", flush=True)
    _DRAINING.set()
    def _drain_then_stop():
        end = time.time() + GRACE_SECONDS
        while ENGINE.active_count() > 0 and time.time() < end:
            time.sleep(0.2)
        _HTTPD.shutdown()
    threading.Thread(target=_drain_then_stop, daemon=True).start()
signal.signal(signal.SIGTERM, _graceful_term)

def _load_and_warm():
    global PRE, DEC
    print("loading neffs onto NeuronCores...", flush=True)
    t0 = time.time()
    PRE = torch.jit.load(PRE_NEFF)
    DEC = torch.jit.load(DEC_NEFF)
    threading.Thread(target=ENGINE.run, daemon=True).start()
    try:
        st = Stream(tok.apply_chat_template([{"role": "user", "content": "Hi"}],
                    add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"][0].tolist(),
                    3, 0.0, 0, 1.0, set(), None)
        ENGINE.submit(st)
        st.done.wait(timeout=300)
    except Exception as e:
        print("warmup:", e, flush=True)
    _READY.set()
    print(f"READY in {round(time.time()-t0,1)}s — serving on :{PORT} "
          f"(batch-{B} continuous batching, peak RSS {rss_gb()} GB)", flush=True)
    if SELFTEST:
        results = [None] * 4
        def one(i):
            txt, pt, ct, fin = run_chat([{"role": "user", "content":
                "What is the capital of France? Answer in one word."}], 8, 0.0, 0, 1.0, None)
            results[i] = txt
        ths = [threading.Thread(target=one, args=(i,)) for i in range(4)]
        [t.start() for t in ths]; [t.join(timeout=120) for t in ths]
        print(f"SELFTEST 4-concurrent capital-of-France -> {results!r} (expect 4x 'Paris') "
              f"peakRSS={rss_gb()}GB", flush=True)

threading.Thread(target=_load_and_warm, daemon=True).start()
print(f"HTTP listening on :{PORT} — batch-{B} engine; queue max={MAX_QUEUE}, gen timeout {GEN_TIMEOUT}s", flush=True)
try:
    _HTTPD.serve_forever()
except KeyboardInterrupt:
    pass
print("server stopped", flush=True)
