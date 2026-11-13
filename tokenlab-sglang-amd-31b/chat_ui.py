#!/usr/bin/env python3
"""Browser chat UI for Gemma 4 31B on AMD Token Lab. Standard library only.

Forked from ~/tpu-inference/tpu-vllm-v5e1-2b/chat_ui.py (the v5e-1 vLLM page). Serves one
page on 127.0.0.1 and forwards /v1/* to Token Lab with the API key attached, so the key
never reaches the browser. 429 and 502/503/504 are retried before the reply starts
(tokenlab.open_api); the number of retries comes back in the X-Tokenlab-Retries header.
Two more routes feed the page: /ui/status (does /v1/models list the model) and /ui/info
(the model card from /v1/models plus this rig's tpu.env). A hosted API has no /metrics
or /version, so the TPU page's engine and load panels are replaced by this session's
own counts: requests, retries, tokens and points (usage.cost).

    python3 chat_ui.py [--port 8766]
"""
import argparse
import http.server
import json
import threading

import tokenlab

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemma 4 31B on Token Lab</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --text: #1d2330; --muted: #667085; --line: #e3e6eb;
    --accent: #2f6fed; --accent-text: #ffffff; --user: #e8effd; --bot: #ffffff; --code: #f1f3f6;
    --ok: #1f9d55; --bad: #d64545; --warn: #c58a00;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #12151b; --panel: #1a1e26; --text: #e6e8ec; --muted: #98a2b3; --line: #2a303b;
      --accent: #5b8cff; --accent-text: #0b0d12; --user: #1f2a44; --bot: #1a1e26; --code: #232833;
      --ok: #3ccf7a; --bad: #ff6b6b; --warn: #f0b429;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  .app { display: grid; grid-template-columns: 1fr 300px; height: 100vh; }
  @media (max-width: 860px) { .app { grid-template-columns: 1fr; } aside { display: none; } }
  main { display: flex; flex-direction: column; min-width: 0; }
  header { display: flex; align-items: center; gap: 12px; padding: 12px 20px;
           border-bottom: 1px solid var(--line); background: var(--panel); }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header .sub { color: var(--muted); font-size: 13px; }
  .badge { margin-left: auto; display: inline-flex; align-items: center; gap: 6px; font-size: 13px;
           padding: 4px 10px; border: 1px solid var(--line); border-radius: 999px; white-space: nowrap; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--warn); }
  .dot.ok { background: var(--ok); } .dot.bad { background: var(--bad); }
  #log { flex: 1; overflow-y: auto; padding: 24px 20px; display: flex; flex-direction: column; gap: 16px; }
  .empty { margin: auto; text-align: center; color: var(--muted); max-width: 520px; }
  .empty h2 { color: var(--text); font-size: 20px; margin: 0 0 8px; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 16px; }
  .chip { border: 1px solid var(--line); background: var(--panel); color: var(--text); border-radius: 999px;
          padding: 6px 12px; cursor: pointer; font: inherit; font-size: 13px; }
  .chip:hover { border-color: var(--accent); }
  .msg { max-width: 820px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 4px; }
  .bubble { padding: 12px 14px; border-radius: 12px; border: 1px solid var(--line); overflow-wrap: anywhere; }
  .msg.user .bubble { background: var(--user); border-color: transparent; align-self: flex-end; max-width: 85%; }
  .msg.bot .bubble { background: var(--bot); }
  .who { font-size: 12px; color: var(--muted); }
  .msg.user .who { align-self: flex-end; }
  .stats { font-size: 12px; color: var(--muted); }
  details.think { color: var(--muted); font-size: 13px; border-left: 3px solid var(--line);
                  padding-left: 10px; margin-bottom: 8px; white-space: pre-wrap; }
  details.think summary { cursor: pointer; }
  .bubble p { margin: 0 0 8px; } .bubble p:last-child { margin-bottom: 0; }
  .bubble pre { background: var(--code); padding: 10px; border-radius: 8px; overflow-x: auto; margin: 8px 0; }
  .bubble code { background: var(--code); padding: 1px 4px; border-radius: 4px; font-size: 13px; }
  .bubble pre code { padding: 0; background: none; }
  .bubble ul, .bubble ol { margin: 4px 0 8px 20px; padding: 0; }
  .cursor .answer::after { content: "▍"; animation: blink 1s steps(1) infinite; color: var(--accent); }
  @keyframes blink { 50% { opacity: 0; } }
  form { border-top: 1px solid var(--line); background: var(--panel); padding: 12px 20px; }
  .row { max-width: 820px; margin: 0 auto; display: flex; gap: 8px; align-items: flex-end; }
  textarea { flex: 1; resize: none; min-height: 44px; max-height: 200px; padding: 11px 12px; border-radius: 10px;
             border: 1px solid var(--line); background: var(--bg); color: var(--text); font: inherit; }
  textarea:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  button.primary { background: var(--accent); color: var(--accent-text); border: 0; border-radius: 10px;
                   padding: 0 18px; height: 44px; font: inherit; font-weight: 600; cursor: pointer; }
  button.ghost { background: none; border: 1px solid var(--line); color: var(--text); border-radius: 10px;
                 height: 44px; padding: 0 12px; font: inherit; cursor: pointer; }
  .hint { max-width: 820px; margin: 6px auto 0; font-size: 12px; color: var(--muted); }
  aside { border-left: 1px solid var(--line); background: var(--panel); padding: 16px; overflow-y: auto; font-size: 13px; }
  aside h3 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 18px 0 8px; }
  aside h3:first-child { margin-top: 0; }
  dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 10px; margin: 0; }
  dt { color: var(--muted); } dd { margin: 0; overflow-wrap: anywhere; }
  label.ctl { display: block; margin: 8px 0 4px; color: var(--muted); }
  input[type=range] { width: 100%; }
  aside textarea { width: 100%; min-height: 70px; font-size: 13px; }
  .note { color: var(--muted); font-size: 12px; margin-top: 8px; }
  label.attach { display: inline-flex; align-items: center; justify-content: center; cursor: pointer; }
  label.attach.on { border-color: var(--accent); }
  .bubble img { max-width: 260px; max-height: 200px; border-radius: 8px; display: block; margin-bottom: 6px; }
</style>
</head>
<body>
<div class="app">
  <main>
    <header>
      <div>
        <h1 id="title">Gemma 4 31B · AMD Token Lab</h1>
        <div class="sub" id="sub">connecting…</div>
      </div>
      <span class="badge"><span class="dot" id="dot"></span><span id="badgeText">checking…</span></span>
    </header>
    <div id="log">
      <div class="empty" id="empty">
        <h2>Ask Gemma 4 31B on AMD GPUs</h2>
        <div>Replies stream from AMD Token Lab through this machine's local proxy, which holds the API key.</div>
        <div class="chips" id="chips"></div>
      </div>
    </div>
    <form id="form">
      <div class="row">
        <label class="ghost attach" title="Attach an image (Gemma 4 31B reads images)">🖼<input type="file" id="image" accept="image/*" hidden></label>
        <textarea id="input" rows="1" placeholder="Message Gemma…  (Enter to send, Shift+Enter for a new line)"></textarea>
        <button type="button" class="ghost" id="clear" title="Start a new conversation">New</button>
        <button type="submit" class="primary" id="send">Send</button>
      </div>
      <div class="hint" id="hint">The conversation is sent with each message · Stop closes the connection</div>
    </form>
  </main>
  <aside>
    <h3>Settings</h3>
    <label class="ctl" for="maxTokens">Max tokens: <span id="maxTokensVal">512</span></label>
    <input type="range" id="maxTokens" min="16" max="2048" step="16" value="512">
    <label class="ctl" for="temp">Temperature: <span id="tempVal">server default</span></label>
    <input type="range" id="temp" min="-0.1" max="1.5" step="0.1" value="-0.1">
    <label class="ctl" for="topp">Top-p: <span id="toppVal">server default</span></label>
    <input type="range" id="topp" min="0" max="1" step="0.05" value="1">
    <label class="ctl"><input type="checkbox" id="think"> Thinking (reasoning shown above the answer)</label>
    <label class="ctl" for="system">System prompt (optional)</label>
    <textarea id="system" placeholder="e.g. Answer in one short paragraph."></textarea>
    <h3>Last reply</h3>
    <dl id="last"><dt>—</dt><dd>nothing yet</dd></dl>
    <h3>On Token Lab</h3>
    <dl id="engine"><dt>—</dt><dd>loading…</dd></dl>
    <h3>This session</h3>
    <dl id="load"><dt>—</dt><dd>nothing yet</dd></dl>
    <div class="note">Times are measured in this browser, so they include the network and any queueing at Token Lab. Points: usage.cost as Token Lab reports it.</div>
  </aside>
</div>
<script>
const $ = (id) => document.getElementById(id);
const log = $("log"), input = $("input"), send = $("send");
let history = [], model = null, controller = null, image = null;

const examples = [
  "Why is the sky blue? One paragraph.",
  "Write a haiku about GPUs.",
  "Explain a key-value cache in a transformer, simply.",
  "Give me three fun facts about octopuses.",
];
for (const ex of examples) {
  const b = document.createElement("button");
  b.className = "chip"; b.type = "button"; b.textContent = ex;
  b.onclick = () => { input.value = ex; submit(); };
  $("chips").appendChild(b);
}

function esc(s) { return s.replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
function inline(s) {
  return esc(s).replace(/`([^`]+)`/g, "<code>$1</code>")
               .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
               .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
}
function render(md) {
  const out = []; const lines = md.split("\n"); let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (l.startsWith("```")) {
      const code = []; i++;
      while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]);
      i++; out.push("<pre><code>" + esc(code.join("\n")) + "</code></pre>"); continue;
    }
    if (/^\s*([-*•]|\d+\.)\s+/.test(l)) {
      const ordered = /^\s*\d+\./.test(l); const items = [];
      while (i < lines.length && /^\s*([-*•]|\d+\.)\s+/.test(lines[i]))
        items.push("<li>" + inline(lines[i++].replace(/^\s*([-*•]|\d+\.)\s+/, "")) + "</li>");
      out.push((ordered ? "<ol>" : "<ul>") + items.join("") + (ordered ? "</ol>" : "</ul>")); continue;
    }
    if (/^#{1,6}\s/.test(l)) { out.push("<p><strong>" + inline(l.replace(/^#+\s/, "")) + "</strong></p>"); i++; continue; }
    if (l.trim() === "") { i++; continue; }
    const para = [];
    while (i < lines.length && lines[i].trim() !== "" && !lines[i].startsWith("```") && !/^\s*([-*•]|\d+\.)\s+/.test(lines[i]))
      para.push(inline(lines[i++]));
    out.push("<p>" + para.join("<br>") + "</p>");
  }
  return out.join("");
}

function addMsg(role, text) {
  $("empty")?.remove();
  const m = document.createElement("div"); m.className = "msg " + role;
  m.innerHTML = `<div class="who">${role === "user" ? "You" : "Gemma 4 31B · Token Lab"}</div><div class="bubble"></div><div class="stats"></div>`;
  const bubble = m.querySelector(".bubble");
  if (role === "user") bubble.innerHTML = (image ? `<img src="${image}" alt="attached image">` : "") + render(text);
  else bubble.innerHTML = `<details class="think" hidden><summary>Reasoning</summary><div></div></details><div class="answer"></div>`;
  log.appendChild(m); log.scrollTop = log.scrollHeight;
  return m;
}

function setDL(id, rows) {
  rows = rows.filter(([, v]) => v !== undefined && v !== null && v !== "");
  $(id).innerHTML = rows.length ? rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(String(v))}</dd>`).join("")
                                : "<dt>—</dt><dd>unavailable</dd>";
}

async function status() {
  try {
    const s = await (await fetch("/ui/status")).json();
    const ok = s.listed === true;
    $("dot").className = "dot " + (ok ? "ok" : "bad");
    $("badgeText").textContent = ok ? "model listed" : (s.error || "not answering");
    return ok;
  } catch { $("dot").className = "dot bad"; $("badgeText").textContent = "proxy not answering"; return false; }
}

async function info() {
  try {
    const e = await (await fetch("/ui/info")).json();
    const m = e.model || {}, r = e.rig || {}, l = e.session || {};
    model = m.id || r.MODEL_NAME || model;
    $("sub").textContent = [model, r.RUNTIME_NOTE, r.HARDWARE_NOTE].filter(Boolean).join(" · ");
    setDL("engine", [
      ["model", m.id], ["name", m.name],
      ["context", m.context_length && m.context_length + " tokens"],
      ["inputs", (m.architecture || {}).input_modalities?.join(", ")],
      ["service", m.description], ["hardware", r.HARDWARE_NOTE],
      ["params", (m.supported_parameters || []).join(", ")],
      ["endpoint", e.upstream],
    ]);
    setDL("load", [
      ["requests", l.requests], ["retried (429/5xx)", l.retries], ["failed", l.failed],
      ["prompt tokens", l.prompt_tokens], ["cached prompt tokens", l.cached_tokens],
      ["completion tokens", l.completion_tokens],
      ["points used", l.cost != null ? l.cost.toFixed(8) : null],
    ]);
  } catch { setDL("engine", []); }
}

async function submit() {
  const text = input.value.trim();
  if (!text || controller) return;
  if (!model) await info();
  input.value = ""; autosize();
  const content = image ? [{ type: "image_url", image_url: { url: image } }, { type: "text", text }] : text;
  history.push({ role: "user", content });
  addMsg("user", text);
  image = null; $("image").value = ""; $("image").parentElement.classList.remove("on");
  const bot = addMsg("bot", ""); const bubble = bot.querySelector(".bubble");
  const think = bubble.querySelector(".think"), thinkBody = think.querySelector("div"), answer = bubble.querySelector(".answer");
  bubble.classList.add("cursor");
  send.textContent = "Stop"; controller = new AbortController();
  const sys = $("system").value.trim();
  const messages = (sys ? [{ role: "system", content: sys }] : []).concat(history);
  const body = { model, messages, stream: true, max_tokens: +$("maxTokens").value, stream_options: { include_usage: true },
                 chat_template_kwargs: { enable_thinking: $("think").checked } };
  if (+$("temp").value >= 0) body.temperature = +$("temp").value;
  if (+$("topp").value < 1) body.top_p = +$("topp").value;
  const t0 = performance.now(); let ttft = null, reply = "", reasoning = "", usage = null, finish = null, retries = 0, sawDone = false;
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST", headers: { "content-type": "application/json" }, signal: controller.signal,
      body: JSON.stringify(body),
    });
    retries = +(r.headers.get("x-tokenlab-retries") || 0);
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`);
    const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
    while (true) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, k).trim(); buf = buf.slice(k + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim(); if (data === "[DONE]") { sawDone = true; continue; }
        const j = JSON.parse(data);
        if (j.error) throw new Error(j.error.message || JSON.stringify(j.error));
        if (j.usage) usage = j.usage;
        const c = (j.choices || [])[0] || {};
        const d = c.delta || {};
        const rc = d.reasoning_content || d.reasoning;
        if ((rc || d.content) && ttft === null) ttft = (performance.now() - t0) / 1000;
        if (rc) { reasoning += rc; think.hidden = false; thinkBody.textContent = reasoning; }
        if (d.content) { reply += d.content; answer.innerHTML = render(reply); }
        if (c.finish_reason) finish = c.finish_reason;
        log.scrollTop = log.scrollHeight;
      }
    }
  } catch (err) {
    if (err.name === "AbortError") { finish = "stopped"; sawDone = true; }
    else answer.innerHTML = render(reply + `\n\n**Error:** ${err.message}`);
  }
  bubble.classList.remove("cursor");
  // Token Lab's FAQ: a stream that ends without [DONE] was cut off; send it again.
  if (!sawDone && !finish) answer.innerHTML = render(reply + "\n\n**Stream ended early** (no [DONE]); send the message again.");
  const total = (performance.now() - t0) / 1000;
  const n = usage ? usage.completion_tokens : null;
  const rate = n && n > 1 && ttft !== null && total > ttft ? (n - 1) / (total - ttft) : null;
  bot.querySelector(".stats").textContent =
    [ttft !== null ? `first token ${ttft.toFixed(2)} s` : null, rate ? `${rate.toFixed(1)} tok/s` : null,
     n !== null ? `${n} tokens` : null, `${total.toFixed(1)} s`, retries ? `${retries} retries` : null, finish].filter(Boolean).join(" · ");
  setDL("last", [["first token", ttft !== null ? ttft.toFixed(3) + " s" : "—"], ["decode", rate ? rate.toFixed(1) + " tok/s" : "—"],
                 ["tokens", n ?? "—"], ["prompt tokens", usage ? usage.prompt_tokens : "—"],
                 ["reasoning", reasoning ? reasoning.length + " chars" : null],
                 ["cached prompt tokens", usage?.prompt_tokens_details?.cached_tokens],
                 ["points", usage?.cost != null ? usage.cost.toFixed(8) : null],
                 ["retries before reply", retries || null],
                 ["total", total.toFixed(2) + " s"], ["finish", finish ?? "—"]]);
  if (reply) history.push({ role: "assistant", content: reply }); else history.pop();
  controller = null; send.textContent = "Send"; input.focus();
  info();
}

function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 200) + "px"; }
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("form").requestSubmit(); } });
$("form").addEventListener("submit", (e) => { e.preventDefault(); if (controller) controller.abort(); else submit(); });
$("clear").onclick = () => { if (controller) controller.abort(); location.reload(); };
$("maxTokens").oninput = (e) => { $("maxTokensVal").textContent = e.target.value; };
$("topp").oninput = (e) => { $("toppVal").textContent = +e.target.value < 1 ? (+e.target.value).toFixed(2) : "server default"; };
$("image").onchange = (e) => {
  const f = e.target.files[0]; if (!f) return;
  const rd = new FileReader();
  rd.onload = () => { image = rd.result; $("image").parentElement.classList.add("on"); $("hint").textContent = `Image attached: ${f.name} · sent with the next message`; };
  rd.readAsDataURL(f);
};
$("temp").oninput =(e) => { $("tempVal").textContent = +e.target.value < 0 ? "server default" : (+e.target.value).toFixed(1); };

status().then((ok) => { if (ok) info(); });
setInterval(async () => { if (await status() && !controller) info(); }, 15000);
input.focus();
</script>
</body>
</html>"""


class Session:
    """Counts for the side panel, summed from the usage block of each finished reply."""

    lock = threading.Lock()
    counts = {"requests": 0, "retries": 0, "failed": 0, "prompt_tokens": 0, "cached_tokens": 0,
              "completion_tokens": 0, "cost": 0.0}

    @classmethod
    def add(cls, **kw):
        with cls.lock:
            for k, v in kw.items():
                cls.counts[k] += v or 0

    @classmethod
    def snapshot(cls):
        with cls.lock:
            return dict(cls.counts)


def usage_from_sse(tail: bytes):
    """The last `usage` object in the tail of an event stream (include_usage puts it in the final chunk)."""
    for line in reversed(tail.decode("utf-8", "replace").splitlines()):
        line = line.strip()
        if line.startswith("data:") and '"usage"' in line:
            try:
                u = json.loads(line[5:].strip()).get("usage")
            except ValueError:
                continue
            if u:
                return u
    return None


class Handler(http.server.BaseHTTPRequestHandler):
    cfg = {}

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/ui/status":
            self._json(self._status())
        elif self.path == "/ui/info":
            self._json(self._info())
        elif self.path.startswith("/v1/"):
            self._forward(None)
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.path.startswith("/v1/"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        self._forward(self.rfile.read(length))

    def _status(self):
        try:
            m = tokenlab.model_entry(self.cfg)
        except (OSError, RuntimeError, ValueError) as e:
            return {"listed": False, "error": str(e)[:120]}
        return {"listed": m is not None, "error": None if m else "model not listed"}

    def _info(self):
        rig_keys = ("MODEL_NAME", "RUNTIME_NOTE", "HARDWARE_NOTE", "PLATFORM_NOTE", "CONTEXT_LENGTH")
        info = {"upstream": self.cfg["TOKENLAB_BASE_URL"], "rig": {k: self.cfg[k] for k in rig_keys if k in self.cfg},
                "session": Session.snapshot()}
        try:
            info["model"] = tokenlab.model_entry(self.cfg)
        except (OSError, RuntimeError, ValueError):
            pass
        return info

    def _send(self, code, data, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _forward(self, body):
        retries = []
        path = self.path[len("/v1"):]
        try:
            resp = tokenlab.open_api(self.cfg, path, body, on_retry=lambda *a: retries.append(a))
        except (OSError, RuntimeError) as e:
            Session.add(failed=1, requests=1 if body else 0)
            self._json({"error": {"message": f"Token Lab unreachable: {e}"}}, 502)
            return
        # HTTP/1.0 response with no Content-Length: the body ends when the connection
        # closes, which lets the event stream pass through chunk by chunk. When the
        # browser stops (Stop button), the write fails and closing `resp` drops the
        # upstream connection.
        tail = b""
        try:
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
            self.send_header("X-Tokenlab-Retries", str(len(retries)))
            self.end_headers()
            while chunk := resp.read1(65536):
                tail = (tail + chunk)[-16384:]
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            resp.close()
        if body is not None:
            u = usage_from_sse(tail) if b"data:" in tail else _json_usage(tail)
            u = u or {}
            Session.add(
                requests=1, retries=len(retries), failed=0 if resp.status == 200 else 1,
                prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"), cost=u.get("cost"),
            )

    def log_message(self, fmt, *args):
        pass


def _json_usage(raw: bytes):
    try:
        return json.loads(raw).get("usage")
    except (ValueError, AttributeError):
        return None


def main():
    cfg = tokenlab.load_config()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=int(cfg.get("UI_PORT", 8766)))
    a = p.parse_args()
    if not cfg.get("TOKENLAB_API_KEY"):
        raise SystemExit("TOKENLAB_API_KEY is not set: put it in this rig's .env (mode 0600) or the environment")
    Handler.cfg = cfg
    server = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Chat UI on http://localhost:{a.port} -> {cfg['TOKENLAB_BASE_URL']} ({cfg['MODEL_NAME']})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
