#!/usr/bin/env python3
"""Browser chat UI for Gemma 4 E2B as a Lambda GPU function (Lambda Managed Instances).

Forked from tokenlab-sglang-amd-31b/chat_ui.py, itself from ~/tpu-inference/tpu-vllm-v5e1-2b/chat_ui.py.
Serves one page on 127.0.0.1. The page posts OpenAI chat requests to /v1/chat/completions here; this
proxy sends each one as a synchronous Lambda Invoke of the published version with this machine's AWS
credentials, so nothing about AWS reaches the browser. The image's handler forwards the payload as-is
to vLLM and returns the whole completion: there is no stream, so the page shows the reply when it is
complete, with the round trip and end-to-end tok/s. /ui/status reports the published version's state
and scaling; /ui/info this rig's tpu.env and this session's counts.

    python3 chat_ui.py [--port 8768]
"""

import argparse
import http.server
import json
import sys
import threading
import time
import urllib.request

import boto3
from botocore.config import Config

import lmi

E = lmi.E
QUALIFIER = "$LATEST.PUBLISHED"
CLIENT = boto3.client(
    "lambda",
    region_name=E["AWS_REGION"],
    config=Config(max_pool_connections=64, read_timeout=int(E["TIMEOUT_S"]) + 30, retries={"max_attempts": 0}),
)

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemma 4 E2B on Lambda</title>
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
        <h1 id="title">Gemma 4 E2B · AWS Lambda GPU</h1>
        <div class="sub" id="sub">connecting…</div>
      </div>
      <span class="badge"><span class="dot" id="dot"></span><span id="badgeText">checking…</span></span>
    </header>
    <div id="log">
      <div class="empty" id="empty">
        <h2>Ask Gemma 4 E2B on a Lambda function</h2>
        <div>Each message is one Lambda invocation on an NVIDIA L4, sent by this machine's local proxy with its AWS credentials. Replies arrive whole.</div>
        <div class="chips" id="chips"></div>
      </div>
    </div>
    <form id="form">
      <div class="row">
        <textarea id="input" rows="1" placeholder="Message Gemma…  (Enter to send, Shift+Enter for a new line)"></textarea>
        <button type="button" class="ghost" id="clear" title="Start a new conversation">New</button>
        <button type="submit" class="primary" id="send">Send</button>
      </div>
      <div class="hint" id="hint">The conversation is sent with each message · Stop abandons the wait (the invocation still finishes on Lambda)</div>
    </form>
  </main>
  <aside>
    <h3>Settings</h3>
    <label class="ctl" for="maxTokens">Max tokens: <span id="maxTokensVal">512</span></label>
    <input type="range" id="maxTokens" min="16" max="4096" step="16" value="512">
    <label class="ctl" for="temp">Temperature: <span id="tempVal">server default</span></label>
    <input type="range" id="temp" min="-0.1" max="1.5" step="0.1" value="-0.1">
    <label class="ctl" for="topp">Top-p: <span id="toppVal">server default</span></label>
    <input type="range" id="topp" min="0" max="1" step="0.05" value="1">
    <label class="ctl"><input type="checkbox" id="think"> Thinking (reasoning shown above the answer)</label>
    <label class="ctl" for="system">System prompt (optional)</label>
    <textarea id="system" placeholder="e.g. Answer in one short paragraph."></textarea>
    <h3>Last reply</h3>
    <dl id="last"><dt>—</dt><dd>nothing yet</dd></dl>
    <h3>On Lambda</h3>
    <dl id="engine"><dt>—</dt><dd>loading…</dd></dl>
    <h3>This session</h3>
    <dl id="load"><dt>—</dt><dd>nothing yet</dd></dl>
    <div class="note">Times are measured in this browser, so they include the round trip to the region and the Lambda invoke path. tok/s is completion tokens over the whole round trip.</div>
  </aside>
</div>
<script>
const $ = (id) => document.getElementById(id);
const log = $("log"), input = $("input"), send = $("send");
let history = [], model = null, controller = null;

const examples = [
  "Why is the sky blue? One paragraph.",
  "Write a haiku about serverless GPUs.",
  "Explain a key-value cache in a transformer, simply.",
  "What is 17 * 23? Show your working.",
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
  m.innerHTML = `<div class="who">${role === "user" ? "You" : "Gemma 4 E2B · Lambda"}</div><div class="bubble"></div><div class="stats"></div>`;
  const bubble = m.querySelector(".bubble");
  if (role === "user") bubble.innerHTML = render(text);
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
    const ok = s.state === "Active" && s.max !== 0;
    $("dot").className = "dot " + (ok ? "ok" : (s.state ? "" : "bad"));
    $("badgeText").textContent = s.error ? s.error : (s.max === 0 ? "scaled to 0 (make up)" : `${s.state} · min ${s.min ?? "?"} env`);
    return !s.error;
  } catch { $("dot").className = "dot bad"; $("badgeText").textContent = "proxy not answering"; return false; }
}

async function info() {
  try {
    const e = await (await fetch("/ui/info")).json();
    const r = e.rig || {}, l = e.session || {}, f = e.function || {};
    model = r.MODEL_NAME || model;
    $("sub").textContent = [model, `vLLM ${r.VLLM_VERSION}`, `${r.INSTANCE_TYPE} · ${r.ACCELERATOR_TYPE}`, r.AWS_REGION].join(" · ");
    setDL("engine", [
      ["function", `${r.FUNCTION_NAME}:${f.version || "?"}`], ["state", f.state],
      ["instance", r.INSTANCE_TYPE], ["GPU", r.ACCELERATOR_TYPE], ["region", r.AWS_REGION],
      ["environments", f.min != null ? `min ${f.min}${f.max != null ? ", max " + f.max : ""}` : null],
      ["per environment", `${r.PER_ENV_MAX_CONCURRENCY} requests at once`],
      ["memory", `${r.MEMORY_MB} MB, GPU ≥ ${r.ACCELERATOR_MEMORY_GB} GB`],
      ["context", `${r.MAX_MODEL_LEN} tokens`], ["engine", `vLLM ${r.VLLM_VERSION}`],
      ["weights", r.QUANTIZATION],
    ]);
    setDL("load", [
      ["invocations", l.requests], ["failed", l.failed],
      ["prompt tokens", l.prompt_tokens], ["completion tokens", l.completion_tokens],
      ["Lambda time", l.seconds ? l.seconds.toFixed(1) + " s" : null],
    ]);
  } catch { setDL("engine", []); }
}

async function submit() {
  const text = input.value.trim();
  if (!text || controller) return;
  if (!model) await info();
  input.value = ""; autosize();
  history.push({ role: "user", content: text });
  addMsg("user", text);
  const bot = addMsg("bot", ""); const bubble = bot.querySelector(".bubble");
  const think = bubble.querySelector(".think"), thinkBody = think.querySelector("div"), answer = bubble.querySelector(".answer");
  bubble.classList.add("cursor");
  send.textContent = "Stop"; controller = new AbortController();
  const sys = $("system").value.trim();
  const messages = (sys ? [{ role: "system", content: sys }] : []).concat(history);
  const body = { model, messages, max_tokens: +$("maxTokens").value,
                 chat_template_kwargs: { enable_thinking: $("think").checked } };
  if (+$("temp").value >= 0) body.temperature = +$("temp").value;
  if (+$("topp").value < 1) body.top_p = +$("topp").value;
  const t0 = performance.now();
  answer.textContent = "invoking…";
  const tick = setInterval(() => { answer.textContent = `invoking… ${((performance.now() - t0) / 1000).toFixed(0)} s`; }, 1000);
  let reply = "", reasoning = "", usage = null, finish = null, lambdaS = null;
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST", headers: { "content-type": "application/json" }, signal: controller.signal,
      body: JSON.stringify(body),
    });
    lambdaS = r.headers.get("x-lambda-seconds");
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error?.message || `HTTP ${r.status}`);
    const c = (j.choices || [])[0] || {}, m = c.message || {};
    reasoning = m.reasoning || m.reasoning_content || "";
    reply = m.content || "";
    usage = j.usage; finish = c.finish_reason;
    if (reasoning) { think.hidden = false; thinkBody.textContent = reasoning; }
    answer.innerHTML = render(reply);
  } catch (err) {
    answer.innerHTML = err.name === "AbortError" ? render("**Stopped waiting.**") : render(`**Error:** ${err.message}`);
    if (err.name === "AbortError") finish = "stopped";
  }
  clearInterval(tick);
  bubble.classList.remove("cursor");
  const total = (performance.now() - t0) / 1000;
  const n = usage ? usage.completion_tokens : null;
  const rate = n ? n / total : null;
  bot.querySelector(".stats").textContent =
    [n !== null ? `${n} tokens` : null, `${total.toFixed(2)} s`, rate ? `${rate.toFixed(1)} tok/s end to end` : null, finish].filter(Boolean).join(" · ");
  setDL("last", [["round trip", total.toFixed(2) + " s"], ["proxy → Lambda", lambdaS ? (+lambdaS).toFixed(2) + " s" : null],
                 ["tokens", n ?? "—"], ["prompt tokens", usage ? usage.prompt_tokens : "—"],
                 ["end to end", rate ? rate.toFixed(1) + " tok/s" : "—"],
                 ["reasoning", reasoning ? reasoning.length + " chars" : null], ["finish", finish ?? "—"]]);
  log.scrollTop = log.scrollHeight;
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
$("temp").oninput = (e) => { $("tempVal").textContent = +e.target.value < 0 ? "server default" : (+e.target.value).toFixed(1); };

status().then(() => info());
setInterval(async () => { await status(); if (!controller) info(); }, 15000);
input.focus();
</script>
</body>
</html>"""


class Session:
    """Counts for the side panel, summed from each finished invocation."""

    lock = threading.Lock()
    counts = {"requests": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    @classmethod
    def add(cls, **kw):
        with cls.lock:
            for k, v in kw.items():
                cls.counts[k] += v or 0

    @classmethod
    def snapshot(cls):
        with cls.lock:
            return dict(cls.counts)


_fn_cache = {"t": 0.0, "v": None}


def function_state():
    """State and scaling of the published version, cached for 10 s (the page polls every 15)."""
    if time.time() - _fn_cache["t"] < 10 and _fn_cache["v"]:
        return _fn_cache["v"]
    out = {}
    try:
        c = CLIENT.get_function_configuration(FunctionName=E["FUNCTION_NAME"], Qualifier=QUALIFIER)
        out.update(state=c.get("State"), version=c.get("Version"), reason=c.get("StateReason"))
        s = CLIENT.get_function_scaling_config(FunctionName=E["FUNCTION_NAME"], Qualifier=QUALIFIER)
        sc = s.get("AppliedFunctionScalingConfig") or s.get("RequestedFunctionScalingConfig") or {}
        out.update(min=sc.get("MinExecutionEnvironments"), max=sc.get("MaxExecutionEnvironments"))
    except Exception as e:  # noqa: BLE001 - shown on the page
        out["error"] = str(e)[:120]
    _fn_cache.update(t=time.time(), v=out)
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif route == "/ui/status":
            self._json(function_state())
        elif route == "/ui/info":
            keys = (
                "MODEL_NAME",
                "QUANTIZATION",
                "VLLM_VERSION",
                "AWS_REGION",
                "INSTANCE_TYPE",
                "ACCELERATOR_TYPE",
                "FUNCTION_NAME",
                "MEMORY_MB",
                "ACCELERATOR_MEMORY_GB",
                "PER_ENV_MAX_CONCURRENCY",
                "MAX_MODEL_LEN",
            )
            self._json({"rig": {k: E[k] for k in keys}, "function": function_state(), "session": Session.snapshot()})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/v1/chat/completions":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        body.pop("stream", None)
        body.pop("stream_options", None)
        t0 = time.time()
        try:
            r = CLIENT.invoke(FunctionName=f"{E['FUNCTION_NAME']}:{QUALIFIER}", Payload=json.dumps(body))
            payload = json.loads(r["Payload"].read() or b"{}")
            err = r.get("FunctionError")
        except Exception as e:  # noqa: BLE001 - returned to the page
            payload, err = {"errorMessage": str(e)}, "Invoke"
        dt = time.time() - t0
        u = payload.get("usage") or {}
        Session.add(
            requests=1,
            failed=1 if err else 0,
            prompt_tokens=u.get("prompt_tokens"),
            completion_tokens=u.get("completion_tokens"),
            seconds=dt,
        )
        log(f"invoke -> {'error ' + err if err else 'ok'} {u.get('completion_tokens')} tokens {dt:.2f}s")
        if err:
            msg = payload.get("errorMessage") or json.dumps(payload)[:400]
            self._json({"error": {"message": f"{err}: {msg}"}}, 502, dt)
        else:
            self._json(payload, 200, dt)

    def _send(self, code, data, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200, seconds=None):
        extra = {"X-Lambda-Seconds": f"{seconds:.3f}"} if seconds is not None else None
        self._send(code, json.dumps(obj).encode(), "application/json", extra)

    def log_message(self, fmt, *args):
        pass


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def already_running(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/ui/status", timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=int(E.get("UI_PORT", 8768)))
    a = p.parse_args()
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    except OSError as e:
        if already_running(a.port):
            print(f"Chat UI is already running: open http://localhost:{a.port}", flush=True)
            return
        sys.exit(f"port {a.port} is taken by something else ({e}); try --port {a.port + 1}")
    print(
        f"Chat UI on http://localhost:{a.port} -> Lambda {E['FUNCTION_NAME']}:{QUALIFIER} in {E['AWS_REGION']}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
