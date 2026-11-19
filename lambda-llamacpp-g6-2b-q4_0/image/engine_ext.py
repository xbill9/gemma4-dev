#!/usr/bin/env python3
"""Lambda extension that runs this image's serving engine: ENGINE_CMD, on 127.0.0.1:ENGINE_PORT.

The same lifecycle as AWS's engine extension in its vLLM and SGLang Lambda images
(/opt/engine-extension/engine_extension.py there), for an engine those images do not carry:

  Init      register for SHUTDOWN, start ENGINE_CMD in its own process group, wait until
            ENGINE_HEALTH_PATH answers 200. On failure report /extension/init/error and exit, so
            Lambda discards the environment. Init ends only when every extension asks for its next
            event, so no invocation reaches the handler before the engine is ready.
  Serving   a watchdog checks the process and the health path every ENGINE_HEALTH_INTERVAL s; after
            ENGINE_HEALTH_FAILURES failed probes, or as soon as the process exits, it reports
            /extension/exit/error and exits, so Lambda replaces the environment.
  Shutdown  SIGTERM the process group, SIGKILL after 1 s (Lambda allows extensions 2 s).

Environment: ENGINE_CMD (shell-quoted), ENGINE_PORT (8000), ENGINE_HEALTH_PATH (/health),
ENGINE_SERVER_TIMEOUT (600), ENGINE_HEALTH_INTERVAL (10), ENGINE_HEALTH_TIMEOUT (30),
ENGINE_HEALTH_FAILURES (3).
"""

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

API = "2020-01-01/extension"


def log(msg):
    print(f"[engine-ext] {msg}", file=sys.stderr, flush=True)


class Client:
    def __init__(self, name):
        self.base = f"http://{os.environ['AWS_LAMBDA_RUNTIME_API']}/{API}"
        self.name, self.id = name, None

    def _req(self, method, path, body=None, headers=None, timeout=None):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(f"{self.base}/{path}", data=data, method=method, headers=headers or {})
        if data is not None:
            r.add_header("Content-Type", "application/json")
        return urllib.request.urlopen(r, timeout=timeout)

    def register(self):
        with self._req("POST", "register", {"events": ["SHUTDOWN"]}, {"Lambda-Extension-Name": self.name}, 30) as r:
            self.id = r.headers["Lambda-Extension-Identifier"]

    def next_event(self):
        with self._req("GET", "event/next", headers={"Lambda-Extension-Identifier": self.id}) as r:
            return json.loads(r.read() or b"{}")

    def error(self, phase, kind, msg):
        try:
            self._req("POST", f"{phase}/error", {"errorMessage": msg, "errorType": kind, "stackTrace": []},
                      {"Lambda-Extension-Identifier": self.id, "Lambda-Extension-Function-Error-Type": kind}, 5).close()
        except OSError as e:
            log(f"could not report {phase} error: {e}")


def healthy(url, timeout):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def stop(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        for _ in range(20):
            if proc.poll() is not None:
                return
            time.sleep(0.05)
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main(name):
    env = os.environ
    client = Client(name)
    client.register()
    url = f"http://127.0.0.1:{env.get('ENGINE_PORT', '8000')}{env.get('ENGINE_HEALTH_PATH', '/health')}"
    probe = int(env.get("ENGINE_HEALTH_TIMEOUT", "30"))
    proc = None
    try:
        cmd = shlex.split(env["ENGINE_CMD"])
        log(f"starting: {shlex.join(cmd)}")
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, start_new_session=True)
        deadline = t0 + int(env.get("ENGINE_SERVER_TIMEOUT", "600"))
        while not healthy(url, probe):
            if proc.poll() is not None:
                raise RuntimeError(f"engine exited with code {proc.returncode} before {url} answered")
            if time.monotonic() > deadline:
                raise RuntimeError(f"{url} did not answer within ENGINE_SERVER_TIMEOUT")
            time.sleep(1)
        log(f"engine is healthy after {time.monotonic() - t0:.1f} s")
    except Exception as e:  # noqa: BLE001 - reported to Lambda, which discards the environment
        log(f"init failed: {e}")
        client.error("init", "Extension.EngineInitFailed", str(e))
        if proc is not None:
            stop(proc)
        return 1

    stopping = threading.Event()

    def watchdog():
        fails, every, limit = 0, int(env.get("ENGINE_HEALTH_INTERVAL", "10")), int(env.get("ENGINE_HEALTH_FAILURES", "3"))
        while True:
            for _ in range(every):
                if proc.poll() is not None:
                    reason = f"engine exited with code {proc.returncode}"
                    break
                time.sleep(1)
            else:
                fails = 0 if healthy(url, probe) else fails + 1
                if fails < limit:
                    continue
                reason = f"engine failed {fails} health probes"
            if stopping.is_set():
                return
            log(f"{reason}; recycling the execution environment")
            client.error("exit", "Extension.EngineUnhealthy", reason)
            stop(proc)
            os._exit(1)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        while True:
            ev = client.next_event()
            if ev.get("eventType") == "SHUTDOWN":
                log(f"shutdown: {ev.get('shutdownReason')}")
                return 0
    finally:
        stopping.set()
        stop(proc)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
