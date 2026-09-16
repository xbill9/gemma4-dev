"""Offline unit tests for the local llama.cpp rig's MCP server.

unittest, never pytest. The whole `mcp` module is mocked before `server` is
imported, so nothing here touches the network, a subprocess, or the GPU.

Because `mcp` is a MagicMock, `mcp.list_tools()` needs an explicit AsyncMock —
see the get_help test.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

RIG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RIG_DIR))

# Mock the mcp package before importing server. A bare MagicMock is NOT enough
# here: `@mcp.tool()` would then return a MagicMock instead of the decorated
# coroutine, and every tool test fails with "'MagicMock' object can't be
# awaited" — which reads as a broken server and is really a broken fake.
# So `tool()` is a pass-through decorator and the real functions survive.


class _FakeMCPServer:
    def __init__(self, name):
        self.name = name

    def tool(self, *args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    async def list_tools(self):
        return []

    def run(self):
        raise AssertionError("mcp.run() must never be called from a test")


_mcpserver_module = MagicMock()
_mcpserver_module.MCPServer = _FakeMCPServer
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.mcpserver"] = _mcpserver_module

import attest  # noqa: E402
import server  # noqa: E402


class TestRigIdentity(unittest.TestCase):
    """The registered name must equal the directory, or two loaded rigs are
    indistinguishable at the call site (root CLAUDE.md)."""

    def test_rig_name_matches_directory(self):
        self.assertEqual(server.RIG_NAME, RIG_DIR.name)
        self.assertEqual(server.RIG_NAME, "local-llamacpp-1650ti-2b-q4_0")

    def test_server_name_defaults_to_rig_name(self):
        self.assertEqual(server.MCP_SERVER_NAME, server.RIG_NAME)


class TestNoProvisioning(unittest.TestCase):
    """A `local` rig that grows capacity-finding machinery has the wrong name.

    This is the test that keeps the slot-1 claim honest, so it asserts on the
    source rather than on the exported symbols — a helper is as much of a
    violation as a tool.
    """

    FORBIDDEN = ("find_tpu", "queued_resource", "gcloud", "boto3", "tpu_zones_status")

    def test_source_has_no_provisioning(self):
        source = (RIG_DIR / "server.py").read_text()
        for token in self.FORBIDDEN:
            with self.subTest(token=token):
                # Allowed in prose explaining the absence; not in code.
                code = "\n".join(
                    line for line in source.splitlines()
                    if not line.lstrip().startswith("#")
                )
                self.assertNotIn(token + "(", code)

    def test_no_mmap_is_never_passed(self):
        """--no-mmap defeats TENSOR_READ_LAZY and turns a comfortable fit into an OOM."""
        source = (RIG_DIR / "server.py").read_text()
        self.assertNotIn('"--no-mmap"', source)


class TestModelInfo(unittest.IsolatedAsyncioTestCase):
    async def test_missing_model_path_is_reported(self):
        with patch.object(server, "MODEL_PATH", ""):
            out = await server.model_info()
        self.assertIn("❌", out)
        self.assertIn("MODEL_PATH", out)

    async def test_missing_file_is_reported(self):
        with patch.object(server, "MODEL_PATH", "/nonexistent/model.gguf"):
            out = await server.model_info()
        self.assertIn("❌", out)
        self.assertIn("not found", out)


class TestStartModelServer(unittest.IsolatedAsyncioTestCase):
    async def test_refuses_when_binary_missing(self):
        with patch.object(server, "LLAMA_SERVER_BIN", "/nonexistent/llama-server"):
            out = await server.start_model_server()
        self.assertIn("❌", out)
        self.assertIn("llama-server", out)

    async def test_reports_already_running(self):
        with patch.object(server, "LLAMA_SERVER_BIN", __file__), \
             patch.object(server, "MODEL_PATH", __file__), \
             patch.object(server, "_read_pid", return_value=4242):
            out = await server.start_model_server()
        self.assertIn("✅", out)
        self.assertIn("4242", out)


class TestSpawn(unittest.IsolatedAsyncioTestCase):
    """The daemon spawn must outlive the process that made it."""

    async def test_spawns_detached(self):
        proc = MagicMock()
        proc.pid = 5150
        spawn = MagicMock(return_value=proc)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(server, "LLAMA_SERVER_BIN", __file__), \
             patch.object(server, "MODEL_PATH", __file__), \
             patch.object(server, "_read_pid", return_value=None), \
             patch.object(server, "RUN_DIR", Path(tmp)), \
             patch.object(server, "PID_FILE", Path(tmp) / "pid"), \
             patch.object(server, "LOG_FILE", Path(tmp) / "log"), \
             patch.object(server.subprocess, "Popen", spawn):
            out = await server.start_model_server()
        self.assertIn("5150", out)
        argv = spawn.call_args.args[0]
        self.assertEqual(argv[argv.index("-ngl") + 1], str(server.N_GPU_LAYERS))
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])

    async def test_spawn_survives_this_process(self):
        """REGRESSION 2026-09-16. asyncio's subprocess transport kills a live
        child when it is torn down (__del__ -> close() -> _proc.kill()), and
        start_new_session does not prevent it, so every server started through
        this tool died with the interpreter. Verified against a real `sleep 60`.
        subprocess.Popen only warns; it does not kill."""
        src = (RIG_DIR / "server.py").read_text()
        spawn = src[src.index("def _spawn_detached"):src.index("async def start_model_server")]
        # The docstring names the trap; the CODE must not use it.
        body = spawn.split('"""')[-1]
        self.assertIn("subprocess.Popen", body)
        self.assertNotIn("create_subprocess_exec", body)
        self.assertIn("start_new_session=True", body)


class TestStopModelServer(unittest.IsolatedAsyncioTestCase):
    async def test_stop_when_not_running(self):
        # PID_FILE is a PosixPath, whose methods are read-only — patch the
        # module attribute, not the method on the instance.
        with patch.object(server, "_read_pid", return_value=None), \
             patch.object(server, "PID_FILE", MagicMock()):
            out = await server.stop_model_server()
        self.assertIn("✅", out)
        self.assertIn("Not running", out)


class TestPidDiscovery(unittest.TestCase):
    """A server started outside this process must still be found.

    `make serve` is foreground and writes no pid file, so "no pid file" is the
    NORMAL state here, not an edge case. Reading it as "not running" made
    model_server_status report ❌ at a healthy endpoint and made
    stop_model_server claim success without stopping anything (2026-09-08).
    """

    def test_falls_back_to_port_owner_when_no_pid_file(self):
        with patch.object(server, "_pidfile_pid", return_value=None), \
             patch.object(server, "_pid_owning_port", return_value=31337) as owner:
            self.assertEqual(server._read_pid(), 31337)
        owner.assert_called_once_with(int(server.PORT))

    def test_pid_file_wins_and_skips_the_scan(self):
        with patch.object(server, "_pidfile_pid", return_value=42), \
             patch.object(server, "_pid_owning_port") as owner:
            self.assertEqual(server._read_pid(), 42)
        owner.assert_not_called()

    def test_none_when_nothing_owns_the_port(self):
        with patch.object(server, "_pidfile_pid", return_value=None), \
             patch.object(server, "_pid_owning_port", return_value=None):
            self.assertIsNone(server._read_pid())

    def test_listening_inodes_parses_proc_net_tcp(self):
        # 0A is TCP_LISTEN; 1F90 is 8080. The second row is ESTABLISHED on the
        # same port and must not be picked up, or we would find a client.
        table = (
            "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
            "   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 55501 1 0 0\n"
            "   1: 0100007F:1F90 0100007F:C001 01 00000000:00000000 00:00000000 00000000  1000        0 55502 1 0 0\n"
            "   2: 0100007F:0050 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 55503 1 0 0\n"
        )
        with patch.object(server.Path, "read_text", return_value=table):
            self.assertEqual(attest.listening_inodes(8080), {"55501"})
            self.assertEqual(attest.listening_inodes(80), {"55503"})
            self.assertEqual(attest.listening_inodes(9999), set())

    def test_listening_inodes_survives_missing_proc_files(self):
        with patch.object(server.Path, "read_text", side_effect=OSError("no /proc")):
            self.assertEqual(attest.listening_inodes(8080), set())


class TestStopModelServerDiscovered(unittest.IsolatedAsyncioTestCase):
    async def test_stops_a_server_it_did_not_start(self):
        killed = []
        with patch.object(server, "_read_pid", return_value=4242), \
             patch.object(server, "_pidfile_pid", return_value=None), \
             patch.object(server, "PID_FILE", MagicMock()), \
             patch.object(server.os, "kill", side_effect=lambda p, s: killed.append((p, s))):
            out = await server.stop_model_server()
        self.assertEqual(killed, [(4242, server.signal.SIGTERM)])
        self.assertIn("✅", out)
        self.assertIn("not started through this server", out)


def fake_attest(device="gpu", **over):
    """An attestation as attest_port would return it. Offline: /proc is not read."""
    att = {
        "serving": True, "port": 8080, "pid": 4242,
        "exe": "/home/xbill/llama.cpp/build/bin/llama-server",
        "exe_sha256": "a" * 64,
        "device": device,
        "gpu_libs": [] if device == "cpu" else ["ggml-cuda", "libcuda"],
        "n_gpu_layers": 0 if device in ("cpu", "mixed") else 99,
        "cuda_visible_devices": "" if device == "cpu" else "0",
        "model_arg": "/home/xbill/models/gemma-4-E2B-it-qat-q4_0/gemma-4-E2B_q4_0-it.gguf",
        "threads": "4", "threads_batch": "8", "ctx_size": "8192",
        "argv": ["llama-server"],
    }
    att.update(over)
    return att


class TestModelServerStatus(unittest.IsolatedAsyncioTestCase):
    """/health decides whether something is UP; attestation decides whose arm it is."""

    def _client(self, status_code=None, error=None):
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        if error is not None:
            client.get = AsyncMock(side_effect=error)
        else:
            resp = MagicMock()
            resp.status_code = status_code
            client.get = AsyncMock(return_value=resp)
        return client

    async def test_healthy_without_a_pid_file_is_success(self):
        # THE REGRESSION. Before the fix this returned ❌ at a live server.
        with patch.object(server, "_read_pid", return_value=4242), \
             patch.object(server, "_pidfile_pid", return_value=None), \
             patch.object(server, "attest_port", return_value=fake_attest("gpu")), \
             patch.object(server.httpx, "AsyncClient", return_value=self._client(200)):
            out = await server.model_server_status()
        self.assertIn("✅", out)
        self.assertIn("Serving", out)
        self.assertIn("not started through this server", out)

    async def test_healthy_from_pid_file_is_success(self):
        with patch.object(server, "_read_pid", return_value=4242), \
             patch.object(server, "_pidfile_pid", return_value=4242), \
             patch.object(server, "attest_port", return_value=fake_attest("gpu")), \
             patch.object(server.httpx, "AsyncClient", return_value=self._client(200)):
            out = await server.model_server_status()
        self.assertIn("✅", out)
        self.assertNotIn("not started through this server", out)

    async def test_healthy_but_cpu_arm_is_not_success(self):
        """THE CONTROL'S SILENT FAILURE, from this side: the CPU twin serves the
        same model on this same port."""
        with patch.object(server, "_read_pid", return_value=4242), \
             patch.object(server, "_pidfile_pid", return_value=None), \
             patch.object(server, "attest_port", return_value=fake_attest("cpu")), \
             patch.object(server.httpx, "AsyncClient", return_value=self._client(200)):
            out = await server.model_server_status()
        self.assertIn("❌", out)
        self.assertNotIn("✅", out)
        self.assertIn("not this arm", out)

    async def test_nothing_running_is_reported_down(self):
        with patch.object(server, "_read_pid", return_value=None), \
             patch.object(server, "_pidfile_pid", return_value=None), \
             patch.object(server.httpx, "AsyncClient",
                          return_value=self._client(error=server.httpx.ConnectError("refused"))):
            out = await server.model_server_status()
        self.assertIn("❌", out)
        self.assertIn("not running", out)

    async def test_up_but_not_answering_is_still_loading(self):
        with patch.object(server, "_read_pid", return_value=4242), \
             patch.object(server, "_pidfile_pid", return_value=4242), \
             patch.object(server.httpx, "AsyncClient",
                          return_value=self._client(error=server.httpx.ConnectError("refused"))):
            out = await server.model_server_status()
        self.assertIn("📡", out)
        self.assertIn("Still loading", out)


class TestMetricsFlag(unittest.TestCase):
    """/metrics is llama.cpp's endpoint and is 501 unless --metrics is passed."""

    def test_flag_is_passed_when_enabled(self):
        with patch.object(server, "METRICS", "1"):
            self.assertIn("--metrics", server._server_command())

    def test_flag_is_absent_when_disabled(self):
        with patch.object(server, "METRICS", "0"):
            self.assertNotIn("--metrics", server._server_command())


class TestServerCommandMatchesMakefile(unittest.TestCase):
    """`make serve` and `start_model_server` must launch the same server.

    They drifted once: 2026-09-10 the MCP tool started llama-server without -fa,
    -t or --parallel, so it came up with 4 slots and 6 threads while `make serve`
    carried the measured config. A benchmark against the MCP-started server
    would have measured a configuration no tpu.env line describes.
    """

    @staticmethod
    def _makefile_flags() -> set:
        import re
        text = (RIG_DIR / "Makefile").read_text()
        recipe = text.split("\nserve:\n", 1)[1].split("\n\n", 1)[0]
        return set(re.findall(r"(?<![\w-])(--?[a-z][\w-]*)", recipe))

    def test_same_flag_set(self):
        with patch.object(server, "METRICS", "1"):
            cmd = server._server_command()
        self.assertEqual({t for t in cmd if t.startswith("-")}, self._makefile_flags())

    def test_measured_levers_come_from_tpu_env(self):
        with patch.object(server, "FLASH_ATTENTION", "1"), \
             patch.object(server, "THREADS", "4"), \
             patch.object(server, "PARALLEL_SLOTS", "1"):
            cmd = server._server_command()
        for flag, value in (("-fa", "1"), ("-t", "4"), ("--parallel", "1")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value, flag)

    def test_context_size_override(self):
        cmd = server._server_command("16384")
        self.assertEqual(cmd[cmd.index("-c") + 1], "16384")

    def test_never_disables_mmap(self):
        self.assertNotIn("--no-mmap", server._server_command())


class TestRunCommand(unittest.IsolatedAsyncioTestCase):
    async def test_missing_binary_returns_127(self):
        rc, _, err = await server.run_command(["definitely-not-a-real-binary-xyz"])
        self.assertEqual(rc, 127)
        self.assertIn("not found", err)


class TestQueryModelReasoning(unittest.IsolatedAsyncioTestCase):
    """Gemma 4 emits a thinking block, and llama.cpp puts it in `reasoning_content`.

    A caller that reads only `content` sees an empty string and concludes the
    server is broken. MEASURED 2026-09-03: "Name three TPU generations" spent
    1274 chars reasoning before 22 chars of answer, so a small max_tokens
    reliably produces exactly that empty string.
    """

    def _response(self, content, reasoning, finish_reason="stop"):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"finish_reason": finish_reason,
                         "message": {"role": "assistant",
                                     "content": content,
                                     "reasoning_content": reasoning}}],
            "usage": {"prompt_tokens": 24, "completion_tokens": 64},
            "timings": {"predicted_per_second": 71.34},
        }
        return resp

    async def test_reasoning_only_is_not_reported_as_success(self):
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(return_value=self._response("", "Thinking Process: ...", "length"))
        with patch.object(server, "attest_port", return_value=fake_attest("gpu")), \
             patch.object(server.httpx, "AsyncClient", return_value=client):
            out = await server.query_model("hi", max_tokens=64)
        self.assertNotIn("✅", out)
        self.assertIn("Reasoning only", out)
        self.assertIn("max_tokens", out)

    async def test_answer_reports_reasoning_was_suppressed(self):
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(return_value=self._response("TPU v1, TPU v2, TPU v5", "x" * 1274))
        with patch.object(server, "attest_port", return_value=fake_attest("gpu")), \
             patch.object(server.httpx, "AsyncClient", return_value=client):
            out = await server.query_model("hi")
        self.assertIn("✅", out)
        self.assertIn("TPU v1", out)
        self.assertIn("1274 chars of reasoning", out)

    async def test_refuses_to_query_the_cpu_arm(self):
        """Refuse BEFORE producing a number that would be labelled with the wrong device."""
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(return_value=self._response("some answer", ""))
        with patch.object(server, "attest_port", return_value=fake_attest("cpu")), \
             patch.object(server.httpx, "AsyncClient", return_value=client):
            out = await server.query_model("hi")
        self.assertIn("❌", out)
        self.assertIn("Refusing", out)
        client.post.assert_not_called()

    def test_default_max_tokens_is_generous(self):
        import inspect
        default = inspect.signature(server.query_model).parameters["max_tokens"].default
        self.assertGreaterEqual(default, 512, "small default truncates mid-thought and returns empty content")


class TestGetHelp(unittest.IsolatedAsyncioTestCase):
    async def test_lists_tools(self):
        tool = MagicMock()
        tool.name = "gpu_status"
        tool.description = "Report the local GPU: name, compute capability, VRAM."
        with patch.object(server.mcp, "list_tools", AsyncMock(return_value=[tool])):
            out = await server.get_help()
        self.assertIn("gpu_status", out)
        self.assertIn("control plane", out)


if __name__ == "__main__":
    unittest.main()
