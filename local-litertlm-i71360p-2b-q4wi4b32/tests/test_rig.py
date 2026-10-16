"""Offline tests: no network, no subprocess, no model file. unittest, never pytest."""

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RIG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RIG))

# Mock the whole mcp module BEFORE importing server (root CLAUDE.md).
_mcp = mock.MagicMock()
_mcp.server.mcpserver.MCPServer.return_value.tool.return_value = lambda f: f
sys.modules.setdefault("mcp", _mcp)
sys.modules.setdefault("mcp.server", _mcp.server)
sys.modules.setdefault("mcp.server.mcpserver", _mcp.server.mcpserver)

import numpy as np  # noqa: E402

import attest  # noqa: E402
import build_litertlm  # noqa: E402
import server  # noqa: E402


def _env():
    out = {}
    for line in (RIG / "tpu.env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k] = v.strip('"')
    return out


class TestNaming(unittest.TestCase):
    def test_server_named_after_directory(self):
        self.assertEqual(server.RIG_NAME, RIG.name)

    def test_model_id_is_model_name(self):
        env = _env()
        self.assertEqual(server.model_id(env["MODEL_PATH"]), env["MODEL_NAME"])

    def test_model_id_plain_file(self):
        self.assertEqual(server.model_id("/x/y/foo.litertlm"), "foo.litertlm")


class TestCpuOnly(unittest.TestCase):
    def test_expected_device_is_cpu(self):
        self.assertEqual(attest.EXPECTED_DEVICE, "cpu")

    def test_backend_not_from_env(self):
        with mock.patch.dict(os.environ, {"BACKEND": "gpu"}):
            self.assertEqual(server.render_config()["models"][server.model_id()]["backend"], "cpu")

    def test_config_values_from_env(self):
        env = _env()
        entry = server.render_config()["models"][env["MODEL_NAME"]]
        self.assertEqual(entry["max_num_tokens"], int(env["MAX_NUM_TOKENS"]))
        self.assertEqual(entry["cpu_thread_count"], int(env["CPU_THREAD_COUNT"]))
        self.assertEqual(entry["thinking"], env["THINKING"] == "true")

    def test_max_num_tokens_fits_export(self):
        env = _env()
        self.assertLessEqual(int(env["MAX_NUM_TOKENS"]), int(env["CACHE_LENGTH"]))


class TestServeParity(unittest.TestCase):
    def test_make_serve_matches_server_command(self):
        mk = (RIG / "Makefile").read_text()
        line = re.search(r"^serve:.*\n\t(.+)$", mk, re.M).group(1)
        for flag in ("serve", "--config", "--host", "--port"):
            self.assertIn(flag, line)
            self.assertIn(flag, server._server_command())

    def test_make_config_uses_render_config(self):
        self.assertIn("server.render_config()", (RIG / "Makefile").read_text())


class TestAttest(unittest.TestCase):
    def test_is_litertlm_serve(self):
        self.assertTrue(attest.is_litertlm_serve(["/usr/bin/python3", "/home/x/bin/litert-lm", "serve"]))
        self.assertFalse(attest.is_litertlm_serve(["/opt/llama-server", "-m", "x.gguf"]))
        self.assertFalse(attest.is_litertlm_serve(["litert-lm", "run"]))

    def test_config_entry_overrides_default(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"default": {"backend": "gpu", "cpu_thread_count": 4}, "models": {"a/b": {"backend": "cpu"}}}, f)
        try:
            self.assertEqual(attest.config_entry(f.name, "a/b"), {"backend": "cpu", "cpu_thread_count": 4})
            self.assertEqual(attest.config_entry(f.name, "c")["backend"], "gpu")
            self.assertEqual(attest.config_entry("/nonexistent", "a/b"), {})
        finally:
            os.unlink(f.name)

    def test_mixed_and_foreign(self):
        argv = ["python3", "litert-lm", "serve", "--config", "/c.json"]
        cases = [
            ("cpu", [], argv, "cpu"),
            ("cpu", ["libOpenCL"], argv, "mixed"),
            ("gpu", ["libOpenCL"], argv, "gpu"),
            (None, [], argv, "mixed"),
            ("cpu", [], ["llama-server"], "foreign"),
        ]
        for backend, libs, av, want in cases:
            with (
                mock.patch.object(attest, "pid_owning_port", return_value=1),
                mock.patch.object(attest, "_cmdline", return_value=av),
                mock.patch.object(attest, "config_entry", return_value={"backend": backend}),
                mock.patch.object(attest, "_mapped_gpu_libs", return_value=libs),
            ):
                self.assertEqual(attest.attest_port(8092, "m")["device"], want, (backend, libs, av))

    def test_mismatch(self):
        self.assertIsNone(attest.mismatch({"serving": True, "device": "cpu", "port": 1}))
        self.assertIn(
            "not `litert-lm serve`", attest.mismatch({"serving": True, "device": "foreign", "port": 1, "argv": ["x"]})
        )


class TestGridStep(unittest.TestCase):
    def test_recovers_step_with_level_minus_8(self):
        rng = np.random.default_rng(0)
        step = np.float32(0.0123)
        levels = rng.integers(-8, 8, size=(64, 32)).astype(np.float32)
        levels[:, 0] = -8  # the case max|w|/7 gets wrong
        g = levels * step
        s, ok = build_litertlm.grid_step(g)
        self.assertTrue(ok.all())
        np.testing.assert_allclose(s[..., 0], step, rtol=1e-6)
        np.testing.assert_array_equal(np.rint(g / s), levels)

    def test_chunked_matches_whole(self):
        rng = np.random.default_rng(2)
        g = (rng.integers(-8, 8, size=(37, 5, 32)) * np.float32(0.01)).astype(np.float32)
        g[3] = rng.normal(size=(5, 32))
        s, ok = build_litertlm.grid_step(g)
        cs, cok = build_litertlm.grid_step_chunked(g, rows=8)
        np.testing.assert_array_equal(cs, s)
        np.testing.assert_array_equal(cok, ok)

    def test_off_grid_falls_back(self):
        g = np.random.default_rng(1).normal(size=(4, 32)).astype(np.float32)
        s, ok = build_litertlm.grid_step(g)
        self.assertFalse(ok.any())
        np.testing.assert_allclose(s[..., 0], np.abs(g).max(-1) / 7, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
