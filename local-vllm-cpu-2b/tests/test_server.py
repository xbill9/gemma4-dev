"""Offline unit tests for the local vLLM-on-CPU rig's MCP server.

unittest, never pytest. The whole `mcp` module is mocked before `server` is
imported, so nothing here touches the network, a subprocess, or /proc for real.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

RIG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RIG_DIR))


class _FakeFastMCP:
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


_fastmcp_module = MagicMock()
_fastmcp_module.FastMCP = _FakeFastMCP
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.fastmcp"] = _fastmcp_module

import server  # noqa: E402

GIB = 1024 ** 3


class TestRigIdentity(unittest.TestCase):
    def test_rig_name_matches_directory(self):
        self.assertEqual(server.RIG_NAME, RIG_DIR.name)
        self.assertEqual(server.RIG_NAME, "local-vllm-cpu-2b")

    def test_server_name_defaults_to_rig_name(self):
        self.assertEqual(server.MCP_SERVER_NAME, server.RIG_NAME)


class TestNoCloudControlPlane(unittest.TestCase):
    """The most load-bearing class in this suite.

    This directory was a VERBATIM COPY of `gpu-jax-g4dn-2b` until 2026-09-04 —
    its README, CLAUDE.md, engine and `tpu.env` all described an AWS G4dn
    instance with a T4, and 58 files were tracked under that identity including
    a skill directory named `gpu-jax-g4dn-2b-management`. A fork of a cloud rig
    keeps passing its own tests while describing hardware that does not exist,
    so the test that matters is the one asserting the vocabulary is GONE.
    """

    FORBIDDEN = frozenset({
        "boto3", "botocore", "ec2", "ssm", "secretsmanager", "instance_id",
        "ami", "spot", "systemd", "queued_resource", "gcloud", "subprocess_ssm",
        "aws_region", "aws_profile", "hf_secret_id", "deploy_jax_server",
        "find_tpu", "create_g5g_instance", "resolve_ami",
    })

    def _identifiers(self):
        """Every name the module's CODE mentions — imports, variables, attributes,
        functions, and string literals — with docstrings and comments EXCLUDED.

        Checking the raw text does not work: this file's own docstring names
        `gpu-jax-g4dn-2b` as provenance, and a substring search for "ami" hits
        half the English language. The invariant is about code, so the test reads
        code. A helper is as much of a violation as a tool, which is why this
        collects every identifier rather than only the tool names.
        """
        import ast
        import re
        tree = ast.parse((RIG_DIR / "server.py").read_text())
        # Drop docstrings: they are prose and may legitimately name the cloud rig
        # this directory was forked from.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                        and isinstance(first.value.value, str):
                    node.body = node.body[1:] or [ast.Pass()]
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.alias):
                names.add(node.name.split(".")[0])
                names.update(node.name.split("."))
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.update(node.module.split("."))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", node.value))
        return {n.lower() for n in names}

    def test_source_names_no_cloud_control_plane(self):
        found = self._identifiers() & self.FORBIDDEN
        self.assertEqual(found, set(), f"cloud control-plane vocabulary in code: {sorted(found)}")

    def test_no_accelerator_is_claimed(self):
        import os
        self.assertEqual(os.environ.get("ACCELERATOR_TYPE"), "none")
        self.assertEqual(os.environ.get("CHIP_COUNT"), "0")


class TestArtifact(unittest.TestCase):
    def test_model_is_a_safetensors_hub_id_not_a_gguf(self):
        """vLLM cannot load a GGUF on any platform (verified 2026-09-02), so the
        3.35 GB q4_0 file the two 1650 Ti rigs share is unusable here and this
        rig needs its own 10.25 GB download."""
        self.assertIn("/", server.MODEL_NAME)
        self.assertFalse(server.MODEL_NAME.endswith(".gguf"))
        self.assertNotIn(":", server.MODEL_NAME)


def _fake_host(available_gb, total_gb=16.42, swap_gb=15.39, flags=("avx2", "f16c")):
    mem = {
        "MemTotal": int(total_gb * 1e9),
        "MemAvailable": int(available_gb * 1e9),
        "SwapFree": int(swap_gb * 1e9),
    }
    return (patch.object(server, "_meminfo", return_value=mem),
            patch.object(server, "_cpu_flags", return_value=set(flags)))


class TestCapacity(unittest.TestCase):
    """The arithmetic is the rig.

    AN EARLIER VERSION DOUBLED THE WEIGHTS ON A HOST WITHOUT AVX512-BF16, on the
    assumption that the CPU backend upcasts bf16 to fp32. It does not: vLLM's
    `CpuPlatform.supported_dtypes` returns [bfloat16, float16, float32] for x86
    unconditionally. These tests pin the corrected model, because the wrong one
    inflated the reported shortfall 2x and made the rig look impossible.
    """

    def test_isa_does_not_change_the_byte_count(self):
        m, f = _fake_host(9.48)
        with m, f:
            no512 = server._capacity()
        m, f = _fake_host(9.48, flags=("avx512f", "avx512_bf16"))
        with m, f:
            with512 = server._capacity()
        self.assertEqual(no512["weights"], with512["weights"])
        self.assertEqual(no512["weights"], server.MODEL_SAFETENSORS_BYTES)

    def test_source_carries_no_upcast_doubling(self):
        self.assertNotIn("2 if upcasts else 1", (RIG_DIR / "server.py").read_text())

    def test_this_host_is_short_but_the_machine_is_not(self):
        """'Does not fit right now' and 'does not fit this machine' have totally
        different remedies, and conflating them is what made an earlier version
        of this rig report a dead end."""
        m, f = _fake_host(9.48)
        with m, f:
            c = server._capacity()
        self.assertFalse(c["fits"])
        self.assertTrue(c["fits_if_freed"])
        self.assertLess(c["shortfall"], 4e9)

    def test_a_freed_host_fits(self):
        m, f = _fake_host(14.0)
        with m, f:
            self.assertTrue(server._capacity()["fits"])

    def test_w4a16_is_not_a_quarter_of_bf16(self):
        """The checkpoint's `ignore` list keeps the vision tower and embeddings
        at bf16, so 4-bit buys 19%, not 75%. A weights/4 estimate under-predicts
        by ~3x and is what produced the wrong '~2.9 GB' figure."""
        ratio = server.MODEL_W4A16_BYTES / server.MODEL_SAFETENSORS_BYTES
        self.assertGreater(ratio, 0.7)
        self.assertLess(ratio, 0.9)

    def test_meminfo_uses_available_not_free(self):
        """MemFree excludes reclaimable page cache and reads catastrophically
        low on a live desktop; MemAvailable is what an allocation can get."""
        source = (RIG_DIR / "server.py").read_text()
        self.assertIn("MemAvailable", source)
        self.assertNotIn('mem.get("MemFree"', source)


class TestCapacityReport(unittest.IsolatedAsyncioTestCase):
    async def test_short_now_is_distinguished_from_impossible(self):
        m, f = _fake_host(9.48)
        with m, f:
            out = await server.check_host_capacity()
        self.assertIn("⚠️", out)
        self.assertIn("RIGHT NOW", out)
        self.assertIn("fits this MACHINE", out)
        self.assertIn("swap", out.lower())

    async def test_report_always_offers_the_quantized_route(self):
        m, f = _fake_host(9.48)
        with m, f:
            out = await server.check_host_capacity()
        self.assertIn("w4a16", out)
        self.assertIn("8.32 GB", out)
        self.assertIn("19% cut", out)

    async def test_a_fitting_host_reports_headroom(self):
        m, f = _fake_host(14.0)
        with m, f:
            out = await server.check_host_capacity()
        self.assertIn("✅", out)
        self.assertIn("Fits now", out)


class TestStartRefuses(unittest.IsolatedAsyncioTestCase):
    """`start_vllm_server` refusing is the one decision this rig takes away from
    the operator. Exceeding host RAM is accepted by the kernel and paid for in
    swap, so an over-budget serve is indistinguishable from a loading one."""

    async def test_refuses_when_over_budget(self):
        m, f = _fake_host(9.48)
        with m, f, patch.object(server, "_read_pid", return_value=None):
            out = await server.start_vllm_server()
        self.assertIn("❌", out)
        self.assertIn("Refusing to start", out)

    async def test_reports_already_running(self):
        with patch.object(server, "_read_pid", return_value=4242):
            out = await server.start_vllm_server()
        self.assertIn("✅", out)
        self.assertIn("4242", out)


class TestStop(unittest.IsolatedAsyncioTestCase):
    async def test_stop_when_not_running(self):
        with patch.object(server, "_read_pid", return_value=None), \
             patch.object(server, "PID_FILE", MagicMock()):
            out = await server.stop_vllm_server()
        self.assertIn("✅", out)
        self.assertIn("Not running", out)


class TestStatusReportsRam(unittest.IsolatedAsyncioTestCase):
    async def test_not_running_still_reports_ram(self):
        """On an accelerator rig 'still loading' and 'thrashing' look different.
        Here they do not, and collapsed available RAM is the only tell."""
        m, f = _fake_host(9.48)
        with m, f, patch.object(server, "_read_pid", return_value=None):
            out = await server.server_status()
        self.assertIn("RAM available", out)


class TestQueryModel(unittest.IsolatedAsyncioTestCase):
    def test_default_max_tokens_is_generous(self):
        import inspect
        default = inspect.signature(server.query_model).parameters["max_tokens"].default
        self.assertGreaterEqual(default, 512,
                                "a small default truncates mid-thought on a reasoning model")


class TestRunCommand(unittest.IsolatedAsyncioTestCase):
    async def test_missing_binary_returns_127(self):
        rc, _, err = await server.run_command(["definitely-not-a-real-binary-xyz"])
        self.assertEqual(rc, 127)
        self.assertIn("not found", err)


class TestGetHelp(unittest.IsolatedAsyncioTestCase):
    async def test_lists_tools_and_points_at_the_reachable_baseline(self):
        tool = MagicMock()
        tool.name = "check_host_capacity"
        tool.description = "Decide whether this host can serve the checkpoint in RAM."
        with patch.object(server.mcp, "list_tools", AsyncMock(return_value=[tool])):
            out = await server.get_help()
        self.assertIn("check_host_capacity", out)
        self.assertIn("local-jax-cpu-2b", out)


if __name__ == "__main__":
    unittest.main()
