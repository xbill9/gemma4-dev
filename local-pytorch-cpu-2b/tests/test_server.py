"""Offline unit tests for the local PyTorch/transformers rig.

unittest, never pytest. `mcp` is mocked before `server` is imported; nothing here
loads a checkpoint, touches /proc for real, or imports torch.
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

    def tool(self, *a, **k):
        def deco(fn):
            return fn
        return deco

    async def list_tools(self):
        return []

    def run(self):
        raise AssertionError("mcp.run() must never be called from a test")


_m = MagicMock()
_m.FastMCP = _FakeFastMCP
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.fastmcp"] = _m

import server  # noqa: E402

GIB = 1024 ** 3


class TestRigIdentity(unittest.TestCase):
    def test_rig_name_matches_directory(self):
        self.assertEqual(server.RIG_NAME, RIG_DIR.name)
        self.assertEqual(server.RIG_NAME, "local-pytorch-cpu-2b")


class TestNoCloudControlPlane(unittest.TestCase):
    """The most load-bearing class here.

    This directory was a VERBATIM COPY of `gpu-jax-g4dn-2b` until 2026-09-04 —
    58 tracked files describing an AWS G4dn with a T4, shipping a JAX engine
    under a PyTorch name, plus a skill directory called
    `gpu-jax-g4dn-2b-management` (and `make skill-install` rm -rf's its
    destination). A fork of a cloud rig keeps passing its own tests while
    describing hardware that does not exist.
    """

    # Cloud vocabulary. Checked in identifiers AND string literals, because a
    # region or an instance id hides in a string as easily as in a name.
    FORBIDDEN_CLOUD = frozenset({
        "boto3", "botocore", "ec2", "ssm", "secretsmanager", "instance_id",
        "ami", "spot", "systemd", "queued_resource", "gcloud", "aws_region",
        "deploy_jax_server", "find_tpu",
    })
    # Engine vocabulary. Checked in IMPORTS AND ATTRIBUTES ONLY, never strings —
    # this rig's prose legitimately names `local-jax-cpu-2b` as the sibling it is
    # the control for, and a string scan turns that into a false "jax" hit. The
    # invariant is "does not import or call JAX", which is a code question.
    FORBIDDEN_ENGINE = frozenset({"jax", "jnp", "flax", "jaxlib", "vllm"})

    def _names(self, include_strings: bool):
        """Names the CODE mentions, docstrings and comments excluded.

        Raw-text matching does not work: this rig's prose names
        `gpu-jax-g4dn-2b` as provenance and `local-jax-cpu-2b` as its sibling,
        and a substring search for "ami" hits half the English language.
        """
        import ast
        import re
        tree = ast.parse((RIG_DIR / "server.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)) and node.body:
                f = node.body[0]
                if isinstance(f, ast.Expr) and isinstance(f.value, ast.Constant) \
                        and isinstance(f.value.value, str):
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
                names.update(node.name.split("."))
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.update(node.module.split("."))
            elif include_strings and isinstance(node, ast.Constant) \
                    and isinstance(node.value, str):
                names.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", node.value))
        return {n.lower() for n in names}

    def test_no_cloud_control_plane(self):
        found = self._names(include_strings=True) & self.FORBIDDEN_CLOUD
        self.assertEqual(found, set(), f"cloud vocabulary still in code: {sorted(found)}")

    def test_does_not_import_or_call_another_engine(self):
        found = self._names(include_strings=False) & self.FORBIDDEN_ENGINE
        self.assertEqual(found, set(), f"another engine reached from code: {sorted(found)}")

    def test_the_jax_engine_files_are_gone(self):
        for stale in ("jax_engine.py", "jax_openai_server.py", "ports", "init.sh",
                      "project-setup.sh", "requirements-serving.txt"):
            with self.subTest(stale=stale):
                self.assertFalse((RIG_DIR / stale).exists())

    def test_declares_no_accelerator(self):
        import os
        self.assertEqual(os.environ.get("ACCELERATOR_TYPE"), "none")
        self.assertEqual(os.environ.get("DEVICE"), "cpu")


class TestDeviceChoice(unittest.TestCase):
    """The box HAS a GPU and this rig ignores it on purpose.

    4096 MiB against 10.25 GB of weights, and transformers has no analogue of
    llama.cpp's lazy PLE gather, so device_map="auto" would measure PCIe.
    """

    def test_device_is_cpu_and_not_auto(self):
        self.assertEqual(server.DEVICE, "cpu")
        self.assertNotEqual(server.DEVICE, "auto")

    def test_threads_match_physical_cores_not_smt(self):
        """6 physical cores; the other 6 are SMT siblings sharing execution
        units, and a memory-bound decode gains nothing from them."""
        self.assertEqual(server.TORCH_NUM_THREADS, 6)


def _host(available_gb, total_gb=16.42, swap_gb=15.4):
    return patch.object(server, "_meminfo", return_value={
        "MemTotal": int(total_gb * 1e9),
        "MemAvailable": int(available_gb * 1e9),
        "SwapFree": int(swap_gb * 1e9)})


class TestCapacity(unittest.TestCase):
    def test_sized_from_the_file_not_from_rss(self):
        """SAFETENSORS MMAPS. MEASURED 2026-09-04: RSS right after load is
        1.20 GB for a 10.25 GB checkpoint, and only reaches 5.71 GB once
        generation has walked the weights. Sizing from RSS would say a host that
        cannot hold the model is fine."""
        with _host(11.8):
            c = server._capacity()
        self.assertEqual(c["weights"], server.MODEL_SAFETENSORS_BYTES)
        src = (RIG_DIR / "server.py").read_text()
        self.assertNotIn("_rss_bytes()", src.split("def _capacity")[1].split("def ")[0])

    def test_the_measured_host_fits(self):
        with _host(11.8):
            self.assertTrue(server._capacity()["fits"])

    def test_short_now_is_not_impossible(self):
        with _host(9.4):
            c = server._capacity()
        self.assertFalse(c["fits"])
        self.assertTrue(c["fits_if_freed"])


class TestCapacityReport(unittest.IsolatedAsyncioTestCase):
    async def test_fitting_host_reports_headroom(self):
        with _host(11.8):
            out = await server.check_host_capacity()
        self.assertIn("✅", out)
        self.assertIn("Fits now", out)

    async def test_short_host_warns_about_swap_not_failure(self):
        with _host(9.4):
            out = await server.check_host_capacity()
        self.assertIn("⚠️", out)
        self.assertIn("swap", out.lower())


class TestLoadRefuses(unittest.IsolatedAsyncioTestCase):
    async def test_refuses_when_over_budget(self):
        with _host(9.4), patch.object(server, "_MODEL", None):
            out = await server.load_model()
        self.assertIn("❌", out)
        self.assertIn("Refusing to load", out)


class TestGenerationBudget(unittest.TestCase):
    def test_default_budget_is_generous(self):
        """Gemma 4 emits a thinking block first; a small budget measures the
        thinking phase and returns nothing."""
        self.assertGreaterEqual(server.MAX_NEW_TOKENS, 512)


class TestRunCommand(unittest.IsolatedAsyncioTestCase):
    async def test_missing_binary_returns_127(self):
        rc, _, err = await server.run_command(["definitely-not-a-real-binary-xyz"])
        self.assertEqual(rc, 127)
        self.assertIn("not found", err)


class TestGetHelp(unittest.IsolatedAsyncioTestCase):
    async def test_lists_tools_and_states_the_rig_is_a_control(self):
        t = MagicMock()
        t.name = "check_host_capacity"
        t.description = "Can this host hold the weights?"
        with patch.object(server.mcp, "list_tools", AsyncMock(return_value=[t])):
            out = await server.get_help()
        self.assertIn("check_host_capacity", out)
        self.assertIn("local-jax-cpu-2b", out)


if __name__ == "__main__":
    unittest.main()
