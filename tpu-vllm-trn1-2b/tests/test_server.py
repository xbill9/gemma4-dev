"""Offline regression tests for the trn1 MCP server."""

import asyncio
import base64
import filecmp
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import server  # noqa: E402

# The files refresh_skill.py snapshots into both skill copies: the MCP control
# plane *and* the serving payload. Spelled out here rather than imported from
# refresh_skill so that adding a file shows up as a test edit rather than passing
# vacuously against whatever the module happens to hold.
SKILL_SOURCES = ("server.py", "project-setup.sh", "requirements.txt")


def run(coro):
    return asyncio.run(coro)


class ToolCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = {tool.name: tool for tool in run(server.mcp.list_tools())}

    def test_catalog(self):
        expected = {
            "create_trn1_instance", "list_trn1_instances", "start_trn1_instance",
            "stop_trn1_instance", "terminate_trn1_instance", "verify_neuron_health",
            "get_server_logs", "get_endpoint", "query_model",
            "check_trn1_quotas", "get_deployment_config", "get_help", "save_hf_token",
        }
        self.assertEqual(set(self.tools), expected)

    def test_annotations(self):
        destructive = {
            name for name, tool in self.tools.items()
            if tool.annotations.destructive_hint
        }
        self.assertEqual(
            destructive, {"stop_trn1_instance", "terminate_trn1_instance"}
        )
        for name, tool in self.tools.items():
            self.assertTrue(tool.title, name)
            self.assertTrue(tool.description, name)
            self.assertIsNotNone(tool.annotations, name)

    def test_log_tail_is_bounded(self):
        tail = self.tools["get_server_logs"].input_schema["properties"]["tail"]
        self.assertEqual(tail["minimum"], 1)
        self.assertEqual(tail["maximum"], 5000)

    def test_query_cannot_ask_past_max_model_len(self):
        limit = self.tools["query_model"].input_schema["properties"]["max_tokens"]
        self.assertEqual(limit["maximum"], server.MAX_MODEL_LEN - 1)


class Trn1HelpersTests(unittest.TestCase):
    def test_supported_instance_topology(self):
        expected = {
            "trn1.2xlarge": (1, 2),
            "trn1.32xlarge": (16, 32),
            "trn1n.32xlarge": (16, 32),
        }
        for instance_type, (devices, cores) in expected.items():
            self.assertEqual(server._neuron_devices(instance_type), devices)
            self.assertEqual(server._neuron_cores(instance_type), cores)
        for bad in ("g6.xlarge", "inf2.xlarge", "trn2.48xlarge", "trn1.unknown"):
            with self.assertRaises(ValueError):
                server._validate_instance_type(bad)

    def test_user_data_serves_vllm_without_embedding_the_token(self):
        text = server._user_data("trn1.2xlarge")
        self.assertIn(server.VLLM_IMAGE, text)
        self.assertIn(f"vllm serve {server.MODEL_NAME}", text)
        self.assertIn(f"--tensor-parallel-size {server.TENSOR_PARALLEL_SIZE}", text)
        self.assertIn("--device=/dev/neuron0", text)
        self.assertIn(f"-p {server.SERVE_PORT}:8000", text)
        self.assertIn(f"--secret-id {server.HF_SECRET_ID}", text)
        self.assertNotIn("hf_", text)
        proc = subprocess.run(
            ["bash", "-n", "/dev/stdin"], input=text, text=True, capture_output=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_big_instance_exposes_every_device_but_keeps_tp(self):
        text = server._user_data("trn1.32xlarge")
        self.assertIn("--device=/dev/neuron15", text)
        self.assertIn(f"--tensor-parallel-size {server.TENSOR_PARALLEL_SIZE}", text)

    def test_tp_larger_than_the_instance_is_refused(self):
        saved = server.TENSOR_PARALLEL_SIZE
        try:
            server.TENSOR_PARALLEL_SIZE = 4
            with self.assertRaises(ValueError):
                server._user_data("trn1.2xlarge")
        finally:
            server.TENSOR_PARALLEL_SIZE = saved

    def test_launch_defaults_to_spot(self):
        tools = {tool.name: tool for tool in run(server.mcp.list_tools())}
        for name in ("create_trn1_instance", "get_deployment_config"):
            schema = tools[name].input_schema["properties"]
            self.assertTrue(schema["spot"]["default"], name)
            self.assertIn("model", schema, name)

    def test_deployment_config_is_offline_decodable_and_tagged(self):
        result = run(server.get_deployment_config())
        self.assertIn("aws ec2 run-instances", result)
        self.assertIn("MarketType=spot", result)
        self.assertIn(f"Key=ManagedBy,Value={server.RIG_NAME}", result)
        self.assertIn("HttpTokens=required", result)
        encoded = result.split("--user-data '", 1)[1].split("'", 1)[0]
        script = base64.b64decode(encoded).decode()
        self.assertIn("#!/usr/bin/env bash", script)
        self.assertIn(server.VLLM_IMAGE, script)
        self.assertNotIn("MarketType=spot", run(server.get_deployment_config(spot=False)))

    def test_deployment_config_rejects_non_trn1(self):
        for bad in ("g6.xlarge", "inf2.xlarge"):
            self.assertTrue(run(server.get_deployment_config(instance_type=bad)).startswith("❌"))

    def test_discovery_is_scoped_to_this_rig(self):
        self.assertEqual(server.MANAGED_BY, "tpu-vllm-trn1-2b")
        self.assertEqual(server.RIG_NAME, "tpu-vllm-trn1-2b")


class RepoHygieneTests(unittest.TestCase):
    def test_shell_scripts_parse(self):
        for script in ("project-setup.sh", "init.sh", "set_env.sh", "set_adc.sh"):
            proc = subprocess.run(
                ["bash", "-n", str(ROOT / script)], capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 0, f"{script}: {proc.stderr}")

    def test_skill_snapshots_match_sources(self):
        for prefix in (".claude/skills/tpu-vllm-trn1-2b-management", "skills/tpu-vllm-trn1-2b-management"):
            for source in SKILL_SOURCES:
                self.assertTrue(
                    filecmp.cmp(
                        ROOT / source, ROOT / prefix / "mcp" / source, shallow=False
                    ),
                    f"{prefix}/mcp/{source} is stale",
                )


if __name__ == "__main__":
    unittest.main()
