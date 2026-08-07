"""Offline regression tests for the Inf2 MCP server."""

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


def run(coro):
    return asyncio.run(coro)


class ToolCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = {tool.name: tool for tool in run(server.mcp.list_tools())}

    def test_catalog(self):
        expected = {
            "create_inf2_instance", "list_inf2_instances", "start_inf2_instance",
            "stop_inf2_instance", "terminate_inf2_instance", "verify_neuron_health",
            "get_vllm_logs", "get_endpoint", "query_model", "save_hf_token",
            "check_inf2_quotas", "get_deployment_config", "get_help",
        }
        self.assertEqual(set(self.tools), expected)

    def test_annotations(self):
        destructive = {
            name for name, tool in self.tools.items()
            if tool.annotations.destructiveHint
        }
        self.assertEqual(
            destructive, {"stop_inf2_instance", "terminate_inf2_instance"}
        )
        for name, tool in self.tools.items():
            self.assertTrue(tool.title, name)
            self.assertTrue(tool.description, name)
            self.assertIsNotNone(tool.annotations, name)

    def test_log_tail_is_bounded(self):
        tail = self.tools["get_vllm_logs"].inputSchema["properties"]["tail"]
        self.assertEqual(tail["minimum"], 1)
        self.assertEqual(tail["maximum"], 5000)


class Inf2HelpersTests(unittest.TestCase):
    def test_supported_instance_topology(self):
        expected = {
            "inf2.xlarge": (1, 2),
            "inf2.8xlarge": (1, 2),
            "inf2.24xlarge": (6, 12),
            "inf2.48xlarge": (12, 24),
        }
        for instance_type, (devices, cores) in expected.items():
            self.assertTrue(server._is_inf2(instance_type))
            self.assertEqual(server._neuron_devices(instance_type), devices)
            self.assertEqual(server._neuron_cores(instance_type), cores)
        with self.assertRaises(ValueError):
            server._validate_instance_type("g6.xlarge")
        with self.assertRaises(ValueError):
            server._validate_instance_type("inf2.unknown")

    def test_user_data_uses_neuron_devices_and_no_token(self):
        text = server._user_data("meta-llama/test", "inf2.24xlarge")
        for index in range(6):
            self.assertIn(f"--device=/dev/neuron{index}", text)
        self.assertIn("--tensor-parallel-size 12", text)
        self.assertIn("secretsmanager get-secret-value", text)
        self.assertNotIn("hf_", text.lower().replace("hf_token", ""))
        proc = subprocess.run(
            ["bash", "-n", "/dev/stdin"], input=text, text=True, capture_output=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_optb_user_data_is_selfcontained_single_device(self):
        text = server._user_data("ignored", "inf2.xlarge", serving="optb")
        self.assertIn(server.OPTB_IMAGE, text)
        self.assertIn("--device=/dev/neuron0", text)
        self.assertIn("swapon /swapfile", text)
        self.assertNotIn("secretsmanager", text)
        self.assertNotIn("vllm serve", text)
        proc = subprocess.run(
            ["bash", "-n", "/dev/stdin"], input=text, text=True, capture_output=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with self.assertRaises(ValueError):
            server._user_data("ignored", "inf2.24xlarge", serving="optb")
        with self.assertRaises(ValueError):
            server._user_data("ignored", "inf2.xlarge", serving="bogus")

    def test_launch_defaults_to_spot(self):
        tools = {tool.name: tool for tool in run(server.mcp.list_tools())}
        for name in ("create_inf2_instance", "get_deployment_config"):
            schema = tools[name].inputSchema["properties"]
            self.assertTrue(schema["spot"]["default"], name)
            self.assertEqual(schema["serving"]["enum"], ["vllm", "optb"], name)

    def test_deployment_config_spot_and_optb(self):
        result = run(
            server.get_deployment_config(instance_type="inf2.xlarge", serving="optb")
        )
        self.assertIn("MarketType=spot", result)
        encoded = result.split("--user-data '", 1)[1].split("'", 1)[0]
        self.assertIn(server.OPTB_IMAGE, base64.b64decode(encoded).decode())
        ondemand = run(server.get_deployment_config(spot=False))
        self.assertNotIn("MarketType=spot", ondemand)

    def test_deployment_config_is_offline_and_decodable(self):
        result = run(server.get_deployment_config(instance_type="inf2.xlarge"))
        self.assertIn("aws ec2 run-instances", result)
        marker = "--user-data '"
        encoded = result.split(marker, 1)[1].split("'", 1)[0]
        script = base64.b64decode(encoded).decode()
        self.assertIn("#!/usr/bin/env bash", script)

    def test_deployment_config_rejects_non_inf2(self):
        result = run(server.get_deployment_config(instance_type="g6.xlarge"))
        self.assertTrue(result.startswith("❌"))


class RepoHygieneTests(unittest.TestCase):
    def test_shell_scripts_parse(self):
        for script in ("project-setup.sh", "init.sh", "set_env.sh", "set_adc.sh"):
            proc = subprocess.run(
                ["bash", "-n", str(ROOT / script)], capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 0, f"{script}: {proc.stderr}")

    def test_skill_snapshots_match_sources(self):
        for prefix in (".claude/skills/tpu-pytorch-inf2-2b-management", "skills/tpu-pytorch-inf2-2b-management"):
            for source in ("server.py", "project-setup.sh", "requirements.txt"):
                self.assertTrue(
                    filecmp.cmp(
                        ROOT / source, ROOT / prefix / "mcp" / source, shallow=False
                    ),
                    f"{prefix}/mcp/{source} is stale",
                )


if __name__ == "__main__":
    unittest.main()
