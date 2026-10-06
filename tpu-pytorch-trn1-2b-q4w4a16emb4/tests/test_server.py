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
import torch_generate  # noqa: E402
import torch_openai_server  # noqa: E402

# The files refresh_skill.py snapshots into both skill copies: the MCP control
# plane *and* the serving payload. Spelled out here rather than imported from
# refresh_skill so that adding a file shows up as a test edit rather than passing
# vacuously against whatever the module happens to hold.
SKILL_SOURCES = (
    "server.py", "project-setup.sh", "requirements.txt",
    "requirements-serving.txt", "torch_generate.py", "torch_openai_server.py",
    "ct_load.py", "bench_arm.py",
)


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
            "check_trn1_quotas", "get_deployment_config", "get_help",
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

    def test_query_cannot_ask_past_the_traced_graph(self):
        limit = self.tools["query_model"].input_schema["properties"]["max_tokens"]
        self.assertEqual(limit["maximum"], 511)


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

    def test_user_data_is_selfcontained_single_device(self):
        text = server._user_data("trn1.2xlarge")
        self.assertIn(server.OPTB_IMAGE, text)
        self.assertIn("--device=/dev/neuron0", text)
        self.assertNotIn("--device=/dev/neuron1", text)
        self.assertIn(f"-p {server.SERVE_PORT}:8080", text)
        self.assertNotIn("secretsmanager", text)
        self.assertNotIn("vllm serve", text)
        proc = subprocess.run(
            ["bash", "-n", "/dev/stdin"], input=text, text=True, capture_output=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_other_images_take_the_same_run_line(self):
        text = server._user_data("trn1.2xlarge", "docker.io/xbill9/gemma4-optb-26b:xlarge")
        self.assertIn("xbill9/gemma4-optb-26b:xlarge", text)
        self.assertIn("-v gemma4-data:/data", text)

    def test_launch_defaults_to_spot(self):
        tools = {tool.name: tool for tool in run(server.mcp.list_tools())}
        for name in ("create_trn1_instance", "get_deployment_config"):
            schema = tools[name].input_schema["properties"]
            self.assertTrue(schema["spot"]["default"], name)
            self.assertNotIn("serving", schema, name)

    def test_deployment_config_is_offline_decodable_and_tagged(self):
        result = run(server.get_deployment_config())
        self.assertIn("aws ec2 run-instances", result)
        self.assertIn("MarketType=spot", result)
        self.assertIn(f"Key=ManagedBy,Value={server.RIG_NAME}", result)
        self.assertIn("HttpTokens=required", result)
        encoded = result.split("--user-data '", 1)[1].split("'", 1)[0]
        script = base64.b64decode(encoded).decode()
        self.assertIn("#!/usr/bin/env bash", script)
        self.assertIn(server.OPTB_IMAGE, script)
        self.assertNotIn("MarketType=spot", run(server.get_deployment_config(spot=False)))

    def test_deployment_config_rejects_non_trn1(self):
        for bad in ("g6.xlarge", "inf2.xlarge"):
            self.assertTrue(run(server.get_deployment_config(instance_type=bad)).startswith("❌"))

    def test_discovery_is_scoped_to_this_rig(self):
        self.assertEqual(server.MANAGED_BY, "tpu-pytorch-trn1-2b-q4w4a16emb4")
        self.assertEqual(server.RIG_NAME, "tpu-pytorch-trn1-2b-q4w4a16emb4")


class NeuronEngineTests(unittest.TestCase):
    """Offline checks on the engine's static-shape contract.

    Nothing here loads a checkpoint or imports torch: torch_generate defers both,
    so the parts that decide graph geometry are testable on a laptop. The parts
    that need a device are not covered here and cannot be -- `torch_generate.py
    --parity` is what checks those, on the instance.
    """

    def test_prompt_bucket_must_leave_room_to_decode(self):
        with self.assertRaises(ValueError):
            torch_generate.NeuronGemmaEngine(max_total=32, prompt_bucket=32)

    def test_device_is_restricted(self):
        with self.assertRaises(ValueError):
            torch_generate.NeuronGemmaEngine(device="cuda")

    def test_neff_filename_encodes_every_traced_dimension(self):
        """A graph traced at one geometry cannot run at another.

        The filename carries batch, max_total and prompt_bucket so a mismatched
        cache MISSES rather than loading and failing on shape at the first
        request.
        """
        a = torch_generate.NeuronGemmaEngine(batch=1, max_total=128, prompt_bucket=32,
                                             neff_dir="/n")._neff_paths()
        for other in (
            torch_generate.NeuronGemmaEngine(batch=8, max_total=128, prompt_bucket=32,
                                             neff_dir="/n"),
            torch_generate.NeuronGemmaEngine(batch=1, max_total=256, prompt_bucket=32,
                                             neff_dir="/n"),
            torch_generate.NeuronGemmaEngine(batch=1, max_total=128, prompt_bucket=64,
                                             neff_dir="/n"),
        ):
            self.assertNotEqual(a, other._neff_paths())

    def test_stream_cannot_reach_the_park_row(self):
        """Idle slots write KV at `park`; a live stream must stay below it.

        Otherwise a parked slot's write lands on a decoding stream's cache row
        and corrupts it, which shows up as wrong text rather than an error.
        """
        park = 127
        stream = torch_openai_server.Stream(
            prompt_ids=list(range(20)), max_new=10_000, temperature=0.0, top_k=0,
            top_p=1.0, stop_ids=set(), timeout_s=None, ceiling=park,
        )
        highest_position = stream.n0 + stream.max_new - 1
        self.assertLess(highest_position, park)

    def test_stream_always_gets_at_least_one_token(self):
        stream = torch_openai_server.Stream(
            prompt_ids=list(range(126)), max_new=64, temperature=0.0, top_k=0,
            top_p=1.0, stop_ids=set(), timeout_s=None, ceiling=127,
        )
        self.assertGreaterEqual(stream.max_new, 1)


class RepoHygieneTests(unittest.TestCase):
    def test_shell_scripts_parse(self):
        for script in ("project-setup.sh", "init.sh", "set_env.sh", "set_adc.sh"):
            proc = subprocess.run(
                ["bash", "-n", str(ROOT / script)], capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 0, f"{script}: {proc.stderr}")

    def test_skill_snapshots_match_sources(self):
        for prefix in (".claude/skills/tpu-pytorch-trn1-2b-q4w4a16emb4-management", "skills/tpu-pytorch-trn1-2b-q4w4a16emb4-management"):
            for source in SKILL_SOURCES:
                self.assertTrue(
                    filecmp.cmp(
                        ROOT / source, ROOT / prefix / "mcp" / source, shallow=False
                    ),
                    f"{prefix}/mcp/{source} is stale",
                )


if __name__ == "__main__":
    unittest.main()
