"""Smoke and regression tests for the tpu-pytorch-v5e1-12b MCP server and repo invariants.

Standard library + the server's own dependencies only; run via `make test` or
`python3 -m unittest discover -s tests`. No GCP calls are made — tests cover
pure logic, tool registration, template rendering, and repo hygiene.
"""

import asyncio
import filecmp
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import server  # noqa: E402

EXPECTED_DESTRUCTIVE = {
    "destroy_queued_resource",
    "destroy_tpu_vm_instance",
    "manage_queued_resource",
    "manage_vllm_docker",
    "find_tpu",
}


def run(coro):
    return asyncio.run(coro)


class ToolCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = {t.name: t for t in run(server.mcp.list_tools())}

    def test_every_tool_has_title_description_and_annotations(self):
        for name, tool in self.tools.items():
            self.assertTrue(tool.title, f"{name} has no title")
            self.assertTrue(tool.description, f"{name} has no description")
            self.assertIsNotNone(tool.annotations, f"{name} has no annotations")

    def test_destructive_hints_match_expected_set(self):
        destructive = {
            name for name, t in self.tools.items() if t.annotations.destructiveHint
        }
        self.assertEqual(destructive, EXPECTED_DESTRUCTIVE)

    def test_read_only_tools_never_marked_destructive(self):
        for name, t in self.tools.items():
            if t.annotations.readOnlyHint:
                self.assertFalse(
                    t.annotations.destructiveHint,
                    f"{name} is both readOnly and destructive",
                )

    def test_action_and_type_enums_in_schema(self):
        props = self.tools["manage_vllm_docker"].inputSchema["properties"]
        self.assertEqual(
            props["action"]["enum"], ["start", "stop", "restart", "status", "log", "rm"]
        )
        self.assertEqual(
            self.tools["estimate_deployment_cost"].inputSchema["properties"]["tpu_type"]["enum"],
            ["v6e", "v5e", "v5p"],
        )
        for name in ("create_tpu_vm_instance", "find_tpu_vm"):
            workload = self.tools[name].inputSchema["properties"]["workload"]
            self.assertEqual(workload["enum"], ["vllm", "pytorch"], name)

    def test_log_tails_are_bounded(self):
        for name in ("get_vllm_docker_logs", "get_tpu_system_logs", "get_tpu_vm_serial_log"):
            tail = self.tools[name].inputSchema["properties"]["tail"]
            self.assertIn("maximum", tail, f"{name}.tail has no upper bound")


class HelperTests(unittest.TestCase):
    def test_zone_defaults_to_current_global(self):
        self.assertEqual(server._zone(None), server.ZONE)
        self.assertEqual(server._zone("us-east5-b"), "us-east5-b")

    def test_filter_key_metrics_drops_comments_and_noise(self):
        text = (
            "# HELP vllm_requests_running Running requests\n"
            "vllm_requests_running 3.0\n"
            "vllm_request_latency_bucket{le=\"0.5\"} 12\n"
            "process_resident_memory_bytes 1024\n"
        )
        self.assertEqual(
            server._filter_key_metrics(text),
            ["vllm_requests_running 3.0", "process_resident_memory_bytes 1024"],
        )

    def test_estimate_cost_math_and_rejects_bad_topology(self):
        result = run(server.estimate_deployment_cost(hours=2.0, tpu_type="v6e", topology="2x4"))
        self.assertIn("$21.60", result)  # 8 chips * 1.35 * 2h
        self.assertTrue(run(server.estimate_deployment_cost(topology="0x4")).startswith("❌"))

    def test_min_chips_for_model(self):
        self.assertEqual(server._min_chips_for_model("google/gemma-4-31B-it"), 4)
        self.assertEqual(server._min_chips_for_model("google/gemma-4-12B-it"), 1)
        self.assertEqual(server._min_chips_for_model("google/gemma-4-E2B-it"), 1)
        # Quantized checkpoints fit on a single chip regardless of size
        self.assertEqual(server._min_chips_for_model("google/gemma-4-26B-it-qat-q4_0"), 1)
        self.assertEqual(server._min_chips_for_model("google/gemma-4-31B-it-int4"), 1)

    def test_create_blocks_model_too_big_for_accelerator(self):
        result = run(
            server.create_tpu_vm_instance(
                accelerator="v6e-1", model_name="google/gemma-4-31B-it"
            )
        )
        self.assertTrue(result.startswith("❌"))
        self.assertIn("chips", result)

    def test_served_model_id_falls_back_to_none_when_unreachable(self):
        self.assertIsNone(run(server._get_served_model_id("http://127.0.0.1:9")))

    def test_sweep_point_from_bench_result(self):
        result = {
            "model_id": "google/gemma-4-E2B-it",
            "request_throughput": 9.4429,
            "output_throughput": 1208.71,
            "total_token_throughput": 10877.6,
            "median_ttft_ms": 27.1,
            "p99_ttft_ms": 99.4,
            "median_tpot_ms": 6.2,
            "ttfts": [0.027] * 100,
        }
        point = server._sweep_point_from_bench_result(result, 8)
        self.assertEqual(point["concurrency"], 8)
        self.assertEqual(point["request_rate_rps"], 9.44)
        self.assertEqual(point["ttft_ms"], {"median": 27.1, "p99": 99.4})
        self.assertEqual(point["per_stream_tok_per_s"], 161.3)  # 1000 / 6.2
        self.assertNotIn("itl_ms", point)  # absent metrics stay absent
        self.assertNotIn("ttfts", point["raw"])  # per-request arrays dropped

    def test_run_command_success_and_timeout(self):
        rc, out, _ = run(server.run_command(["echo", "hi"]))
        self.assertEqual((rc, out), (0, "hi"))
        rc, _, err = run(server.run_command(["sleep", "5"], timeout=1))
        self.assertEqual(rc, -1)
        self.assertIn("Timeout", err)


class StartupTemplateTests(unittest.TestCase):
    def render(self):
        template = (
            ROOT / ".claude/skills/tpu-pytorch-v5e1-12b-management/mcp/startup_script_template.sh"
        ).read_text()
        return template.format(
            project_id="test-project",
            zone="europe-west4-a",
            model_name="google/gemma-4-31B-it",
            hf_secret_id="hf-token",
            tp_size=8,
            limit_mm_per_prompt_env="export VLLM_LIMIT_MM_PER_PROMPT='{\"image\":4,\"audio\":1}'",
        )

    def test_renders_without_leftover_placeholders_or_token(self):
        rendered = self.render()
        self.assertNotIn("{hf_token}", rendered)
        self.assertIn("secretmanager.googleapis.com", rendered)
        self.assertIn("test-project", rendered)

    def test_rendered_script_passes_bash_syntax_check(self):
        rendered = self.render()
        proc = subprocess.run(
            ["bash", "-n", "/dev/stdin"], input=rendered, text=True, capture_output=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


class PyTorchStartupTemplateTests(unittest.TestCase):
    def render(self):
        template = (
            ROOT / ".claude/skills/tpu-pytorch-v5e1-12b-management/mcp/startup_script_pytorch_template.sh"
        ).read_text()
        return template.format(
            project_id="test-project",
            zone="europe-west4-a",
            torch_pip_spec="tpu-backend-pkg",
            torch_index="us-python.pkg.dev/test-registry/simple/",
            wheels_gcs="gs://test-project-wheels",
            torch_pip_extras="transformers",
            env_exports="FOO=bar BAZ=qux",
        )

    def test_renders_without_leftover_placeholders(self):
        rendered = self.render()
        self.assertNotIn("{project_id}", rendered)
        self.assertNotIn("{torch_pip_spec}", rendered)
        self.assertNotIn("{torch_index}", rendered)
        self.assertIn("tpu-backend-pkg", rendered)
        self.assertIn("us-python.pkg.dev/test-registry/simple/", rendered)
        self.assertIn("TPU environment ready.", rendered)
        self.assertIn("FOO=bar", rendered)

    def test_smoke_test_follows_tpu_rules(self):
        """The embedded smoke test must use plain torch (never importing the backend
        or PyTorch/XLA) and compile with backend="tpu"."""
        rendered = self.render()
        smoke = rendered.split("<< 'PYEOF'")[1].split("PYEOF")[0]
        # Plain torch only: the smoke test must import nothing but torch — importing
        # the backend package directly (or PyTorch/XLA) is the failure this guards.
        imports = {
            line.split()[1].split(".")[0]
            for line in smoke.splitlines()
            if line.startswith(("import ", "from "))
        }
        self.assertEqual(imports, {"torch"}, f"unexpected imports in smoke test: {imports}")
        self.assertNotIn("torch_xla", smoke)
        self.assertIn('backend="tpu"', smoke)
        self.assertIn("bfloat16", smoke)

    def test_rendered_script_passes_bash_syntax_check(self):
        rendered = self.render()
        proc = subprocess.run(
            ["bash", "-n", "/dev/stdin"], input=rendered, text=True, capture_output=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


class RepoHygieneTests(unittest.TestCase):
    def test_shell_scripts_parse(self):
        for script in ("project-setup.sh", "init.sh", "set_env.sh", "set_adc.sh"):
            proc = subprocess.run(
                ["bash", "-n", str(ROOT / script)], capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 0, f"{script}: {proc.stderr}")

    def test_skill_snapshots_in_sync_with_sources(self):
        """Sources at the repo root are authoritative; `make skill` regenerates the
        copies. A mismatch means someone edited one side without resyncing."""
        for src, snap in (
            ("server.py", ".claude/skills/tpu-pytorch-v5e1-12b-management/mcp/server.py"),
            ("project-setup.sh", ".claude/skills/tpu-pytorch-v5e1-12b-management/mcp/project-setup.sh"),
            ("requirements.txt", ".claude/skills/tpu-pytorch-v5e1-12b-management/mcp/requirements.txt"),
        ):
            self.assertTrue(
                filecmp.cmp(ROOT / src, ROOT / snap, shallow=False),
                f"{snap} is stale — run `make skill`",
            )

    def test_plugin_copy_matches_skill(self):
        for rel in (
            "mcp/server.py",
            "SKILL.md",
            "mcp/startup_script_template.sh",
            "mcp/startup_script_pytorch_template.sh",
        ):
            self.assertTrue(
                filecmp.cmp(
                    ROOT / ".claude/skills/tpu-pytorch-v5e1-12b-management" / rel,
                    ROOT / "skills/tpu-pytorch-v5e1-12b-management" / rel,
                    shallow=False,
                ),
                f"skills/tpu-pytorch-v5e1-12b-management/{rel} is stale — run `make skill`",
            )


if __name__ == "__main__":
    unittest.main()
