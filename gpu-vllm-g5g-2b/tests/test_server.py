"""Offline regression tests for the G5g MCP server.

No AWS, no network, no GPU. These pin the facts that make this rig different
from its L4 siblings — the Turing dtype constraints, the arm64 AMI filter, and
the host-RAM floor — because every one of them is a silent copy-paste hazard
from a sibling rig that runs on different silicon.
"""

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
            "create_g5g_instance", "list_g5g_instances", "start_g5g_instance",
            "stop_g5g_instance", "terminate_g5g_instance", "verify_gpu_arch",
            "get_build_progress", "get_vllm_logs", "get_endpoint",
            "verify_model_health", "query_model", "save_hf_token",
            "check_g5g_quotas", "get_deployment_config", "get_help",
        }
        self.assertEqual(set(self.tools), expected)

    def test_annotations(self):
        destructive = {
            name for name, tool in self.tools.items() if tool.annotations.destructiveHint
        }
        self.assertEqual(destructive, {"stop_g5g_instance", "terminate_g5g_instance"})
        for name, tool in self.tools.items():
            self.assertTrue(tool.title, name)
            self.assertTrue(tool.description, name)
            self.assertIsNotNone(tool.annotations, name)

    def test_launch_defaults_to_spot_and_build(self):
        for name in ("create_g5g_instance", "get_deployment_config"):
            schema = self.tools[name].inputSchema["properties"]
            self.assertTrue(schema["spot"]["default"], name)
            self.assertEqual(schema["serving"]["enum"], ["build", "stock"], name)
            self.assertEqual(schema["serving"]["default"], "build", name)


class G5gTopologyTests(unittest.TestCase):
    def test_sizes_match_the_aws_product_page(self):
        expected = {
            "g5g.xlarge": (1, 8),
            "g5g.2xlarge": (1, 16),
            "g5g.4xlarge": (1, 32),
            "g5g.8xlarge": (1, 64),
            "g5g.16xlarge": (2, 128),
            "g5g.metal": (2, 128),
        }
        for instance_type, (gpus, ram) in expected.items():
            self.assertTrue(server._is_g5g(instance_type))
            self.assertEqual(server._gpu_count(instance_type), gpus)
            self.assertEqual(server._host_memory_gb(instance_type), ram)

    def test_tensor_parallel_follows_gpu_count(self):
        self.assertEqual(server._tensor_parallel_size("g5g.2xlarge"), 1)
        self.assertEqual(server._tensor_parallel_size("g5g.16xlarge"), 2)

    def test_xlarge_is_supported_with_swap(self):
        # Measured 2026-08-13: g5g.xlarge serves E2B at 44.24 tok/s *with* a
        # swapfile. Without swap the kernel refuses to mmap the 10.2 GB
        # checkpoint ("Cannot allocate memory") and systemd crash-loops. So the
        # small size is supported, not rejected -- but its user data must carry
        # the swapfile.
        server._validate_instance_type("g5g.xlarge")   # must not raise
        self.assertTrue(server._needs_swap("g5g.xlarge"))
        text = server._user_data("google/gemma-4-E2B-it", "g5g.xlarge", serving="build")
        self.assertIn("mkswap", text)
        self.assertIn("swapon /swapfile", text)
        self.assertIn("/etc/fstab", text)

    def test_larger_sizes_get_no_swapfile(self):
        for size in ("g5g.2xlarge", "g5g.4xlarge", "g5g.16xlarge"):
            self.assertFalse(server._needs_swap(size), size)
            text = server._user_data("google/gemma-4-E2B-it", size, serving="build")
            self.assertNotIn("mkswap", text)

    def test_non_g5g_rejected(self):
        for bad in ("g6.xlarge", "inf2.xlarge", "g5g.unknown"):
            with self.assertRaises(ValueError):
                server._validate_instance_type(bad)


class TuringConstraintTests(unittest.TestCase):
    """Turing (SM 7.5) has no bf16 and no fp8. The L4 sibling rigs hardcode both."""

    def test_dtype_is_float16_not_bfloat16(self):
        flags = server._serve_flags("google/gemma-4-E2B-it", "g5g.2xlarge")
        self.assertIn("--dtype float16", flags)
        self.assertNotIn("bfloat16", flags)

    def test_kv_cache_is_not_fp8(self):
        flags = server._serve_flags("google/gemma-4-E2B-it", "g5g.2xlarge")
        self.assertIn("--kv-cache-dtype auto", flags)
        self.assertNotIn("fp8", flags)

    def test_build_arch_list_is_sm75(self):
        self.assertEqual(server.TORCH_CUDA_ARCH_LIST, "7.5")

    def test_stock_and_built_images_are_distinct(self):
        # If these ever collapse to one value the build path silently becomes a
        # no-op and the rig serves an image with no SM 7.5 kernels.
        self.assertNotEqual(server.VLLM_IMAGE, server.VLLM_STOCK_IMAGE)


class UserDataTests(unittest.TestCase):
    def test_build_mode_passes_arch_list_and_is_detached(self):
        text = server._user_data("google/gemma-4-E2B-it", "g5g.2xlarge", serving="build")
        self.assertIn("torch_cuda_arch_list='7.5'", text)
        self.assertIn("--platform linux/arm64", text)
        self.assertIn(server.VLLM_IMAGE, text)
        # cloud-init must not block for the hours the build takes.
        self.assertIn("nohup", text)
        self.assertIn("BUILD_DONE", text)
        self.assertShellParses(text)

    def test_stock_mode_uses_published_image(self):
        text = server._user_data("google/gemma-4-E2B-it", "g5g.2xlarge", serving="stock")
        self.assertIn(server.VLLM_STOCK_IMAGE, text)
        self.assertNotIn("docker build", text)
        self.assertShellParses(text)

    def test_token_comes_from_secrets_manager_not_user_data(self):
        # User data is readable from instance metadata by anything on the box.
        for serving in ("build", "stock"):
            text = server._user_data("google/gemma-4-E2B-it", "g5g.2xlarge", serving)
            self.assertIn("secretsmanager get-secret-value", text)
            self.assertNotIn("hf_", text.lower().replace("hf_token", ""))

    def test_user_data_rejects_bad_serving_mode(self):
        with self.assertRaises(ValueError):
            server._user_data("m", "g5g.2xlarge", serving="bogus")

    def test_multi_gpu_size_gets_tp2(self):
        text = server._user_data("google/gemma-4-E2B-it", "g5g.16xlarge", serving="build")
        self.assertIn("--tensor-parallel-size 2", text)

    def assertShellParses(self, text):
        proc = subprocess.run(
            ["bash", "-n", "/dev/stdin"], input=text, text=True, capture_output=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


class DeploymentConfigTests(unittest.TestCase):
    def test_config_is_offline_and_decodable(self):
        result = run(server.get_deployment_config(instance_type="g5g.2xlarge"))
        self.assertIn("aws ec2 run-instances", result)
        encoded = result.split("--user-data '", 1)[1].split("'", 1)[0]
        script = base64.b64decode(encoded).decode()
        self.assertIn("#!/usr/bin/env bash", script)

    def test_config_resolves_ami_via_ssm_parameter(self):
        # Two independent requirements, and a name filter only pins one of them.
        # arm64: the legacy tips-tree rigs hardcode an x86_64 DLAMI id that
        # cannot boot on Graviton2. NVIDIA driver: AWS also ships ARM64 DLAMIs
        # for Graviton CPU inference, which boot fine on a G5g with no GPU.
        result = run(server.get_deployment_config())
        self.assertIn("aws ssm get-parameter", result)
        self.assertIn("/arm64/", server.DLAMI_SSM_PARAMETER)
        self.assertIn("nvidia-driver-gpu", server.DLAMI_SSM_PARAMETER)

    def test_ami_name_fallback_requires_the_driver(self):
        # A bare "Deep Learning*ARM64*Ubuntu*" also matches the driverless
        # images, and _resolve_ami takes newest-by-CreationDate.
        self.assertIn("ARM64", server.DLAMI_NAME)
        self.assertIn("Nvidia Driver", server.DLAMI_NAME)

    def test_config_tags_with_rig_name(self):
        result = run(server.get_deployment_config())
        self.assertIn(f"Key=ManagedBy,Value={server.RIG_NAME}", result)

    def test_config_spot_toggle(self):
        self.assertIn("MarketType=spot", run(server.get_deployment_config()))
        self.assertNotIn("MarketType=spot", run(server.get_deployment_config(spot=False)))

    def test_config_rejects_non_g5g_only(self):
        # g5g.xlarge is supported now (it gets a swapfile), so only genuinely
        # wrong instance families should be refused.
        self.assertTrue(run(server.get_deployment_config(instance_type="g6.xlarge")).startswith("❌"))
        ok = run(server.get_deployment_config(instance_type="g5g.xlarge"))
        self.assertFalse(ok.startswith("❌"))
        encoded = ok.split("--user-data '", 1)[1].split("'", 1)[0]
        self.assertIn("mkswap", base64.b64decode(encoded).decode())


class RepoHygieneTests(unittest.TestCase):
    def test_shell_scripts_parse(self):
        for script in ("project-setup.sh", "init.sh", "set_env.sh"):
            proc = subprocess.run(
                ["bash", "-n", str(ROOT / script)], capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 0, f"{script}: {proc.stderr}")

    def test_tpu_env_agrees_with_server_defaults(self):
        # The directory name is a claim about tpu.env (NAMING.md). This asserts
        # the env file and the server actually agree, so the claim stays true.
        values = {}
        for line in (ROOT / "tpu.env").read_text().splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key] = value
        self.assertEqual(values["MODEL_NAME"], server.MODEL_NAME)
        self.assertEqual(values["INSTANCE_TYPE"], server.INSTANCE_TYPE)
        self.assertEqual(values["DTYPE"], server.DTYPE)
        self.assertEqual(values["KV_CACHE_DTYPE"], server.KV_CACHE_DTYPE)
        self.assertEqual(values["TORCH_CUDA_ARCH_LIST"], server.TORCH_CUDA_ARCH_LIST)
        self.assertEqual(int(values["TENSOR_PARALLEL_SIZE"]), server._gpu_count(server.INSTANCE_TYPE))

    def test_rig_name_matches_directory(self):
        self.assertEqual(server.RIG_NAME, ROOT.name)

    def test_skill_is_complete_in_both_copies(self):
        # SKILL.md is a hand-written SOURCE file, but refresh_skill.py only
        # regenerates the mcp/ files beside it. So `rm -rf .claude/skills`
        # destroys it and `make skill` does not bring it back — which is exactly
        # what happened during the t4g->g5g rename. Guard both copies.
        stem = f"{ROOT.name}-management"
        for prefix in (f".claude/skills/{stem}", f"skills/{stem}"):
            skill = ROOT / prefix / "SKILL.md"
            self.assertTrue(skill.is_file(), f"{prefix}/SKILL.md is missing")
            text = skill.read_text()
            self.assertIn(f"name: {stem}", text, f"{prefix}/SKILL.md has a stale name")
            for source in ("server.py", "project-setup.sh", "requirements.txt"):
                self.assertTrue(
                    filecmp.cmp(ROOT / source, ROOT / prefix / "mcp" / source, shallow=False),
                    f"{prefix}/mcp/{source} is stale — run `make skill`",
                )


if __name__ == "__main__":
    unittest.main()
