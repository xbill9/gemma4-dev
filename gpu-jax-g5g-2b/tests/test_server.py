"""Offline regression tests for the G5g JAX MCP server.

No AWS, no network, no GPU. These pin the facts that make this rig different
from its siblings — the Turing dtype constraints, the arm64 AMI filter, the
host-RAM floor, and the shared-memory ceiling that decides which kernels can
run — because every one of them is a silent copy-paste hazard from a sibling rig
that runs on different silicon.

The engine tests below import ports/gemma4 under JAX_PLATFORMS=cpu and then
override the detected platform, so they exercise the Turing branch on a machine
that has no Turing GPU. That is a test of the *policy*, not of the hardware.
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
            "deploy_jax_server", "get_install_progress", "get_jax_logs",
            "get_endpoint", "verify_model_health", "query_model", "save_hf_token",
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

    def test_launch_defaults_to_spot(self):
        for name in ("create_g5g_instance", "get_deployment_config"):
            schema = self.tools[name].inputSchema["properties"]
            self.assertTrue(schema["spot"]["default"], name)
            # There is no `serving` mode here: nothing is built, so there is no
            # stock-vs-build choice the vLLM sibling has to offer.
            self.assertNotIn("serving", schema, name)


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
        text = server._user_data("google/gemma-4-E2B-it", "g5g.xlarge")
        self.assertIn("mkswap", text)
        self.assertIn("swapon /swapfile", text)
        self.assertIn("/etc/fstab", text)

    def test_larger_sizes_get_no_swapfile(self):
        for size in ("g5g.2xlarge", "g5g.4xlarge", "g5g.16xlarge"):
            self.assertFalse(server._needs_swap(size), size)
            text = server._user_data("google/gemma-4-E2B-it", size)
            self.assertNotIn("mkswap", text)

    def test_non_g5g_rejected(self):
        for bad in ("g6.xlarge", "inf2.xlarge", "g5g.unknown"):
            with self.assertRaises(ValueError):
                server._validate_instance_type(bad)


class TuringConstraintTests(unittest.TestCase):
    """Turing (SM 7.5) has no bf16 and no fp8. The L4 sibling rigs hardcode both."""

    def test_dtype_default_is_float16(self):
        self.assertEqual(server.DTYPE, "float16")

    def test_kv_cache_is_not_fp8(self):
        argv = server._serve_argv("google/gemma-4-E2B-it", "g5g.2xlarge")
        self.assertIn("--kv-cache-dtype auto", argv)
        self.assertNotIn("fp8", argv)

    def test_quant_mode_matches_the_dense_checkpoint(self):
        # QUANT_MODE is a claim about MODEL_NAME, not about the chip. A w4a16
        # mode against a dense checkpoint loads garbage rather than failing.
        argv = server._serve_argv(server.MODEL_NAME, "g5g.2xlarge")
        self.assertIn("--quant-mode fp16", argv)
        self.assertNotIn("w4a16", server.MODEL_NAME)

    def test_no_tensor_parallel_flag_is_emitted(self):
        # The JAX engine is single-device. Emitting a TP flag would imply a
        # sharding this rig does not do.
        argv = server._serve_argv("google/gemma-4-E2B-it", "g5g.16xlarge")
        self.assertNotIn("tensor-parallel", argv)


class UserDataTests(unittest.TestCase):
    def test_install_is_wheels_not_a_build(self):
        text = server._user_data("google/gemma-4-E2B-it", "g5g.2xlarge")
        self.assertIn("jax[cuda12]", text)
        self.assertNotIn("docker build", text)
        self.assertNotIn("git clone", text)
        # cloud-init must not block on the install.
        self.assertIn("nohup", text)
        self.assertIn("INSTALL_DONE", text)
        self.assertShellParses(text)

    def test_python_312_is_installed_because_jax_requires_it(self):
        # jax >= 0.11 declares requires-python >= 3.12; Ubuntu 22.04 ships 3.10,
        # so using the DLAMI's system python would fail at pip install time.
        text = server._user_data("google/gemma-4-E2B-it", "g5g.2xlarge")
        self.assertIn("deadsnakes", text)
        self.assertIn("python3.12", text)

    def test_systemd_execstart_is_absolute(self):
        # systemd refuses a relative ExecStart, and the unit would fail to load
        # with a message that says nothing about the interpreter.
        text = server._user_data("google/gemma-4-E2B-it", "g5g.2xlarge")
        self.assertIn("ExecStart=/usr/bin/python3.12", text)

    def test_token_comes_from_secrets_manager_not_user_data(self):
        # User data is readable from instance metadata by anything on the box.
        text = server._user_data("google/gemma-4-E2B-it", "g5g.2xlarge")
        self.assertIn("secretsmanager get-secret-value", text)
        self.assertNotIn("hf_", text.lower().replace("hf_token", ""))

    def test_xtrace_is_disabled_around_the_secret_fetch(self):
        # The script runs under `set -x`, and bash traces assignments WITH their
        # values — so leaving xtrace on would print the token into
        # /var/log/cloud-init-output.log, which is the exact exposure that
        # keeping it out of user data is meant to prevent.
        text = server._user_data("google/gemma-4-E2B-it", "g5g.2xlarge")
        fetch = text.index("secretsmanager get-secret-value")
        self.assertIn("set +x", text[:fetch])
        self.assertLess(text[:fetch].rindex("set +x"), fetch)
        self.assertIn("set -x", text[fetch:])

    def test_env_file_is_locked_down_before_the_token_lands(self):
        text = server._user_data("google/gemma-4-E2B-it", "g5g.2xlarge")
        self.assertLess(text.index("chmod 600 /opt/jax-g5g/env"), text.index("HF_TOKEN="))

    def test_small_hosts_get_swap_and_large_ones_do_not(self):
        small = server._user_data("google/gemma-4-E2B-it", "g5g.xlarge")
        large = server._user_data("google/gemma-4-E2B-it", "g5g.4xlarge")
        self.assertIn("mkswap", small)
        self.assertNotIn("mkswap", large)
        self.assertShellParses(small)

    def test_serving_requirements_match_the_mirror_file(self):
        # A drifted pair is invisible until a serve fails on a missing import.
        listed = {
            line.strip()
            for line in (ROOT / "requirements-serving.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        expected = set(server._SERVING_REQUIREMENTS) | {server.JAX_PIP_SPEC}
        self.assertEqual(listed, expected)

    def assertShellParses(self, text):
        proc = subprocess.run(
            ["bash", "-n", "/dev/stdin"], input=text, text=True, capture_output=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


class PayloadTests(unittest.TestCase):
    """The serving payload ships over SSM, so its size and determinism matter."""

    def test_payload_files_exist_and_are_found(self):
        root = Path(server._payload_root())
        for rel in server._PAYLOAD_FILES:
            self.assertTrue((root / rel).is_file(), rel)

    def test_payload_is_deterministic(self):
        # Idempotent redeploys depend on this: same sources, same bytes.
        self.assertEqual(server._payload_tar_b64(), server._payload_tar_b64())

    def test_payload_fits_one_ssm_run_command(self):
        # SSM caps command document size; user data could not carry this at all
        # (16 KB). Keep a wide margin so adding a module does not silently break
        # deployment.
        size = len(server._payload_tar_b64())
        self.assertLess(size, 80_000, f"payload base64 is {size} bytes")


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
        self.assertEqual(values["QUANT_MODE"], server.QUANT_MODE)
        self.assertEqual(values["JAX_PIP_SPEC"], server.JAX_PIP_SPEC)
        self.assertEqual(values["JAX_PYTHON_VERSION"], server.JAX_PYTHON_VERSION)
        self.assertEqual(values["SERVICE_NAME"], server.SERVICE_NAME)
        self.assertEqual(int(values["JAX_PORT"]), server.JAX_PORT)
        self.assertEqual(int(values["MAX_MODEL_LEN"]), server.MAX_MODEL_LEN)
        self.assertEqual(
            values["XLA_PYTHON_CLIENT_MEM_FRACTION"], server.XLA_PYTHON_CLIENT_MEM_FRACTION
        )
        self.assertEqual(int(values["TENSOR_PARALLEL_SIZE"]), server._gpu_count(server.INSTANCE_TYPE))

    def test_no_vllm_config_survives(self):
        # This rig was forked from the vLLM one. A leftover VLLM_* key would be
        # dead config that reads as live.
        text = (ROOT / "tpu.env").read_text()
        live = [
            ln for ln in text.splitlines()
            if ln and not ln.startswith("#") and ln.split("=")[0].startswith(("VLLM_", "TORCH_CUDA"))
        ]
        self.assertEqual(live, [])

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
            for source in (
                "server.py", "project-setup.sh", "requirements.txt",
                "requirements-serving.txt", "jax_openai_server.py", "jax_engine.py",
                "ports/gemma4/jax_e_loader.py", "ports/gemma4/jax_e_model.py",
            ):
                self.assertTrue(
                    filecmp.cmp(ROOT / source, ROOT / prefix / "mcp" / source, shallow=False),
                    f"{prefix}/mcp/{source} is stale — run `make skill`",
                )


if __name__ == "__main__":
    unittest.main()
