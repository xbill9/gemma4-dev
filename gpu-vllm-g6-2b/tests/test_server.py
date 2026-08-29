"""Offline regression tests for the G6 MCP server.

No AWS, no network, no GPU.

These pin the facts that survived the G5g -> G6 fork and, more importantly, the
ones that DID NOT. That fork changed both the GPU generation (Turing SM 7.5 ->
Ada SM 8.9) and the host architecture (aarch64 -> x86_64), so almost every
sibling constant is a silent copy-paste hazard in one direction or the other:

  * dtype flipped float16 -> bfloat16 (Ada has the datapath; the checkpoint is bf16)
  * the AMI filter flipped arm64 -> x86_64
  * the from-source SM 7.5 build is GONE, and with it TORCH_CUDA_ARCH_LIST,
    VLLM_REF, the stock/built image split and the `serving=` mode
  * host RAM DOUBLED at every size suffix, so the sibling's g5g.xlarge rejection
    does not carry
  * G6 is 4 GiB/vCPU where G5g was 2, so any `RAM // 2` vCPU shortcut doubles

The image-tag test is the sharpest one here: the sibling's *stock* tag is below
this model's measured vLLM floor, and it is exactly the value a fork inherits by
accident.
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
            "create_g6_instance", "list_g6_instances", "start_g6_instance",
            "stop_g6_instance", "terminate_g6_instance", "verify_gpu_arch",
            "get_install_progress", "get_vllm_logs", "get_endpoint",
            "verify_model_health", "query_model", "save_hf_token",
            "check_g6_quotas", "get_deployment_config", "get_help",
        }
        self.assertEqual(set(self.tools), expected)

    def test_annotations(self):
        destructive = {
            name for name, tool in self.tools.items() if tool.annotations.destructiveHint
        }
        self.assertEqual(destructive, {"stop_g6_instance", "terminate_g6_instance"})
        for name, tool in self.tools.items():
            self.assertTrue(tool.title, name)
            self.assertTrue(tool.description, name)
            self.assertIsNotNone(tool.annotations, name)

    def test_codex_gates_name_tools_that_exist(self):
        """A gate on a tool name that does not exist FAILS OPEN and says nothing.

        On the JAX rig's G5g -> G6 fork, .codex/config.toml kept the old
        `*_g5g_*` tool names, so every destructive tool was ungated under Codex
        while appearing to be gated. Nothing caught it, because the only naming
        test pinned server.py against the directory.
        """
        text = (ROOT / ".codex" / "config.toml").read_text()
        marker = f"[mcp_servers.{server.RIG_NAME}.tools."
        gated = {
            line[len(marker):].rstrip("]").strip()
            for line in text.splitlines()
            if line.startswith(marker)
        }
        self.assertTrue(gated, "no tools are gated at all")
        self.assertTrue(
            gated <= set(self.tools),
            f"gates name tools that do not exist: {sorted(gated - set(self.tools))}",
        )
        self.assertIn("terminate_g6_instance", gated)
        self.assertIn("stop_g6_instance", gated)

    def test_launch_defaults_to_spot_and_has_no_serving_mode(self):
        for name in ("create_g6_instance", "get_deployment_config"):
            schema = self.tools[name].inputSchema["properties"]
            self.assertTrue(schema["spot"]["default"], name)
            # The sibling's build/stock choice does not exist here: SM 8.9 is in
            # the published image, so `build` has nothing to do and `stock`
            # nothing to fail at. A parameter whose only value is the default is
            # a knob that suggests a choice nobody has.
            self.assertNotIn("serving", schema, name)


class G6TopologyTests(unittest.TestCase):
    def test_sizes_match_the_aws_product_page(self):
        expected = {
            "g6.xlarge": (1, 16),
            "g6.2xlarge": (1, 32),
            "g6.4xlarge": (1, 64),
            "g6.8xlarge": (1, 128),
            "g6.12xlarge": (4, 192),
            "g6.16xlarge": (1, 256),
            "g6.24xlarge": (4, 384),
            "g6.48xlarge": (8, 768),
        }
        for instance_type, (gpus, ram) in expected.items():
            self.assertTrue(server._is_g6(instance_type))
            self.assertEqual(server._gpu_count(instance_type), gpus)
            self.assertEqual(server._host_memory_gb(instance_type), ram)

    def test_gpu_count_is_not_monotonic_in_size(self):
        """g6.16xlarge is SINGLE-GPU where g5g.16xlarge had two.

        Anything that infers GPU count from the size suffix gets this wrong, and
        a wrong tensor-parallel size is a serve command that fails at startup.
        """
        self.assertEqual(server._gpu_count("g6.12xlarge"), 4)
        self.assertEqual(server._gpu_count("g6.16xlarge"), 1)
        self.assertEqual(server._gpu_count("g6.24xlarge"), 4)

    def test_vcpu_is_not_derived_from_host_ram(self):
        """G6 is 4 GiB/vCPU; G5g was 2. The inherited `RAM // 2` doubles here."""
        self.assertEqual(server._vcpu_count("g6.xlarge"), 4)
        self.assertEqual(server._vcpu_count("g6.2xlarge"), 8)
        self.assertEqual(server._vcpu_count("g6.48xlarge"), 192)
        self.assertNotEqual(
            server._vcpu_count("g6.xlarge"), server._host_memory_gb("g6.xlarge") // 2
        )

    def test_tensor_parallel_follows_gpu_count(self):
        self.assertEqual(server._tensor_parallel_size("g6.xlarge"), 1)
        self.assertEqual(server._tensor_parallel_size("g6.12xlarge"), 4)

    def test_no_g6_size_needs_a_swapfile(self):
        """Every G6 size has >= 16 GiB of host RAM, twice its g5g namesake.

        So the sibling's g5g.xlarge rejection does NOT carry, and the swap block
        never renders. Kept in the code because the threshold is a claim about
        the checkpoint (~10.2 GB to map), not about the host.
        """
        for size in server._G6_SIZES:
            self.assertFalse(server._needs_swap(size), size)
            text = server._user_data("google/gemma-4-E2B-it", size)
            self.assertNotIn("mkswap", text)

    def test_xlarge_is_accepted(self):
        server._validate_instance_type("g6.xlarge")  # must not raise

    def test_non_g6_rejected(self):
        for bad in ("g5g.2xlarge", "inf2.xlarge", "g6.unknown", "g6f.xlarge"):
            with self.assertRaises(ValueError):
                server._validate_instance_type(bad)


class AdaConstraintTests(unittest.TestCase):
    """Ada (SM 8.9) HAS bf16 and fp8. The T4G sibling hardcodes the absence of both."""

    def test_dtype_is_bfloat16_matching_the_checkpoint(self):
        flags = server._serve_flags("google/gemma-4-E2B-it", "g6.xlarge")
        self.assertIn("--dtype bfloat16", flags)
        # float16 here would make vLLM convert every weight on load. The JAX rig
        # on this exact silicon measured that mismatch at 54% of decode on Turing.
        self.assertNotIn("--dtype float16", flags)

    def test_kv_cache_follows_the_model_dtype(self):
        flags = server._serve_flags("google/gemma-4-E2B-it", "g6.xlarge")
        self.assertIn("--kv-cache-dtype auto", flags)
        # fp8 is REACHABLE here, unlike on Turing, and is deliberately not on:
        # KV is ~18 KiB/token, so the whole cache at 16K is ~288 MiB of 23034.
        self.assertNotIn("fp8", flags)

    def test_the_build_path_is_gone(self):
        """Nothing compiles here, so an arch list or source ref would be an inert
        setting that looks meaningful."""
        for dead in ("TORCH_CUDA_ARCH_LIST", "VLLM_REF", "VLLM_STOCK_IMAGE"):
            self.assertFalse(hasattr(server, dead), f"{dead} survived the fork")

    def test_image_is_at_or_above_the_measured_vllm_floor(self):
        """v0.26.0 dies with AmbiguousGlobalPerLayerAttributeError because Gemma
        4's head_dim is per-layer; the fix landed in v0.27.2rc0.

        This is a MODEL constraint, not a chip one, so it carries across the fork
        unchanged. The trap is that the sibling's VLLM_STOCK_IMAGE is v0.27.1 --
        it only ever used that tag to reproduce the SM 7.5 failure and never
        served from it, so it is below the floor and is exactly the value a fork
        inherits by accident.
        """
        self.assertNotIn(":v0.27.1", server.VLLM_IMAGE)
        self.assertNotIn(":v0.26", server.VLLM_IMAGE)
        self.assertTrue(server.VLLM_IMAGE.startswith("vllm/vllm-openai:"))

    def test_attention_backend_is_unpinned(self):
        """Measured on the sibling: vLLM v0.27 does not recognize
        VLLM_ATTENTION_BACKEND at all, and forces TRITON_ATTN for Gemma 4
        regardless, because the head dims are heterogeneous.

        Leaving it unpinned means vLLM dispatches for the real part. Pinning a
        backend is how the sibling ended up carrying an unlanded Triton patch.
        """
        self.assertEqual(server.ATTENTION_BACKEND, "")
        text = server._user_data("google/gemma-4-E2B-it", "g6.xlarge")
        # And an EMPTY value must not be exported: vLLM seeing the variable set
        # to "" is not the same as not seeing it.
        self.assertNotIn("VLLM_ATTENTION_BACKEND", text)


class UserDataTests(unittest.TestCase):
    def test_pulls_the_published_image_and_builds_nothing(self):
        text = server._user_data("google/gemma-4-E2B-it", "g6.xlarge")
        self.assertIn(f"docker pull {server.VLLM_IMAGE}", text)
        self.assertNotIn("docker build", text)
        self.assertNotIn("git clone", text)
        self.assertShellParses(text)

    def test_stage_markers_make_a_stall_attributable(self):
        text = server._user_data("google/gemma-4-E2B-it", "g6.xlarge")
        for marker in ("image-pull-start", "image-pull-done", "serving-started"):
            self.assertIn(marker, text)
        self.assertIn("INSTALL_DONE", text)

    def test_token_comes_from_secrets_manager_not_user_data(self):
        # User data is readable from instance metadata by anything on the box.
        text = server._user_data("google/gemma-4-E2B-it", "g6.xlarge")
        self.assertIn("secretsmanager get-secret-value", text)
        self.assertNotIn("hf_", text.lower().replace("hf_token", ""))

    def test_secret_fetch_disables_shell_tracing(self):
        # The script can run under `set -x`, and bash traces assignments WITH
        # THEIR VALUES -- so without this the token lands in the console log.
        text = server._user_data("google/gemma-4-E2B-it", "g6.xlarge")
        self.assertIn("set +x", text)
        self.assertLess(
            text.index("set +x"), text.index("secretsmanager"), "set +x must precede the fetch"
        )

    def test_multi_gpu_size_gets_tp4(self):
        text = server._user_data("google/gemma-4-E2B-it", "g6.12xlarge")
        self.assertIn("--tensor-parallel-size 4", text)

    def test_no_mkswap_q_flag(self):
        """`mkswap -q` is a busybox flag util-linux rejects with
        `invalid option -- 'q'`. Under `set -e` it killed cloud-init on the
        sibling BEFORE anything logged, costing a launch."""
        for size in server._G6_SIZES:
            self.assertNotIn("mkswap -q", server._user_data("m", size))

    def assertShellParses(self, text):
        proc = subprocess.run(
            ["bash", "-n", "/dev/stdin"], input=text, text=True, capture_output=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


class DeploymentConfigTests(unittest.TestCase):
    def test_config_is_offline_and_decodable(self):
        result = run(server.get_deployment_config(instance_type="g6.xlarge"))
        self.assertIn("aws ec2 run-instances", result)
        encoded = result.split("--user-data '", 1)[1].split("'", 1)[0]
        script = base64.b64decode(encoded).decode()
        self.assertIn("#!/usr/bin/env bash", script)

    def test_config_resolves_ami_via_ssm_parameter(self):
        # The architecture requirement FLIPPED at the fork: an arm64 DLAMI
        # cannot boot a G6 at all.
        result = run(server.get_deployment_config())
        self.assertIn("aws ssm get-parameter", result)
        self.assertIn("/x86_64/", server.DLAMI_SSM_PARAMETER)
        self.assertNotIn("/arm64/", server.DLAMI_SSM_PARAMETER)
        self.assertIn("nvidia-driver-gpu", server.DLAMI_SSM_PARAMETER)

    def test_ami_name_fallback_matches_the_same_family_as_the_ssm_path(self):
        """Changing the SSM path without changing this filter in the same commit
        is a revert that reports success: the fallback quietly resolves a
        different image.

        The sibling's pattern required "ARM64 AMI" CONTIGUOUSLY, which matches
        none of the base images ("Deep Learning Base OSS Nvidia Driver GPU AMI").
        """
        self.assertNotIn("ARM64", server.DLAMI_NAME)
        self.assertIn("Base", server.DLAMI_NAME)
        self.assertIn("Nvidia Driver GPU", server.DLAMI_NAME)

    def test_config_and_create_provision_the_same_root_volume(self):
        """The sibling PRINTED VolumeSize=200 while its create tool LAUNCHED 100.

        A copy-pasteable repro command that provisions a different volume from
        the tool it documents is how a manual reproduction fails to reproduce.
        """
        result = run(server.get_deployment_config())
        self.assertIn(f"VolumeSize={server.ROOT_VOLUME_GB}", result)
        self.assertIn(f"Throughput={server.ROOT_VOLUME_THROUGHPUT_MBPS}", result)
        self.assertIn(f"Iops={server.ROOT_VOLUME_IOPS}", result)

    def test_gp3_throughput_satisfies_the_iops_rule(self):
        """gp3 requires throughput <= IOPS * 0.25, enforced at run-instances
        time -- so violating it fails a LAUNCH, not just a disk."""
        self.assertLessEqual(
            server.ROOT_VOLUME_THROUGHPUT_MBPS, server.ROOT_VOLUME_IOPS * 0.25
        )

    def test_config_tags_with_rig_name(self):
        result = run(server.get_deployment_config())
        self.assertIn(f"Key=ManagedBy,Value={server.RIG_NAME}", result)

    def test_config_spot_toggle(self):
        self.assertIn("MarketType=spot", run(server.get_deployment_config()))
        self.assertNotIn("MarketType=spot", run(server.get_deployment_config(spot=False)))

    def test_config_rejects_non_g6(self):
        self.assertTrue(
            run(server.get_deployment_config(instance_type="g5g.2xlarge")).startswith("❌")
        )
        self.assertFalse(
            run(server.get_deployment_config(instance_type="g6.xlarge")).startswith("❌")
        )


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
        self.assertEqual(values["VLLM_IMAGE"], server.VLLM_IMAGE)
        self.assertEqual(values["SERVICE_NAME"], server.SERVICE_NAME)
        # Both AMI keys, together: changing one without the other is the silent
        # revert described in DeploymentConfigTests.
        self.assertEqual(values["DLAMI_SSM_PARAMETER"], server.DLAMI_SSM_PARAMETER)
        self.assertEqual(values["DLAMI_NAME"], server.DLAMI_NAME)
        self.assertEqual(int(values["ROOT_VOLUME_GB"]), server.ROOT_VOLUME_GB)
        self.assertEqual(
            int(values["ROOT_VOLUME_THROUGHPUT_MBPS"]), server.ROOT_VOLUME_THROUGHPUT_MBPS
        )
        self.assertEqual(int(values["ROOT_VOLUME_IOPS"]), server.ROOT_VOLUME_IOPS)
        self.assertEqual(
            int(values["TENSOR_PARALLEL_SIZE"]), server._gpu_count(server.INSTANCE_TYPE)
        )

    def test_no_turing_or_aarch64_key_survived_the_fork(self):
        text = (ROOT / "tpu.env").read_text()
        for key in ("TORCH_CUDA_ARCH_LIST=", "VLLM_REF=", "VLLM_STOCK_IMAGE=",
                    "CUDA_TOOLKIT_PACKAGE="):
            self.assertNotIn(f"\n{key}", text, f"{key} survived the fork as live config")

    def test_rig_name_matches_directory(self):
        self.assertEqual(server.RIG_NAME, ROOT.name)

    def test_project_setup_derives_the_skill_stem(self):
        """A hardcoded stem is what silently survives a rename.

        On the JAX rig's fork this literal still named the OLD rig, so the script
        could not find the skill and died with `cannot locate the bundled skill`
        -- the rig was UNREGISTERABLE, not merely misregistered.
        """
        text = (ROOT / "project-setup.sh").read_text()
        self.assertNotIn('SKILL_STEM="gpu-', text)
        self.assertIn('SKILL_STEM="$(basename "$SCRIPT_DIR")-management"', text)

    def test_registration_files_agree_on_the_server_name(self):
        """Four places name this server, and a mismatch makes /mcp and the tool
        prefix disagree about what this rig is.

        Comments are stripped before checking for the sibling's name: these files
        deliberately EXPLAIN the fork, and prose about `g5g` is the documentation
        working rather than a stale value. Only live config is checked.
        """
        for rel in (".mcp.json", ".claude-plugin/plugin.json", ".codex/config.toml",
                    ".claude/settings.local.json"):
            path = ROOT / rel
            if not path.is_file():  # .mcp.json and settings.local.json are gitignored
                continue
            text = path.read_text()
            self.assertIn(server.RIG_NAME, text, rel)
            live = "\n".join(
                line for line in text.splitlines() if not line.lstrip().startswith("#")
            )
            self.assertNotIn("g5g", live, f"{rel} still names the sibling rig in live config")

    def test_benchmarks_carries_no_other_rigs_runs(self):
        """Benchmark JSON travelled with the forks in this tree, and several rigs
        carry numbers measured on hardware they are not. This rig has served
        nothing, so runs/ must be absent or empty."""
        runs = ROOT / "benchmarks" / "runs"
        self.assertTrue(not runs.exists() or not any(runs.iterdir()))

    def test_skill_is_complete_in_both_copies(self):
        # SKILL.md is a hand-written SOURCE file, but refresh_skill.py only
        # regenerates the mcp/ files beside it. So `rm -rf .claude/skills`
        # destroys it and `make skill` does not bring it back.
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
