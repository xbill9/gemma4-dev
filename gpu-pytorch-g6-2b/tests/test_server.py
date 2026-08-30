"""Offline regression tests for the G6 JAX MCP server.

No AWS, no network, no GPU. These pin the facts that make this rig different
from its siblings — the Ada dtype policy, the x86_64 AMI filter, the
host-RAM floor, and the shared-memory ceiling that decides which kernels can
run — because every one of them is a silent copy-paste hazard from a sibling rig
that runs on different silicon.

The engine tests below import ports/gemma4 under JAX_PLATFORMS=cpu and then
stub the detected compute capability, so they exercise both the Ada and the
Turing branch on a machine with no GPU at all. That is a test of the *policy*,
not of the hardware.
"""

import asyncio
import base64
import filecmp
import subprocess
import sys
import unittest
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import server  # noqa: E402

# Spelled out rather than read from server, so that a change to the served
# checkpoint or the default size shows up as a test edit rather than passing
# vacuously against whatever the module happens to hold.
MODEL = "google/gemma-4-E2B-it"
DEFAULT_SIZE = "g6.2xlarge"
SWAP_SIZE = "g6.xlarge"        # the one size small enough to need a swapfile

# The eight files refresh_skill.py snapshots into both skill copies: the MCP
# control plane *and* the serving payload, because an installed skill still has
# to be able to run deploy_torch_server.
SKILL_SOURCES = (
    "server.py", "project-setup.sh", "requirements.txt",
    "requirements-serving.txt", "torch_openai_server.py", "torch_generate.py",
)


def run(coro):
    return asyncio.run(coro)


def user_data(instance_type=DEFAULT_SIZE, model=MODEL):
    return server._user_data(model, instance_type)


def user_data_from_config(config):
    """Pull the cloud-init script back out of a get_deployment_config rendering."""
    encoded = config.split("--user-data '", 1)[1].split("'", 1)[0]
    return base64.b64decode(encoded).decode()


def tpu_env():
    values = {}
    for line in (ROOT / "tpu.env").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key] = value
    return values


class BashSyntaxMixin:
    """`bash -n` assertions, shared by the rendered-script and repo-file tests."""

    def assertShellParses(self, text, label="rendered script"):
        proc = subprocess.run(
            ["bash", "-n", "/dev/stdin"], input=text, text=True, capture_output=True
        )
        self.assertEqual(proc.returncode, 0, f"{label}: {proc.stderr}")

    def assertScriptParses(self, path):
        proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"{path.name}: {proc.stderr}")


class ToolCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = {tool.name: tool for tool in run(server.mcp.list_tools())}

    def test_catalog(self):
        expected = {
            "create_g6_instance", "list_g6_instances", "start_g6_instance",
            "stop_g6_instance", "terminate_g6_instance", "verify_gpu_arch",
            "deploy_torch_server", "get_install_progress", "get_torch_logs",
            "get_endpoint", "verify_model_health", "query_model", "save_hf_token",
            "check_g6_quotas", "get_deployment_config", "get_help", "get_metrics",
        }
        self.assertEqual(set(self.tools), expected)

    def test_annotations(self):
        destructive = {
            name for name, tool in self.tools.items() if tool.annotations.destructiveHint
        }
        self.assertEqual(destructive, {"stop_g6_instance", "terminate_g6_instance"})
        for name, tool in self.tools.items():
            with self.subTest(tool=name):
                self.assertTrue(tool.title)
                self.assertTrue(tool.description)
                self.assertIsNotNone(tool.annotations)

    def test_launch_defaults_to_spot(self):
        for name in ("create_g6_instance", "get_deployment_config"):
            with self.subTest(tool=name):
                schema = self.tools[name].inputSchema["properties"]
                self.assertTrue(schema["spot"]["default"])
                # There is no `serving` mode here: nothing is built, so there is
                # no stock-vs-build choice the vLLM sibling has to offer.
                self.assertNotIn("serving", schema)


class G6TopologyTests(unittest.TestCase):
    """Instance-shape policy: which sizes exist, and what each one implies."""

    def test_sizes_match_the_aws_product_page(self):
        # VERIFIED 2026-08-29 against `ec2 describe-instance-types`, not a
        # product page. Two G5g habits are wrong here and both are pinned:
        # g6.16xlarge is SINGLE-GPU, and there is no g6.metal.
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
            with self.subTest(instance_type=instance_type):
                self.assertTrue(server._is_g6(instance_type))
                self.assertEqual(server._gpu_count(instance_type), gpus)
                self.assertEqual(server._host_memory_gb(instance_type), ram)

    def test_tensor_parallel_follows_gpu_count(self):
        self.assertEqual(server._tensor_parallel_size("g6.2xlarge"), 1)
        # SINGLE-GPU despite the suffix -- the G5g rig's 16xlarge had 2.
        self.assertEqual(server._tensor_parallel_size("g6.16xlarge"), 1)
        self.assertEqual(server._tensor_parallel_size("g6.12xlarge"), 4)
        self.assertEqual(server._tensor_parallel_size("g6.48xlarge"), 8)

    def test_every_size_is_supported_and_only_the_smallest_needs_swap(self):
        """On G6 exactly ONE size is at or below the gate, and that is new.

        The G5g rig had TWO sizes in this set, because G5g's xlarge/2xlarge are
        8/16 GiB. G6 has twice the host RAM at every suffix -- 16/32 GiB -- so
        only g6.xlarge sits on the inclusive 16 GiB gate. Transferring the G5g
        verdict onto the same G6 size name is exactly the mistake this pins.

        The g5g 2xlarge OOM that forced the gate inclusive was a JAX
        `quantize_ple_table` upcast; this rig has no PLE path at all.
        """
        server._validate_instance_type(SWAP_SIZE)   # must not raise
        self.assertTrue(server._needs_swap(SWAP_SIZE))
        for size in ("g6.2xlarge", "g6.4xlarge", "g6.8xlarge", "g6.12xlarge",
                     "g6.16xlarge", "g6.24xlarge", "g6.48xlarge"):
            with self.subTest(instance_type=size):
                server._validate_instance_type(size)
                self.assertFalse(server._needs_swap(size))

    def test_no_swap_block_for_the_default_size(self):
        # g6.2xlarge is the size this rig launches and measures, and at 32 GiB
        # it is above the gate -- so the swap path is UNEXERCISED on G6.
        self.assertNotIn("swapfile", user_data(instance_type="g6.2xlarge"))
        self.assertIn("swapfile", user_data(instance_type="g6.xlarge"))

    def test_non_g6_rejected(self):
        # g6.metal does not exist; g5g.* is the sibling family.
        for bad in ("g6.metal", "g5g.2xlarge", "inf2.xlarge", "g6.unknown"):
            with self.subTest(instance_type=bad), self.assertRaises(ValueError):
                server._validate_instance_type(bad)


class UserDataTests(BashSyntaxMixin, unittest.TestCase):
    """The rendered cloud-init script, which installs the runtime and nothing else."""

    def test_install_is_wheels_not_a_build(self):
        # Derived from TORCH_PIP_SPEC, never a literal: a hardcoded "jax[cuda12]"
        # here turns a routine CUDA-line bump into a test edit, which is friction
        # against the standing preference for latest versions. The claim under
        # test is "we install the configured spec from wheels", not which spec.
        text = user_data()
        self.assertIn(server.TORCH_PIP_SPEC, text)
        self.assertNotIn("docker build", text)
        self.assertNotIn("git clone", text)
        # cloud-init must not block on the install.
        self.assertIn("nohup", text)
        self.assertIn("INSTALL_DONE", text)
        self.assertShellParses(text)

    def test_a_modern_python_is_installed_because_jax_requires_it(self):
        # jax >= 0.11 declares requires-python >= 3.12 and Ubuntu 22.04 ships
        # 3.10, so the DLAMI's system python would fail at pip install time.
        # Asserted against TORCH_PYTHON_VERSION rather than a literal so the
        # interpreter can be moved forward without editing tests.
        text = user_data()
        self.assertIn("deadsnakes", text)
        self.assertIn(f"python{server.TORCH_PYTHON_VERSION}", text)
        self.assertGreaterEqual(
            tuple(int(x) for x in server.TORCH_PYTHON_VERSION.split(".")), (3, 12),
            "jax >= 0.11 requires Python 3.12 or newer",
        )

    def test_systemd_execstart_is_absolute(self):
        # systemd refuses a relative ExecStart, and the unit would fail to load
        # with a message that says nothing about the interpreter.
        self.assertIn(f"ExecStart=/usr/bin/python{server.TORCH_PYTHON_VERSION}", user_data())

    def test_execstart_is_repointed_at_the_installed_interpreter(self):
        # MEASURED 2026-08-19 on i-063d52c913140b787: the DLAMI already ships
        # /usr/local/bin/python3.12, which precedes /usr/bin on PATH. install.sh
        # calls bare `python3.12`, so jax landed in /usr/local, while the unit's
        # hardcoded ExecStart=/usr/bin/python3.12 crash-looped on
        # ModuleNotFoundError -- AFTER the install reported success, because the
        # verify step resolves through PATH too.
        text = user_data()
        # Since 2026-08-29 the path is READ from the marker install_runtime
        # wrote rather than re-resolved: on this rig torch lives in the DLAMI's
        # venv, which is not on PATH as `python3.12` at all, so `command -v`
        # could not name the interpreter that has it.
        self.assertIn('PY_BIN="$(cat /opt/torch-g6/PYTHON_BIN)"', text)
        self.assertIn("ExecStart=$PY_BIN", text)
        # The rewrite has to happen after the install, not in the unit template.
        self.assertLess(text.index("install_runtime\nverify_gpu"), text.index("PY_BIN="))

    def test_token_comes_from_secrets_manager_not_user_data(self):
        # User data is readable from instance metadata by anything on the box.
        text = user_data()
        self.assertIn("secretsmanager get-secret-value", text)
        self.assertNotIn("hf_", text.lower().replace("hf_token", ""))

    def test_xtrace_is_disabled_around_the_secret_fetch(self):
        # The script runs under `set -x`, and bash traces assignments WITH their
        # values — so leaving xtrace on would print the token into
        # /var/log/cloud-init-output.log, which is the exact exposure that
        # keeping it out of user data is meant to prevent.
        text = user_data()
        fetch = text.index("secretsmanager get-secret-value")
        self.assertIn("set +x", text[:fetch])
        self.assertLess(text[:fetch].rindex("set +x"), fetch)
        self.assertIn("set -x", text[fetch:])

    def test_env_file_is_locked_down_before_the_token_lands(self):
        text = user_data()
        self.assertLess(text.index("chmod 600 /opt/torch-g6/env"), text.index("HF_TOKEN="))

    def test_swap_block_uses_only_portable_flags(self):
        """`mkswap -q` is busybox; util-linux rejects it and `set -e` kills cloud-init.

        On G6 only g6.xlarge renders the block at all (16 GiB, on the inclusive
        gate), and that size has never been launched here -- so this is again a
        path nobody exercises, which is exactly how the busybox flag survived on
        the G5g sibling until it cost a launch. Pinned rather than trusted.
        """
        rendered = user_data(instance_type=SWAP_SIZE)
        # Executable lines only -- the comment above the fix quotes the bad flag
        # on purpose, and matching prose would fail vacuously.
        code = "\n".join(ln for ln in rendered.splitlines()
                          if not ln.lstrip().startswith("#"))
        self.assertIn("mkswap /swapfile", code)
        self.assertNotIn("mkswap -q", code)

    def test_swapfile_is_rendered_only_for_the_smallest_host(self):
        """Only g6.xlarge (16 GiB) is at or below the inclusive gate on G6."""
        rendered = user_data(SWAP_SIZE)
        for fragment in ("mkswap", "swapon /swapfile", "/etc/fstab"):
            self.assertIn(fragment, rendered)
        self.assertShellParses(rendered, SWAP_SIZE)
        for size in ("g6.2xlarge", "g6.4xlarge", "g6.8xlarge", "g6.16xlarge"):
            with self.subTest(instance_type=size):
                self.assertNotIn("mkswap", user_data(size))

    def test_serving_requirements_match_the_mirror_file(self):
        # A drifted pair is invisible until a serve fails on a missing import.
        listed = {
            line.strip()
            for line in (ROOT / "requirements-serving.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        expected = (
            set(server._SERVING_REQUIREMENTS)
            | set(server._PROFILING_REQUIREMENTS)
            | {server.TORCH_PIP_SPEC}
        )
        self.assertEqual(listed, expected)

    def test_bootstrap_installs_the_profiling_tools(self):
        # Asked for explicitly, and easy to lose: they install in a non-fatal
        # branch, so a silent drop would leave the box serving and unprofilable.
        rendered = server._user_data("google/gemma-4-E2B-it", "g6.2xlarge")
        for pkg in server._PROFILING_REQUIREMENTS:
            self.assertIn(pkg, rendered)
        self.assertIn("profiling-deps", rendered)

    def test_bootstrap_has_no_jax_left_in_it(self):
        # Four separate fork-debris breakages lived here until 2026-08-29, and
        # every one of them was fatal at a different stage: verify_gpu imported
        # jax (killing install.sh under `set -e`), ExecStart pointed at
        # jax_openai_server.py, _serve_argv emitted the JAX port's flags, and the
        # pip spec was quoted as one requirement. None was visible offline
        # because nothing asserted on the rendered script.
        rendered = server._user_data("google/gemma-4-E2B-it", "g6.2xlarge")
        self.assertNotIn("import jax", rendered)
        self.assertNotIn("jax_openai_server.py", rendered)
        self.assertIn("torch_openai_server.py", rendered)

    def test_serve_argv_only_emits_flags_the_server_defines(self):
        # torch_openai_server.py defines exactly --model/--host/--port/--seq.
        # argparse exits 2 on an unknown flag, so an extra one here crash-loops
        # the unit with the reason only in journalctl.
        argv = server._serve_argv("google/gemma-4-E2B-it", "g6.2xlarge")
        allowed = {"--model", "--host", "--port", "--seq"}
        flags = {tok for tok in argv.split() if tok.startswith("--")}
        self.assertEqual(flags, allowed)


class MetricsParsingTests(unittest.TestCase):
    """_parse_prom is pure, so the interesting parts pin offline."""

    EXPOSITION = "\n".join([
        "# HELP tpu_jax_precision_info Dtypes and quantisation resolved on device",
        "# TYPE tpu_jax_precision_info gauge",
        # Ada resolves bfloat16 and pre_ampere=false. The T4G sibling's fixture
        # said float16/true; if you are copying a transcript between the two
        # rigs, this row is the one that must change.
        'tpu_jax_precision_info{model="google/gemma-4-E2B-it",compute_dtype="bfloat16",'
        'quant_mode="bf16",kv_cache_dtype="bfloat16",kv_cache_requested="auto",'
        'ple_bits="0",int8_lm_head="false",pre_ampere="false"} 1',
        "",
        'tpu_jax_requests_total{model="google/gemma-4-E2B-it",status="success"} 3',
        'tpu_jax_requests_total{model="google/gemma-4-E2B-it",status="failed"} 0',
        'tpu_jax_decode_tokens_per_second{model="google/gemma-4-E2B-it"} 12.3',
        'tpu_jax_hbm_used_bytes{device="cuda:0"} 9299057152',
        "",
    ])

    def setUp(self):
        self.samples, self.precision, self.model = server._parse_prom(self.EXPOSITION)

    def test_served_checkpoint_survives_parsing(self):
        # The model label is dropped from every row as noise; if it were not
        # hoisted out first, a metrics transcript could not be checked against
        # MODEL_NAME -- which is the whole point of quoting one.
        self.assertEqual(self.model, "google/gemma-4-E2B-it")

    def test_precision_is_split_out_not_rendered_as_a_number(self):
        # The info series carries its payload in labels and a constant value of
        # 1, so leaving it among the samples would print a meaningless "1" row.
        self.assertNotIn("tpu_jax_precision_info", "".join(self.samples))
        self.assertEqual(self.precision["compute_dtype"], "bfloat16")
        self.assertEqual(self.precision["kv_cache_dtype"], "bfloat16")
        self.assertEqual(self.precision["kv_cache_requested"], "auto")
        self.assertEqual(self.precision["quant_mode"], "bf16")
        self.assertEqual(self.precision["pre_ampere"], "false")

    def test_distinguishing_labels_are_kept(self):
        # status= genuinely separates two series; dropping it would collide them.
        self.assertEqual(self.samples['tpu_jax_requests_total{status="success"}'], 3.0)
        self.assertEqual(self.samples['tpu_jax_requests_total{status="failed"}'], 0.0)
        self.assertEqual(self.samples["tpu_jax_decode_tokens_per_second"], 12.3)
        self.assertEqual(self.samples['tpu_jax_hbm_used_bytes{device="cuda:0"}'], 9299057152.0)

    def test_comments_and_blank_lines_are_ignored(self):
        self.assertFalse([k for k in self.samples if k.startswith("#")])

    def test_empty_exposition_yields_nothing(self):
        samples, precision, model = server._parse_prom("")
        self.assertEqual((samples, precision, model), ({}, {}, None))


class PayloadTests(unittest.TestCase):
    """The serving payload ships over SSM, so its size and determinism matter."""

    def test_payload_files_exist_and_are_found(self):
        root = Path(server._payload_root())
        for rel in server._PAYLOAD_FILES:
            with self.subTest(path=rel):
                self.assertTrue((root / rel).is_file())


    def test_payload_gzip_header_carries_no_timestamp(self):
        """The flake the deterministic test could only catch by luck.

        Bytes 4:8 of a gzip stream are MTIME. `tarfile.open(mode="w:gz")` fills
        them from the clock, so two calls either side of a second boundary
        produced different payloads while every tar entry was correctly zeroed.
        Assert the header directly rather than hoping two calls race.
        """
        import base64 as _b64
        blob = _b64.b64decode(server._payload_tar_b64())
        self.assertEqual(blob[:2], b"\x1f\x8b")       # gzip magic
        self.assertEqual(blob[4:8], b"\x00\x00\x00\x00", "gzip MTIME must be zeroed")

    def test_payload_is_deterministic(self):
        # Idempotent redeploys depend on this: same sources, same bytes.
        self.assertEqual(server._payload_tar_b64(), server._payload_tar_b64())

    def test_payload_fits_one_ssm_run_command(self):
        # SSM caps command document size; user data could not carry this at all
        # (16 KB). Keep a wide margin so adding a module does not silently break
        # deployment.
        size = len(server._payload_tar_b64())
        self.assertLess(size, 80_000, f"payload base64 is {size} bytes")


class DeployRestartTests(unittest.TestCase):
    """A redeploy has to replace the running process, not just the files."""

    def _command(self, restart):
        captured = {}

        async def fake_ssm(instance_id, command, timeout=300):
            captured["command"] = command
            return "ActiveState=active"

        original = server._ssm
        server._ssm = fake_ssm
        try:
            run(server.deploy_torch_server(instance_id="i-test", restart=restart))
        finally:
            server._ssm = original
        return captured["command"]

    def test_redeploy_restarts_rather_than_enable_now(self):
        # MEASURED 2026-08-23 on i-02f74ac9b944576c5: `systemctl enable --now` is
        # a no-op against an already-running unit, so a redeploy shipped the new
        # files and left the OLD process serving. `is-active` then printed
        # "active" and the tool reported success. Verified by /health still
        # returning the pre-deploy payload 17 minutes after the redeploy.
        command = self._command(restart=True)
        self.assertIn(f"systemctl restart {server.SERVICE_NAME}", command)
        self.assertNotIn("enable --now", command)

    def test_restart_reports_something_that_can_disprove_staleness(self):
        # "active" is true of the stale process too. The start timestamp and PID
        # are what let an operator tell a fresh process from the old one.
        command = self._command(restart=True)
        self.assertIn("ExecMainStartTimestamp", command)
        self.assertIn("MainPID", command)

    def test_restart_false_touches_no_units(self):
        command = self._command(restart=False)
        self.assertNotIn("systemctl", command)


class DeploymentConfigTests(unittest.TestCase):
    def test_config_is_offline_and_decodable(self):
        result = run(server.get_deployment_config(instance_type=DEFAULT_SIZE))
        self.assertIn("aws ec2 run-instances", result)
        self.assertIn("#!/usr/bin/env bash", user_data_from_config(result))

    def test_config_resolves_ami_via_ssm_parameter(self):
        # G6 is x86_64 -- the fork from G5g flipped the host architecture, so an
        # arm64 parameter path here resolves an AMI that cannot boot at all.
        self.assertIn("aws ssm get-parameter", run(server.get_deployment_config()))
        self.assertIn("/x86_64/", server.DLAMI_SSM_PARAMETER)
        self.assertNotIn("/arm64/", server.DLAMI_SSM_PARAMETER)
        self.assertIn("nvidia-driver-gpu", server.DLAMI_SSM_PARAMETER)

    def test_ami_name_fallback_does_not_name_the_arch(self):
        """The arch is NOT in an x86_64 DLAMI's name, and asserting it is a trap.

        VERIFIED 2026-08-29 against describe-images: only the ARM64 images
        announce their arch. Substituting "arm64"->"x86_64" in the G5g rig's
        pattern yields "Deep Learning x86_64 AMI OSS ...", which matched
        ZERO images -- a fallback that silently never fires. The architecture is
        pinned by an explicit filter on the call instead.
        """
        self.assertNotIn("x86_64", server.DLAMI_NAME)
        self.assertNotIn("ARM64", server.DLAMI_NAME)
        self.assertIn("Nvidia Driver", server.DLAMI_NAME)

    def test_config_tags_with_rig_name(self):
        result = run(server.get_deployment_config())
        self.assertIn(f"Key=ManagedBy,Value={server.RIG_NAME}", result)

    def test_config_spot_toggle(self):
        self.assertIn("MarketType=spot", run(server.get_deployment_config()))
        self.assertNotIn("MarketType=spot", run(server.get_deployment_config(spot=False)))

    def test_config_rejects_non_g6_only(self):
        # Every G6 size is supported, including the smallest (it gets a
        # swapfile). Only other families are refused -- g5g.2xlarge is the
        # sibling rig's default and is the realistic way to get this wrong.
        for bad in ("g5g.2xlarge", "g6.metal", "p4d.24xlarge"):
            with self.subTest(instance_type=bad):
                self.assertTrue(
                    run(server.get_deployment_config(instance_type=bad)).startswith("❌"))
        for good in (SWAP_SIZE, "g6.2xlarge", "g6.48xlarge"):
            with self.subTest(instance_type=good):
                self.assertFalse(
                    run(server.get_deployment_config(instance_type=good)).startswith("❌"))
        # ...and the smallest size still carries its swapfile through the config.
        cfg = run(server.get_deployment_config(instance_type=SWAP_SIZE))
        self.assertIn("mkswap", user_data_from_config(cfg))


class CubinCompatibilityTests(unittest.TestCase):
    """The arch-list check must not demand an EXACT sm_XY match.

    MEASURED 2026-08-29 on i-0f5ad4013e7265ee4: the DLAMI's torch 2.13.0+cu130
    carries ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120'] and the L4 is
    sm_89, so an exact match ABORTED THE INSTALL on a perfectly healthy GPU.
    fp16/bf16/fp32 matmuls then all ran correctly off the sm_86 cubin and
    torch.cuda.is_bf16_supported() was True.

    CUDA binary compatibility: a cubin runs on any device of the same MAJOR
    version whose minor is >= the one it was built for. The exact test was
    inherited from the Turing sibling, where sm_75 is present and it happened
    to hold.
    """

    # Exactly what the DLAMI's torch 2.13.0+cu130 reported on 2026-08-29.
    ARCHS: ClassVar[list[str]] = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]

    @staticmethod
    def _compat(archs, major, minor):
        """Mirror of the check rendered into the bootstrap."""
        return [a for a in archs
                if a.startswith("sm_") and a[3:].isdigit()
                and int(a[3:-1]) == major and int(a[3:][-1]) <= minor]

    def test_the_l4_is_covered_by_sm86(self):
        self.assertEqual(self._compat(self.ARCHS, 8, 9), ["sm_80", "sm_86"])

    def test_a_higher_minor_does_not_cover_a_lower_one(self):
        # sm_86 must NOT be offered as cover for an sm_80 device.
        self.assertEqual(self._compat(self.ARCHS, 8, 0), ["sm_80"])

    def test_a_different_major_never_counts(self):
        # sm_90 is Hopper; it cannot run Ada code and vice versa.
        self.assertNotIn("sm_90", self._compat(self.ARCHS, 8, 9))
        self.assertEqual(self._compat(["sm_90"], 8, 9), [])

    def test_turing_still_matches_exactly(self):
        self.assertEqual(self._compat(self.ARCHS, 7, 5), ["sm_75"])

    def test_the_bootstrap_does_not_use_an_exact_match(self):
        rendered = user_data()
        self.assertNotIn('in torch.cuda.get_arch_list()', rendered)
        self.assertIn("compatible cubins:", rendered)

    def test_the_probe_checks_the_dtype_it_will_serve(self):
        """An fp16-only probe passes on a chip whose bf16 path is broken.

        bf16 is what this rig actually serves, so both the bootstrap and the
        verify tool select the dtype from the capability rather than hardcoding
        float16. Asserted against the source, since the probe is a string built
        for a remote interpreter.
        """
        src = (ROOT / "server.py").read_text()
        self.assertIn("torch.cuda.is_bf16_supported()", src)
        # the bootstrap picks the dtype rather than pinning fp16
        self.assertIn("dtype = torch.float16 if (major, minor) < (8, 0) else torch.bfloat16", src)
        self.assertNotIn("print('fp16 matmul ok:'", src)


class LintCoverageTests(unittest.TestCase):
    """`make lint` lints a HARDCODED list, so a new module is silently unlinted.

    That is not hypothetical: profile_decode.py sat outside the list and was red
    for a day. ports/ is excluded on purpose -- ruff's UP006/UP045 would rewrite
    its Dict/Optional annotations, which the monorepo CLAUDE.md forbids and which
    would drift it from the copy tpu-jax-v5e1-2b shares.
    """

    def test_every_top_level_module_is_linted(self):
        listed = (ROOT / "Makefile").read_text()
        modules = sorted(p.name for p in ROOT.glob("*.py"))
        self.assertTrue(modules, "no top-level modules found")
        for m in modules:
            with self.subTest(module=m):
                self.assertIn(m, listed, f"{m} is not in the `make lint` file list")


class LatestVersionPolicyTests(BashSyntaxMixin, unittest.TestCase):
    """Newest release is the default here; a pin needs a named constraint."""

    def test_torch_is_not_installed_by_pip(self):
        """torch comes from the DLAMI -- but the REASON is weaker than on G5g.

        On the T4G sibling this was a hard requirement: upstream PyPI wheels omit
        sm_75, so `pip install torch` there serves on CPU or demands a build.
        **That argument does not transfer to Ada.** Upstream CUDA wheels carry
        sm_89, so this rig COULD pip-install torch.

        It still does not, and the honest reason is install time rather than
        capability: the PyTorch DLAMI already has a matching torch+driver pair,
        which keeps a multi-GB wheel download off the critical path on a spot
        host that may be reclaimed mid-install. Recorded as a choice, not a
        constraint, so nobody re-derives a Turing rationale for it.
        """
        self.assertNotIn("torch", server.TORCH_PIP_SPEC)
        self.assertIn("transformers", server.TORCH_PIP_SPEC)
        self.assertNotIn("==", server.TORCH_PIP_SPEC)
        # Ubuntu 26.04 ships 3.14 as the system interpreter. This value only
        # seeds candidate NAMES: the bootstrap probes for the interpreter that
        # can already import torch (the DLAMI's venv) and installs into that.
        self.assertEqual(server.TORCH_PYTHON_VERSION, "3.14")

    def test_ami_is_a_pytorch_image_not_the_base_one(self):
        """Inverted from the JAX sibling, on purpose.

        There the PyTorch DLAMI was GBs of deliberately unused image, because
        jax brings its own CUDA. Here the image's torch IS the runtime, so the
        base driver image would mean downloading a multi-GB wheel at boot.
        Unlike on Turing that would still *work* (upstream wheels carry sm_89);
        it would just put the download on the critical path of a spot host.
        """
        self.assertIn("pytorch", server.DLAMI_SSM_PARAMETER)
        self.assertNotIn("base-oss-nvidia-driver-gpu-ubuntu",
                         server.DLAMI_SSM_PARAMETER)

    def test_ami_line_is_one_aws_still_rebuilds(self):
        # `/latest/` is only the newest build WITHIN a PyTorch+Ubuntu line, and
        # AWS freezes those. The old pin resolved to an image built 2026-05-02
        # and could never move again: it READ as "track latest" and was a pin to
        # a dead line. 22.04 is where those lines stopped.
        self.assertNotIn("ubuntu-22.04", server.DLAMI_SSM_PARAMETER)
        self.assertNotIn("ubuntu-24.04", server.DLAMI_SSM_PARAMETER)
        self.assertIn("/x86_64/", server.DLAMI_SSM_PARAMETER)
        self.assertIn("nvidia-driver-gpu", server.DLAMI_SSM_PARAMETER)

    def test_name_fallback_matches_the_base_images_it_now_targets(self):
        # _resolve_ami takes newest-by-CreationDate, so a loose pattern silently
        # selects the wrong image. This rig needs the PyTorch line specifically:
        # the base driver image and the driverless Graviton-CPU image must both
        # fail to match, for opposite reasons.
        import fnmatch
        # Real names, copied from describe-images on 2026-08-29 -- not invented.
        current = "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.13 (Ubuntu 26.04) 20260829"
        base = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 26.04) 20260828"
        arm = "Deep Learning ARM64 AMI OSS Nvidia Driver GPU PyTorch 2.11 (Amazon Linux 2023) 20260828"
        self.assertTrue(fnmatch.fnmatch(current, server.DLAMI_NAME))
        # The base driver image has no torch: it must not match.
        self.assertFalse(fnmatch.fnmatch(base, server.DLAMI_NAME))
        # The ARM64 image cannot boot on G6 at all. It fails the pattern twice
        # over -- wrong arch prefix, and Amazon Linux rather than Ubuntu.
        self.assertFalse(fnmatch.fnmatch(arm, server.DLAMI_NAME))

    def test_deadsnakes_is_conditional_not_unconditional(self):
        # Ubuntu 26.04 ships python3.14 as the SYSTEM interpreter, so the PPA is
        # off the critical path there -- but the branch has to remain for anyone
        # overriding DLAMI_SSM_PARAMETER back to 22.04/24.04.
        text = user_data()
        self.assertIn(f"command -v python{server.TORCH_PYTHON_VERSION}", text)
        self.assertIn("ppa:deadsnakes/ppa", text)
        self.assertLess(text.index(f"command -v python{server.TORCH_PYTHON_VERSION}"),
                        text.index("ppa:deadsnakes/ppa"))

    def test_pip_can_install_into_an_externally_managed_interpreter(self):
        # PEP 668: Ubuntu marks its system interpreter externally-managed from
        # 23.04 on, so on the 24.04/26.04 bases every system-wide pip install
        # fails with `error: externally-managed-environment`. Without this the
        # move to a newer base bricks the install.
        text = user_data()
        self.assertIn("--break-system-packages", text)
        # get-pip.py runs before PIP is defined and needs the flag of its own.
        self.assertIn("get-pip.py |", text)
        # Comments mentioning get-pip.py are not invocations of it.
        get_pip = [
            ln for ln in text.splitlines()
            if "get-pip.py" in ln and not ln.lstrip().startswith("#")
        ]
        self.assertTrue(all("--break-system-packages" in ln for ln in get_pip), get_pip)

    def test_a_missing_aws_cli_is_reported_rather_than_silently_tokenless(self):
        # Changing the base image made this newly plausible. A missing CLI, an
        # empty secret and a denied GetSecretValue used to render identically as
        # "no token", surfacing minutes later as a 401 on the download.
        text = user_data()
        self.assertIn("command -v aws", text)
        self.assertIn("401", text)

    def test_the_token_still_never_reaches_user_data(self):
        # The rewrite above must not have relaxed the invariant the whole block
        # exists for: xtrace off across the fetch, and no literal token.
        text = user_data()
        self.assertNotIn("hf_", text.lower().replace("hf_token=$hf", ""))
        self.assertLess(text.index("set +x"), text.index("secretsmanager get-secret-value"))
        self.assertLess(text.index("secretsmanager get-secret-value"),
                        text.rindex("set -x"))

    def test_rendered_bootstrap_still_parses(self):
        text = user_data()
        self.assertShellParses(text, "cloud-init")
        inner = text.split("<<'INSTEOF'\n", 1)[1].split("\nINSTEOF", 1)[0]
        self.assertShellParses(inner, "install.sh")


class InstallTracingTests(unittest.TestCase):
    """A dead bootstrap and a running one must not share a rendering."""

    def _progress(self, ssm_output):
        async def fake_ssm(instance_id, command, timeout=300):
            self._command = command
            return ssm_output

        original = server._ssm
        server._ssm = fake_ssm
        try:
            return run(server.get_install_progress(instance_id="i-test"))
        finally:
            server._ssm = original

    def test_it_asks_cloud_init_for_its_own_verdict(self):
        # The install log only exists once cloud-init has reached install.sh, so
        # it cannot testify about anything that killed cloud-init before that.
        self._progress("INSTALL IN PROGRESS")
        self.assertIn("cloud-init status", self._command)
        self.assertIn("/var/log/cloud-init-output.log", self._command)

    def test_cloud_init_error_is_not_reported_as_in_progress(self):
        # The 2026-08-26 signature: `mkswap -q` failed under `set -e` in the swap
        # block, which renders BEFORE install.sh is written. The old rendering
        # was "INSTALL IN PROGRESS" + "no install log yet", forever -- identical
        # to a healthy slow install.
        verdict = self._progress(
            "INSTALL IN PROGRESS\n--- cloud-init ---\nstatus: error\n"
            "--- install log ---\nNO INSTALL LOG: cloud-init never reached install.sh"
        )
        self.assertIn("❌", verdict)
        self.assertIn("cloud-init FAILED", verdict)
        self.assertIn("NOT a slow install", verdict)

    def test_done_without_an_install_log_is_also_a_failure(self):
        # cloud-init can exit 0 having never backgrounded install.sh. Nothing is
        # installing, and waiting will not change that.
        verdict = self._progress(
            "INSTALL IN PROGRESS\n--- cloud-init ---\nstatus: done\n"
            "--- install log ---\nNO INSTALL LOG: cloud-init never reached install.sh"
        )
        self.assertIn("❌", verdict)
        self.assertIn("never wrote", verdict)

    def test_early_boot_without_a_log_is_still_in_progress(self):
        # cloud-init genuinely still running is the one case where a missing
        # install log is benign, and it must not be reported as a failure.
        verdict = self._progress(
            "INSTALL IN PROGRESS\n--- cloud-init ---\nstatus: running\n"
            "--- install log ---\nNO INSTALL LOG: cloud-init never reached install.sh"
        )
        self.assertIn("⏳", verdict)
        self.assertNotIn("❌", verdict)

    def test_complete_outranks_everything_else(self):
        verdict = self._progress("INSTALL COMPLETE\nstatus: done")
        self.assertIn("✅", verdict)
        self.assertIn("deploy_torch_server", verdict)


class InstallStageTimingTests(BashSyntaxMixin, unittest.TestCase):
    """The install was the longest phase of a deploy and the only untimed one."""

    def test_every_install_step_emits_a_stage_marker(self):
        text = user_data()
        for name in ("apt-base", "torch-interpreter", "pip-bootstrap", "torch-deps",
                     "serving-deps", "profiling-deps", "gpu-verify"):
            with self.subTest(stage=name):
                self.assertIn(f"stage {name}", text)

    def test_torch_deps_are_timed_separately_from_apt(self):
        # This is the whole point of the split: a spot reclamation at 21 minutes
        # (MEASURED 2026-08-25) left no record of which step was running, and apt
        # and several GB of CUDA wheels are very different things to attack.
        text = user_data()
        self.assertLess(text.index("stage apt-base"), text.index("stage torch-deps"))

    def test_stage_helper_is_defined_before_it_is_called(self):
        text = user_data()
        self.assertLess(text.index("stage() {"), text.index("stage apt-base"))


class RootVolumeTests(unittest.TestCase):
    """gp3 defaults to 125 MiB/s and the load sat on it."""

    def test_launch_and_documented_command_agree(self):
        # These disagreed: get_deployment_config printed VolumeSize=200 while
        # create_g6_instance launched 100, so the copy-pasteable repro command
        # provisioned a different volume from the tool it documents.
        rendered = run(server.get_deployment_config())
        self.assertIn(f"VolumeSize={server.ROOT_VOLUME_GB}", rendered)
        self.assertNotIn("VolumeSize=200", rendered)

    def test_throughput_is_set_rather_than_left_at_the_gp3_default(self):
        # MEASURED 2026-08-25: read_shards ~139 MB/s and download ~116 MB/s, both
        # on gp3's 125 MiB/s baseline. Two unrelated stages landing on one number
        # is a volume ceiling.
        self.assertGreater(server.ROOT_VOLUME_THROUGHPUT_MBPS, 125)
        self.assertIn(f"Throughput={server.ROOT_VOLUME_THROUGHPUT_MBPS}",
                      run(server.get_deployment_config()))

    def test_gp3_throughput_to_iops_ratio_is_satisfiable(self):
        # gp3 rejects the volume outright if throughput exceeds IOPS * 0.25 MiB/s,
        # and it rejects it at RUN-INSTANCES time -- so a bad pair here is a
        # launch failure, not a slow disk.
        self.assertLessEqual(server.ROOT_VOLUME_THROUGHPUT_MBPS,
                             server.ROOT_VOLUME_IOPS * 0.25)
        self.assertLessEqual(server.ROOT_VOLUME_THROUGHPUT_MBPS, 1000)
        self.assertGreaterEqual(server.ROOT_VOLUME_IOPS, 3000)
        self.assertLessEqual(server.ROOT_VOLUME_IOPS, 16000)


# The JAX rig's CompilationCacheTests were REMOVED with the feature they covered.
# torch compiles nothing on this path, so an XLA compilation cache has nothing to
# persist: the restore would sync an empty prefix and the timer would upload one,
# both reporting success forever. Tests that pin a no-op are worse than none --
# they make the no-op look load-bearing.
class RepoHygieneTests(BashSyntaxMixin, unittest.TestCase):
    def test_shell_scripts_parse(self):
        for script in ("project-setup.sh", "init.sh", "set_env.sh"):
            with self.subTest(script=script):
                self.assertScriptParses(ROOT / script)

    def test_tpu_env_agrees_with_server_defaults(self):
        # The directory name is a claim about tpu.env (NAMING.md). This asserts
        # the env file and the server actually agree, so the claim stays true.
        values = tpu_env()
        for key in (
            "MODEL_NAME", "INSTANCE_TYPE", "DTYPE", "KV_CACHE_DTYPE", "QUANT_MODE",
            "TORCH_PIP_SPEC", "TORCH_PYTHON_VERSION", "SERVICE_NAME",
                        # The AMI keys were NOT covered here, which is the pair most able to
            # break a launch on their own and the pair most likely to drift:
            # they changed together on 2026-08-27 and nothing would have noticed
            # if only one of them had.
            "DLAMI_SSM_PARAMETER", "DLAMI_NAME",
            # until 2026-08-27; if the env file and the server ever disagree
            # about the path again, the S3 sync backs up the wrong directory.
        ):
            with self.subTest(key=key):
                self.assertEqual(values[key], getattr(server, key))
        for key in ("TORCH_PORT", "MAX_MODEL_LEN"):
            with self.subTest(key=key):
                self.assertEqual(int(values[key]), getattr(server, key))
        self.assertEqual(
            int(values["TENSOR_PARALLEL_SIZE"]), server._gpu_count(server.INSTANCE_TYPE)
        )
        # Booleans are spelled as a string in the env file and a bool in the
        # server, so they need their own comparison rather than being skipped --
        # INT8_LM_HEAD went from off to on on 2026-08-26 and nothing was
        # asserting the two agreed.
        self.assertEqual(
            values["INT8_LM_HEAD"].lower() in ("1", "true", "yes"), server.INT8_LM_HEAD
        )

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
        # what happened during the t4g->g6 rename. Guard both copies.
        stem = f"{ROOT.name}-management"
        for prefix in (f".claude/skills/{stem}", f"skills/{stem}"):
            with self.subTest(copy=prefix):
                skill = ROOT / prefix / "SKILL.md"
                self.assertTrue(skill.is_file(), f"{prefix}/SKILL.md is missing")
                self.assertIn(
                    f"name: {stem}", skill.read_text(),
                    f"{prefix}/SKILL.md has a stale name",
                )
            # subTest keeps the loop going past a failure, so the wiped-directory
            # case this test exists to catch would reach filecmp.cmp and bury the
            # one useful message under a FileNotFoundError per source file.
            if not (ROOT / prefix / "mcp").is_dir():
                continue
            for source in SKILL_SOURCES:
                snapshot = ROOT / prefix / "mcp" / source
                with self.subTest(copy=prefix, source=source):
                    self.assertTrue(snapshot.is_file(), f"{snapshot} is missing — run `make skill`")
                    self.assertTrue(
                        filecmp.cmp(ROOT / source, snapshot, shallow=False),
                        f"{prefix}/mcp/{source} is stale — run `make skill`",
                    )


if __name__ == "__main__":
    unittest.main()

