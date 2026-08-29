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

# Spelled out rather than read from server, so that a change to the served
# checkpoint or the default size shows up as a test edit rather than passing
# vacuously against whatever the module happens to hold.
MODEL = "google/gemma-4-E2B-it"
DEFAULT_SIZE = "g5g.2xlarge"
SWAP_SIZE = "g5g.xlarge"        # the one size small enough to need a swapfile

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
            "create_g5g_instance", "list_g5g_instances", "start_g5g_instance",
            "stop_g5g_instance", "terminate_g5g_instance", "verify_gpu_arch",
            "deploy_torch_server", "get_install_progress", "get_torch_logs",
            "get_endpoint", "verify_model_health", "query_model", "save_hf_token",
            "check_g5g_quotas", "get_deployment_config", "get_help", "get_metrics",
        }
        self.assertEqual(set(self.tools), expected)

    def test_annotations(self):
        destructive = {
            name for name, tool in self.tools.items() if tool.annotations.destructiveHint
        }
        self.assertEqual(destructive, {"stop_g5g_instance", "terminate_g5g_instance"})
        for name, tool in self.tools.items():
            with self.subTest(tool=name):
                self.assertTrue(tool.title)
                self.assertTrue(tool.description)
                self.assertIsNotNone(tool.annotations)

    def test_launch_defaults_to_spot(self):
        for name in ("create_g5g_instance", "get_deployment_config"):
            with self.subTest(tool=name):
                schema = self.tools[name].inputSchema["properties"]
                self.assertTrue(schema["spot"]["default"])
                # There is no `serving` mode here: nothing is built, so there is
                # no stock-vs-build choice the vLLM sibling has to offer.
                self.assertNotIn("serving", schema)


class G5gTopologyTests(unittest.TestCase):
    """Instance-shape policy: which sizes exist, and what each one implies."""

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
            with self.subTest(instance_type=instance_type):
                self.assertTrue(server._is_g5g(instance_type))
                self.assertEqual(server._gpu_count(instance_type), gpus)
                self.assertEqual(server._host_memory_gb(instance_type), ram)

    def test_tensor_parallel_follows_gpu_count(self):
        self.assertEqual(server._tensor_parallel_size("g5g.2xlarge"), 1)
        self.assertEqual(server._tensor_parallel_size("g5g.16xlarge"), 2)

    def test_every_size_is_supported_and_the_two_smallest_need_swap(self):
        """The threshold is INCLUSIVE, and 2xlarge is the case that proves it.

        Two different pressures, both measured, and 16 GiB is short for both:

          * g5g.xlarge (8 GiB), 2026-08-13: without swap the kernel refuses to
            mmap the 10.2 GB checkpoint at all and systemd crash-loops.
          * g5g.2xlarge (16 GiB), 2026-08-26: mmaps fine, then `--ple-bits 8`
            is OOM-KILLED five times at 14.3 GB anon-rss, because
            quantize_ple_table upcasts the 4.70 GB PLE table while the full
            tree is resident. `< 16` excluded exactly this size, so the rig
            provisioned swap for the small host and skipped the one where the
            quantization path needs it.
        """
        for size in (SWAP_SIZE, "g5g.2xlarge"):
            with self.subTest(instance_type=size):
                server._validate_instance_type(size)   # must not raise
                self.assertTrue(server._needs_swap(size))
        for size in ("g5g.4xlarge", "g5g.8xlarge", "g5g.16xlarge", "g5g.metal"):
            with self.subTest(instance_type=size):
                server._validate_instance_type(size)
                self.assertFalse(server._needs_swap(size))

    def test_swap_block_is_rendered_for_the_default_size(self):
        # The default this rig actually launches. Before 2026-08-26 this
        # rendered no swapfile and ple_bits could not load.
        self.assertIn("swapfile", user_data(instance_type="g5g.2xlarge"))

    def test_non_g5g_rejected(self):
        for bad in ("g6.xlarge", "inf2.xlarge", "g5g.unknown"):
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
        self.assertIn('PY_BIN="$(cat /opt/torch-g5g/PYTHON_BIN)"', text)
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
        self.assertLess(text.index("chmod 600 /opt/torch-g5g/env"), text.index("HF_TOKEN="))

    def test_swap_block_uses_only_portable_flags(self):
        """`mkswap -q` is busybox; util-linux rejects it and `set -e` kills cloud-init.

        This was latent for as long as only g5g.xlarge rendered the block and
        nobody launched one. Making the threshold inclusive on 2026-08-26 pointed
        it at the DEFAULT size and the instance came up with an empty
        /opt/torch-g5g and no install log.
        """
        rendered = user_data(instance_type="g5g.2xlarge")
        # Executable lines only -- the comment above the fix quotes the bad flag
        # on purpose, and matching prose would fail vacuously.
        code = "\n".join(ln for ln in rendered.splitlines()
                          if not ln.lstrip().startswith("#"))
        self.assertIn("mkswap /swapfile", code)
        self.assertNotIn("mkswap -q", code)

    def test_swapfile_is_rendered_for_the_two_smallest_hosts(self):
        """2xlarge moved into this set on 2026-08-26, deliberately.

        It has exactly 16 GiB and the old `< 16` gate excluded it, so ple_bits
        was OOM-killed on the rig's own default size. See
        InstanceTypeTests.test_every_size_is_supported_and_the_two_smallest_need_swap.
        """
        for size in (SWAP_SIZE, "g5g.2xlarge"):
            with self.subTest(instance_type=size):
                rendered = user_data(size)
                for fragment in ("mkswap", "swapon /swapfile", "/etc/fstab"):
                    self.assertIn(fragment, rendered)
                self.assertShellParses(rendered, size)
        for size in ("g5g.4xlarge", "g5g.8xlarge", "g5g.16xlarge"):
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
        rendered = server._user_data("google/gemma-4-E2B-it", "g5g.2xlarge")
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
        rendered = server._user_data("google/gemma-4-E2B-it", "g5g.2xlarge")
        self.assertNotIn("import jax", rendered)
        self.assertNotIn("jax_openai_server.py", rendered)
        self.assertIn("torch_openai_server.py", rendered)

    def test_serve_argv_only_emits_flags_the_server_defines(self):
        # torch_openai_server.py defines exactly --model/--host/--port/--seq.
        # argparse exits 2 on an unknown flag, so an extra one here crash-loops
        # the unit with the reason only in journalctl.
        argv = server._serve_argv("google/gemma-4-E2B-it", "g5g.2xlarge")
        allowed = {"--model", "--host", "--port", "--seq"}
        flags = {tok for tok in argv.split() if tok.startswith("--")}
        self.assertEqual(flags, allowed)


class MetricsParsingTests(unittest.TestCase):
    """_parse_prom is pure, so the interesting parts pin offline."""

    EXPOSITION = "\n".join([
        "# HELP tpu_jax_precision_info Dtypes and quantisation resolved on device",
        "# TYPE tpu_jax_precision_info gauge",
        'tpu_jax_precision_info{model="google/gemma-4-E2B-it",compute_dtype="float16",'
        'quant_mode="fp16",kv_cache_dtype="float16",kv_cache_requested="auto",'
        'ple_bits="0",int8_lm_head="false",pre_ampere="true"} 1',
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
        self.assertEqual(self.precision["compute_dtype"], "float16")
        self.assertEqual(self.precision["kv_cache_dtype"], "float16")
        self.assertEqual(self.precision["kv_cache_requested"], "auto")
        self.assertEqual(self.precision["quant_mode"], "fp16")
        self.assertEqual(self.precision["pre_ampere"], "true")

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
        # Two independent requirements, and a name filter only pins one of them.
        # arm64: the legacy tips-tree rigs hardcode an x86_64 DLAMI id that
        # cannot boot on Graviton2. NVIDIA driver: AWS also ships ARM64 DLAMIs
        # for Graviton CPU inference, which boot fine on a G5g with no GPU.
        self.assertIn("aws ssm get-parameter", run(server.get_deployment_config()))
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
        rejected = run(server.get_deployment_config(instance_type="g6.xlarge"))
        self.assertTrue(rejected.startswith("❌"))
        ok = run(server.get_deployment_config(instance_type=SWAP_SIZE))
        self.assertFalse(ok.startswith("❌"))
        self.assertIn("mkswap", user_data_from_config(ok))


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
        """The inverse of the JAX rig, and the reason this rig can exist.

        torch comes from the DLAMI, already built with sm_75 in its arch list.
        Upstream PyPI aarch64 wheels do NOT carry Turing, so a `pip install
        torch` here would either fall back to CPU silently or demand a
        from-source build. The pip spec must therefore carry the serving
        libraries and nothing else.
        """
        self.assertNotIn("torch", server.TORCH_PIP_SPEC)
        self.assertIn("transformers", server.TORCH_PIP_SPEC)
        self.assertNotIn("==", server.TORCH_PIP_SPEC)
        # 3.12 is what the Ubuntu 24.04 PyTorch DLAMI ships as its interpreter.
        self.assertEqual(server.TORCH_PYTHON_VERSION, "3.12")

    def test_ami_is_a_pytorch_image_not_the_base_one(self):
        """Inverted from the JAX sibling, on purpose.

        There the PyTorch DLAMI was GBs of unused image. Here it is the entire
        point: `docs/turing-aarch64-gap.md` measured `torch 2.12.0+cu132` on
        this line carrying `sm_75`, and that prebuilt Turing support is what
        removes the build from this rig. Do not "correct" this back to the base
        driver image -- that would put us on upstream aarch64 wheels, which omit
        Turing.
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
        self.assertIn("/arm64/", server.DLAMI_SSM_PARAMETER)
        self.assertIn("nvidia-driver-gpu", server.DLAMI_SSM_PARAMETER)

    def test_name_fallback_matches_the_base_images_it_now_targets(self):
        # _resolve_ami takes newest-by-CreationDate, so a loose pattern silently
        # selects the wrong image. This rig needs the PyTorch line specifically:
        # the base driver image and the driverless Graviton-CPU image must both
        # fail to match, for opposite reasons.
        import fnmatch
        current = "Deep Learning ARM64 AMI OSS Nvidia Driver GPU PyTorch 2.12 (Ubuntu 24.04) 20260724"
        base = "Deep Learning ARM64 Base OSS Nvidia Driver GPU AMI (Ubuntu 26.04) 20260825"
        driverless = "Deep Learning ARM64 AMI Graviton (Ubuntu 22.04) 20260501"
        # The PyTorch line is what this rig wants -- torch prebuilt with sm_75.
        self.assertTrue(fnmatch.fnmatch(current, server.DLAMI_NAME))
        # The base driver image must NOT match: it has no torch, and pip would
        # then pull an aarch64 wheel with no Turing support.
        self.assertFalse(fnmatch.fnmatch(base, server.DLAMI_NAME))
        self.assertFalse(fnmatch.fnmatch(driverless, server.DLAMI_NAME))

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
        # create_g5g_instance launched 100, so the copy-pasteable repro command
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


class CompilationCacheTests(BashSyntaxMixin, unittest.TestCase):
    """Persisting /opt/jax-cache is opt-in and must stay a no-op by default."""

    def _with_uri(self, uri):
        original = server.JAX_CACHE_S3_URI
        server.JAX_CACHE_S3_URI = uri
        try:
            return user_data()
        finally:
            server.JAX_CACHE_S3_URI = original

    def test_default_is_off_and_renders_nothing(self):
        # The default rendering must be exactly what this rig shipped before:
        # no bucket, no IAM, no units. Opting in is the operator's call.
        self.assertEqual(server.JAX_CACHE_S3_URI, "")
        text = self._with_uri("")
        self.assertNotIn("aws s3 sync", text)
        self.assertNotIn("-cache.timer", text)

    def test_enabling_it_renders_valid_bash(self):
        text = self._with_uri("s3://bucket/jax-cache/")
        self.assertShellParses(text, "cloud-init with cache sync")
        inner = text.split("<<'INSTEOF'\n", 1)[1].split("\nINSTEOF", 1)[0]
        self.assertShellParses(inner, "install.sh with cache restore")

    def test_restore_cannot_kill_the_install_on_a_cold_bucket(self):
        # install.sh runs under `set -e`, and the very first launch syncs from a
        # prefix that does not exist yet. Without `|| true` that is the mkswap
        # failure again: cloud-init dies and nothing installs.
        text = self._with_uri("s3://bucket/jax-cache/")
        restore = [ln for ln in text.splitlines() if "aws s3 sync s3://" in ln]
        self.assertTrue(restore, "no restore line rendered")
        self.assertTrue(all("|| true" in ln for ln in restore), restore)

    def test_upload_is_on_a_timer_not_at_shutdown(self):
        # A spot reclamation gives ~2 minutes and does not reliably run
        # ExecStopPost, so a shutdown hook would lose exactly the compiles that
        # the 2026-08-25 reclamation would have cost.
        text = self._with_uri("s3://bucket/jax-cache/")
        self.assertIn("OnUnitActiveSec=", text)
        self.assertNotIn("ExecStopPost", text)

    def test_restore_lands_before_install_done(self):
        # INSTALL_DONE is the readiness signal deploy_torch_server waits on; a
        # cache restored after it would race the first request.
        #
        # Anchored on the `touch`, not on the bare string: an apt comment 90
        # lines earlier also says INSTALL_DONE, and matching that would make this
        # pass no matter where the restore landed.
        text = self._with_uri("s3://bucket/jax-cache/")
        self.assertLess(text.index("stage cache-restore"),
                        text.index(f"touch {server.APP_DIR}/INSTALL_DONE"))


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
            # JAX_COMPILATION_CACHE_DIR is here because it silently did nothing
            # until 2026-08-27; if the env file and the server ever disagree
            # about the path again, the S3 sync backs up the wrong directory.
            "JAX_CACHE_S3_URI",
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
        # what happened during the t4g->g5g rename. Guard both copies.
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

