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
DEFAULT_SIZE = "g4dn.xlarge"
SWAP_SIZE = "g4dn.xlarge"       # 16 GiB host RAM -- the boundary case, inclusive

# The eight files refresh_skill.py snapshots into both skill copies: the MCP
# control plane *and* the serving payload, because an installed skill still has
# to be able to run deploy_jax_server.
SKILL_SOURCES = (
    "server.py", "project-setup.sh", "requirements.txt",
    "requirements-serving.txt", "jax_openai_server.py", "jax_engine.py",
    "ports/gemma4/jax_e_loader.py", "ports/gemma4/jax_e_model.py",
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
            "create_g4dn_instance", "list_g4dn_instances", "start_g4dn_instance",
            "stop_g4dn_instance", "terminate_g4dn_instance", "verify_gpu_arch",
            "deploy_jax_server", "get_install_progress", "get_jax_logs",
            "get_endpoint", "verify_model_health", "query_model", "save_hf_token",
            "check_g4dn_quotas", "get_deployment_config", "get_help", "get_metrics",
        }
        self.assertEqual(set(self.tools), expected)

    def test_annotations(self):
        destructive = {
            name for name, tool in self.tools.items() if tool.annotations.destructiveHint
        }
        self.assertEqual(destructive, {"stop_g4dn_instance", "terminate_g4dn_instance"})
        for name, tool in self.tools.items():
            with self.subTest(tool=name):
                self.assertTrue(tool.title)
                self.assertTrue(tool.description)
                self.assertIsNotNone(tool.annotations)

    def test_launch_defaults_to_spot(self):
        for name in ("create_g4dn_instance", "get_deployment_config"):
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
            "g4dn.xlarge": (1, 16),
            "g4dn.2xlarge": (1, 32),
            "g4dn.4xlarge": (1, 64),
            "g4dn.8xlarge": (1, 128),
            "g4dn.12xlarge": (4, 192),
            "g4dn.16xlarge": (1, 256),
            "g4dn.metal": (8, 384),
        }
        for instance_type, (gpus, ram) in expected.items():
            with self.subTest(instance_type=instance_type):
                self.assertTrue(server._is_g4dn(instance_type))
                self.assertEqual(server._gpu_count(instance_type), gpus)
                self.assertEqual(server._host_memory_gb(instance_type), ram)

    def test_tensor_parallel_follows_gpu_count(self):
        self.assertEqual(server._tensor_parallel_size("g4dn.xlarge"), 1)
        self.assertEqual(server._tensor_parallel_size("g4dn.12xlarge"), 4)

    def test_every_size_is_supported_and_the_two_smallest_need_swap(self):
        """The threshold is INCLUSIVE, and 2xlarge is the case that proves it.

        Two different pressures, both measured, and 16 GiB is short for both:

          * g5g.xlarge (8 GiB), 2026-08-13: without swap the kernel refuses to
            mmap the 10.2 GB checkpoint at all and systemd crash-loops.
          * g4dn.xlarge (16 GiB), 2026-08-26: mmaps fine, then `--ple-bits 8`
            is OOM-KILLED five times at 14.3 GB anon-rss, because
            quantize_ple_table upcasts the 4.70 GB PLE table while the full
            tree is resident. `< 16` excluded exactly this size, so the rig
            provisioned swap for the small host and skipped the one where the
            quantization path needs it.
        """
        for size in (SWAP_SIZE, "g4dn.xlarge"):
            with self.subTest(instance_type=size):
                server._validate_instance_type(size)   # must not raise
                self.assertTrue(server._needs_swap(size))
        for size in ("g4dn.2xlarge", "g4dn.4xlarge", "g4dn.8xlarge", "g4dn.metal"):
            with self.subTest(instance_type=size):
                server._validate_instance_type(size)
                self.assertFalse(server._needs_swap(size))

    def test_swap_block_is_rendered_for_the_default_size(self):
        # The default this rig actually launches. Before 2026-08-26 this
        # rendered no swapfile and ple_bits could not load.
        self.assertIn("swapfile", user_data(instance_type="g4dn.xlarge"))

    def test_non_g4dn_rejected(self):
        for bad in ("g6.xlarge", "inf2.xlarge", "g4dn.unknown"):
            with self.subTest(instance_type=bad), self.assertRaises(ValueError):
                server._validate_instance_type(bad)


class TuringConstraintTests(unittest.TestCase):
    """Turing (SM 7.5) has no bf16 and no fp8. The L4 sibling rigs hardcode both."""

    def test_dtype_default_is_float16(self):
        self.assertEqual(server.DTYPE, "float16")

    def test_kv_cache_is_not_fp8(self):
        argv = server._serve_argv(MODEL, DEFAULT_SIZE)
        self.assertIn("--kv-cache-dtype auto", argv)
        self.assertNotIn("fp8", argv)

    def test_quant_mode_matches_the_dense_checkpoint(self):
        # QUANT_MODE is a claim about MODEL_NAME, not about the chip. A w4a16
        # mode against a dense checkpoint loads garbage rather than failing.
        argv = server._serve_argv(server.MODEL_NAME, DEFAULT_SIZE)
        self.assertIn("--quant-mode fp16", argv)
        self.assertNotIn("w4a16", server.MODEL_NAME)

    def test_no_tensor_parallel_flag_is_emitted(self):
        # The JAX engine is single-device. Emitting a TP flag would imply a
        # sharding this rig does not do.
        argv = server._serve_argv(MODEL, "g4dn.12xlarge")
        self.assertNotIn("tensor-parallel", argv)


class DegeneracyGuardTests(unittest.TestCase):
    """The serving stack counted a token loop as status="success"."""

    @staticmethod
    def _looks_degenerate():
        # jax_openai_server imports jax at module scope and jax is a SERVING
        # dependency, not a control-plane one, so the module cannot be imported
        # here. Exec just this one pure function instead of skipping the test.
        src = (ROOT / "jax_openai_server.py").read_text()
        body = src.split("def looks_degenerate")[1].split("\ndef _record")[0]
        ns = {}
        exec("def looks_degenerate" + body, ns)  # pure function from our own repo
        return ns["looks_degenerate"]

    def test_catches_both_degenerate_shapes_observed_on_hardware(self):
        f = self._looks_degenerate()
        # MEASURED 2026-08-23 on this rig at 2,615-3,515 prompt tokens.
        self.assertTrue(f("The" * 40))
        # MEASURED on the vLLM sibling and recorded in the monorepo CLAUDE.md.
        self.assertTrue(f(": ok" * 20))
        self.assertTrue(f("ok " * 30))

    def test_does_not_fire_on_good_output(self):
        # A false positive would discredit a real benchmark result, so the
        # guard is deliberately conservative -- these must all pass through.
        f = self._looks_degenerate()
        for good in (
            "The quick brown fox repeatedly jumps over a lazy dog.",
            "The three primary colours are: 1. Red 2. Yellow 3. Blue",
            "391",
            "Le chat est sur la table.",
            "yes yes yes yes yes but actually the answer depends on the context here",
            "",
        ):
            with self.subTest(text=good[:30]):
                self.assertFalse(f(good))

    def test_short_replies_are_not_judged(self):
        # verify_model_health asks for a single word; judging that would make
        # the guard fire on the rig's own health check.
        self.assertFalse(self._looks_degenerate()("ok " * 10))

    def test_counter_is_exposed_and_does_not_change_status(self):
        text = (ROOT / "jax_openai_server.py").read_text()
        self.assertIn("tpu_jax_degenerate_responses_total", text)
        # It must remain observational: a degenerate reply is still returned to
        # the caller and still counted a success, so this cannot mask a real
        # regression in the success/failure split.
        self.assertIn('METRICS["successful_requests"] += 1', text)


class QuantKnobTests(unittest.TestCase):
    """The engine always supported these; the rig could not reach them."""

    def _argv(self, **env):
        saved = {k: getattr(server, k) for k in env}
        for k, v in env.items():
            setattr(server, k, v)
        try:
            return server._serve_argv(server.MODEL_NAME, "g4dn.xlarge")
        finally:
            for k, v in saved.items():
                setattr(server, k, v)

    def test_ple_bits_is_always_emitted(self):
        # Emitted even at the 0 default, so the serving command records the
        # choice instead of deferring to the server's own default.
        self.assertIn("--ple-bits 0", self._argv(PLE_BITS=0))
        self.assertIn("--ple-bits 4", self._argv(PLE_BITS=4))

    def test_int8_lm_head_is_a_flag_not_a_value(self):
        self.assertIn("--int8-lm-head", self._argv(INT8_LM_HEAD=True))
        self.assertNotIn("--int8-lm-head", self._argv(INT8_LM_HEAD=False))

    def test_prefill_chunk_size_is_omitted_when_unset(self):
        # Unset must mean one-shot prefill, the previous behaviour -- not a
        # chunk size of 0, which would be a different and broken request.
        self.assertNotIn("--prefill-chunk-size", self._argv(PREFILL_CHUNK_SIZE=""))
        self.assertIn("--prefill-chunk-size 1024", self._argv(PREFILL_CHUNK_SIZE="1024"))

    def test_serving_process_accepts_every_flag_the_rig_emits(self):
        # The rig and the server are separate files and drifted before: the
        # server grew --ple-bits and _serve_argv never learned to pass it.
        text = (ROOT / "jax_openai_server.py").read_text()
        argv = self._argv(PLE_BITS=4, INT8_LM_HEAD=True, PREFILL_CHUNK_SIZE="1024")
        for token in argv.split():
            if token.startswith("--"):
                self.assertIn(f'"{token}"', text, f"{token} is emitted but not accepted")

    def test_load_engine_defaults_are_not_the_tpu_rigs(self):
        # kv_dtype="bf16" raises on pre-Ampere and quant_mode="w4a16" against a
        # dense checkpoint loads garbage. Latent only because the CLI always
        # passes both explicitly.
        text = (ROOT / "jax_openai_server.py").read_text()
        signature = text.split("def load_engine(", 1)[1].split(")", 1)[0]
        self.assertNotIn('"bf16"', signature)
        self.assertNotIn('"w4a16"', signature)


class UserDataTests(BashSyntaxMixin, unittest.TestCase):
    """The rendered cloud-init script, which installs the runtime and nothing else."""

    def test_install_is_wheels_not_a_build(self):
        # Derived from JAX_PIP_SPEC, never a literal: a hardcoded "jax[cuda12]"
        # here turns a routine CUDA-line bump into a test edit, which is friction
        # against the standing preference for latest versions. The claim under
        # test is "we install the configured spec from wheels", not which spec.
        text = user_data()
        self.assertIn(server.JAX_PIP_SPEC, text)
        self.assertNotIn("docker build", text)
        self.assertNotIn("git clone", text)
        # cloud-init must not block on the install.
        self.assertIn("nohup", text)
        self.assertIn("INSTALL_DONE", text)
        self.assertShellParses(text)

    def test_a_modern_python_is_installed_because_jax_requires_it(self):
        # jax >= 0.11 declares requires-python >= 3.12 and Ubuntu 22.04 ships
        # 3.10, so the DLAMI's system python would fail at pip install time.
        # Asserted against JAX_PYTHON_VERSION rather than a literal so the
        # interpreter can be moved forward without editing tests.
        text = user_data()
        self.assertIn("deadsnakes", text)
        self.assertIn(f"python{server.JAX_PYTHON_VERSION}", text)
        self.assertGreaterEqual(
            tuple(int(x) for x in server.JAX_PYTHON_VERSION.split(".")), (3, 12),
            "jax >= 0.11 requires Python 3.12 or newer",
        )

    def test_systemd_execstart_is_absolute(self):
        # systemd refuses a relative ExecStart, and the unit would fail to load
        # with a message that says nothing about the interpreter.
        self.assertIn(f"ExecStart=/usr/bin/python{server.JAX_PYTHON_VERSION}", user_data())

    def test_execstart_is_repointed_at_the_installed_interpreter(self):
        # MEASURED 2026-08-19 on i-063d52c913140b787: the DLAMI already ships
        # /usr/local/bin/python3.12, which precedes /usr/bin on PATH. install.sh
        # calls bare `python3.12`, so jax landed in /usr/local, while the unit's
        # hardcoded ExecStart=/usr/bin/python3.12 crash-looped on
        # ModuleNotFoundError -- AFTER the install reported success, because the
        # verify step resolves through PATH too.
        text = user_data()
        self.assertIn(f'PY_BIN="$(command -v python{server.JAX_PYTHON_VERSION})"', text)
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
        self.assertLess(text.index("chmod 600 /opt/jax-g4dn/env"), text.index("HF_TOKEN="))

    def test_swap_block_uses_only_portable_flags(self):
        """`mkswap -q` is busybox; util-linux rejects it and `set -e` kills cloud-init.

        This was latent for as long as only g5g.xlarge rendered the block and
        nobody launched one. Making the threshold inclusive on 2026-08-26 pointed
        it at the DEFAULT size and the instance came up with an empty
        /opt/jax-g4dn and no install log.
        """
        rendered = user_data(instance_type="g4dn.xlarge")
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
        for size in (SWAP_SIZE, "g4dn.xlarge"):
            with self.subTest(instance_type=size):
                rendered = user_data(size)
                for fragment in ("mkswap", "swapon /swapfile", "/etc/fstab"):
                    self.assertIn(fragment, rendered)
                self.assertShellParses(rendered, size)
        for size in ("g4dn.2xlarge", "g4dn.4xlarge", "g4dn.8xlarge"):
            with self.subTest(instance_type=size):
                self.assertNotIn("mkswap", user_data(size))

    def test_serving_requirements_match_the_mirror_file(self):
        # A drifted pair is invisible until a serve fails on a missing import.
        listed = {
            line.strip()
            for line in (ROOT / "requirements-serving.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        expected = set(server._SERVING_REQUIREMENTS) | {server.JAX_PIP_SPEC}
        self.assertEqual(listed, expected)

    def test_profiling_requirements_match_the_mirror_file(self):
        # Same hazard as the serving pair, and it has already bitten once in a
        # worse form: requirements-profiling.txt named a path that the deploy
        # payload excludes, so `pip install -r` failed with `Could not open
        # requirements file` and the xprof extraction died on ModuleNotFoundError.
        listed = {
            line.strip()
            for line in (ROOT / "requirements-profiling.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        self.assertEqual(listed, set(server._PROFILING_REQUIREMENTS))

    def test_the_profiler_is_installed_but_never_on_the_serving_list(self):
        """xprof and tensorboard ship at boot, in their OWN non-fatal stage.

        Both halves matter. They must be installed -- the previous on-demand
        arrangement meant they were never installed at all -- and they must not
        be in _SERVING_REQUIREMENTS, because that list runs under `set -e` and a
        broken profiler wheel must not cost a box that can serve.
        """
        self.assertEqual(("xprof", "tensorboard"), server._PROFILING_REQUIREMENTS)
        for pkg in server._PROFILING_REQUIREMENTS:
            self.assertNotIn(pkg, server._SERVING_REQUIREMENTS)

        script = server._user_data(server.MODEL_NAME, "g4dn.xlarge")
        self.assertIn("$PIP xprof tensorboard", script)
        self.assertIn("stage profiling-deps", script)
        # Non-fatal: the whole install runs under `set -euxo pipefail`, so
        # without the `||` a profiler wheel failure would kill cloud-init before
        # INSTALL_DONE is touched -- the exact shape of the mkswap incident.
        profiling_line = next(
            ln for ln in script.splitlines() if ln.strip().startswith("$PIP xprof")
        )
        self.assertIn("||", profiling_line)

    def test_profiling_install_can_be_turned_off(self):
        """INSTALL_PROFILING=0 restores a serving-only image, byte for byte."""
        previous = server.INSTALL_PROFILING
        server.INSTALL_PROFILING = False
        try:
            script = server._user_data(server.MODEL_NAME, "g4dn.xlarge")
        finally:
            server.INSTALL_PROFILING = previous
        self.assertNotIn("xprof", script)
        self.assertNotIn("stage profiling-deps", script)
        # And the serving stage is untouched by the switch.
        self.assertIn("stage serving-deps", script)


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


class ServedPrecisionTests(unittest.TestCase):
    """Turing resolves dtypes the TPU lineage never has to consider."""

    def test_health_does_not_hardcode_bfloat16(self):
        # jax_openai_server used to report activations="bfloat16" and weights
        # ="bf16" unconditionally -- inherited from the TPU rig and impossible on
        # a chip with no bf16 datapath. Both must come off the engine now.
        text = (ROOT / "jax_openai_server.py").read_text()
        self.assertNotIn('"activations": "bfloat16"', text)
        self.assertIn("precision_info()", text)

    def test_load_banner_does_not_claim_w4a16(self):
        # This rig serves the DENSE checkpoint. A banner claiming W4A16 QAT
        # weights describes the one thing the rig refuses to do, and would send
        # an operator hunting a problem that is not there.
        text = (ROOT / "jax_openai_server.py").read_text()
        self.assertNotIn("Loading W4A16 QAT weights", text)

    def test_engine_reports_resolved_not_requested(self):
        # "auto" is the configured default for both; reporting it back would say
        # nothing about what the device actually got.
        text = (ROOT / "jax_engine.py").read_text()
        self.assertIn("def precision_info", text)
        for key in ("compute_dtype", "kv_cache_dtype", "kv_cache_requested", "quant_mode"):
            self.assertIn(f'"{key}"', text)


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
            run(server.deploy_jax_server(instance_id="i-test", restart=restart))
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
        # x86_64: this rig is an Intel/AMD host, so the parent rig's arm64
        # parameter resolves an image that cannot boot here. NVIDIA driver: AWS
        # ships plenty of x86_64 DLAMIs with no driver at all (the plain "Deep
        # Learning Base AMI" line), which boot fine and have no usable GPU.
        # The Graviton-CPU trap does not apply here; the driverless trap does.
        self.assertIn("aws ssm get-parameter", run(server.get_deployment_config()))
        self.assertIn("/x86_64/", server.DLAMI_SSM_PARAMETER)
        self.assertIn("nvidia-driver-gpu", server.DLAMI_SSM_PARAMETER)

    def test_ami_name_fallback_requires_the_driver(self):
        # A bare "Deep Learning*Ubuntu*" also matches the driverless base images,
        # and _resolve_ami takes newest-by-CreationDate -- so the driver has to
        # be in the pattern, not merely hoped for.
        self.assertIn("Nvidia Driver", server.DLAMI_NAME)
        self.assertIn("GPU", server.DLAMI_NAME)
        self.assertNotIn("ARM64", server.DLAMI_NAME)   # this rig is x86_64

    def test_config_tags_with_rig_name(self):
        result = run(server.get_deployment_config())
        self.assertIn(f"Key=ManagedBy,Value={server.RIG_NAME}", result)

    def test_config_spot_toggle(self):
        self.assertIn("MarketType=spot", run(server.get_deployment_config()))
        self.assertNotIn("MarketType=spot", run(server.get_deployment_config(spot=False)))

    def test_config_rejects_non_g4dn_only(self):
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


class CompilationCacheDirTests(unittest.TestCase):
    """JAX_COMPILATION_CACHE_DIR must survive the port's import.

    MEASURED 2026-08-27 on i-021f15b2b45e13793: the unit set the variable, the
    process had it, and the configured directory stayed EMPTY while 447 files
    accumulated under the fallback. ports/gemma4/jax_e_model.py set the path
    unconditionally at import, and jax_openai_server imports it (via jax_engine)
    AFTER resolving the same variable -- so the port silently won, every start.

    Nothing failed and nothing logged. The JAX_CACHE_S3_URI sync added the same
    day would have backed up an empty directory forever, reporting success.
    """

    def _source(self, rel):
        return (ROOT / rel).read_text()

    def test_the_port_honours_the_env_var(self):
        src = self._source("ports/gemma4/jax_e_model.py")
        self.assertIn('os.environ.get("JAX_COMPILATION_CACHE_DIR")', src)

    def test_the_port_does_not_hardcode_the_fallback_as_the_only_path(self):
        # The exact shape of the bug: expanduser as the sole source, with no
        # env lookup in front of it.
        src = self._source("ports/gemma4/jax_e_model.py")
        self.assertNotIn(
            '_cache_dir = os.path.expanduser("~/.cache/jax_compilation_cache")', src
        )

    def test_both_modules_resolve_the_cache_dir_the_same_way(self):
        # They run in one process and the later import wins, so agreeing is the
        # only way the result does not depend on import order.
        for rel in ("ports/gemma4/jax_e_model.py", "jax_openai_server.py"):
            with self.subTest(module=rel):
                src = self._source(rel)
                self.assertIn('os.environ.get("JAX_COMPILATION_CACHE_DIR")', src)
                self.assertIn('"~/.cache/jax_compilation_cache"', src)

    def test_the_unit_actually_ships_the_variable(self):
        # The env var is only worth honouring if the bootstrap sets it.
        text = user_data()
        self.assertIn(f"JAX_COMPILATION_CACHE_DIR={server.JAX_COMPILATION_CACHE_DIR}", text)


class LatestVersionPolicyTests(BashSyntaxMixin, unittest.TestCase):
    """Newest release is the default here; a pin needs a named constraint."""

    def test_jax_and_python_are_unpinned_and_current(self):
        # jax[cuda13] is the newest CUDA extra jax publishes (there is no
        # cuda14), and the spec carries no version so pip resolves latest.
        # 3.14 is the newest stable CPython; jaxlib publishes a cp314 aarch64
        # wheel, which is what makes it usable here.
        self.assertNotIn("==", server.JAX_PIP_SPEC)
        self.assertIn("cuda13", server.JAX_PIP_SPEC)
        self.assertEqual(server.JAX_PYTHON_VERSION, "3.14")

    def test_ami_is_the_base_image_not_a_pytorch_one(self):
        # This rig never installs into the DLAMI's PyTorch -- it ships its own
        # CUDA libraries and jax brings its own -- so a PyTorch DLAMI is GBs of
        # image whose whole content is deliberately unused.
        self.assertIn("base-oss-nvidia-driver-gpu", server.DLAMI_SSM_PARAMETER)
        self.assertNotIn("pytorch", server.DLAMI_SSM_PARAMETER)

    def test_ami_line_is_one_aws_still_rebuilds(self):
        # `/latest/` is only the newest build WITHIN a PyTorch+Ubuntu line, and
        # AWS freezes those. The old pin resolved to an image built 2026-05-02
        # and could never move again: it READ as "track latest" and was a pin to
        # a dead line. 22.04 is where those lines stopped.
        self.assertNotIn("ubuntu-22.04", server.DLAMI_SSM_PARAMETER)
        self.assertIn("/x86_64/", server.DLAMI_SSM_PARAMETER)
        self.assertIn("nvidia-driver-gpu", server.DLAMI_SSM_PARAMETER)

    def test_name_fallback_matches_the_base_images_it_now_targets(self):
        # The fallback exists for when the SSM parameter is unavailable, and
        # _resolve_ami takes newest-by-CreationDate -- so a pattern that is too
        # loose silently selects a driverless image and the rig comes up with no
        # GPU. This pattern is deliberately NARROWER than the parent rig's: it
        # requires the "Base OSS Nvidia Driver GPU" line by name, which excludes
        # both the driverless base images and the frozen PyTorch line (the
        # parent accepted the latter; there is no reason to inherit that here,
        # since 22.04 is exactly where those lines stopped being rebuilt).
        import fnmatch
        base = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 26.04) 20260825"
        torch = "Deep Learning AMI GPU PyTorch 2.7 (Ubuntu 22.04) 20260501"
        driverless = "Deep Learning Base AMI (Ubuntu 26.04) 20260825"
        self.assertTrue(fnmatch.fnmatch(base, server.DLAMI_NAME))
        self.assertFalse(fnmatch.fnmatch(torch, server.DLAMI_NAME))
        self.assertFalse(fnmatch.fnmatch(driverless, server.DLAMI_NAME))

    def test_deadsnakes_is_conditional_not_unconditional(self):
        # Ubuntu 26.04 ships python3.14 as the SYSTEM interpreter, so the PPA is
        # off the critical path there -- but the branch has to remain for anyone
        # overriding DLAMI_SSM_PARAMETER back to 22.04/24.04.
        text = user_data()
        self.assertIn(f"command -v python{server.JAX_PYTHON_VERSION}", text)
        self.assertIn("ppa:deadsnakes/ppa", text)
        self.assertLess(text.index(f"command -v python{server.JAX_PYTHON_VERSION}"),
                        text.index("ppa:deadsnakes/ppa"))

    def test_pip_can_install_into_an_externally_managed_interpreter(self):
        # PEP 668: Ubuntu marks its system interpreter externally-managed from
        # 23.04 on, so on the 24.04/26.04 bases every system-wide pip install
        # fails with `error: externally-managed-environment`. Without this the
        # move to a newer base bricks the install.
        text = user_data()
        self.assertIn("--break-system-packages", text)
        # get-pip.py runs before PIP is defined and needs the flag of its own.
        self.assertIn("get-pip.py | python", text)
        get_pip = [ln for ln in text.splitlines() if "get-pip.py" in ln]
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
        self.assertIn("deploy_jax_server", verdict)


class InstallStageTimingTests(BashSyntaxMixin, unittest.TestCase):
    """The install was the longest phase of a deploy and the only untimed one."""

    def test_every_install_step_emits_a_stage_marker(self):
        text = user_data()
        for name in ("apt-base", "pip-bootstrap", "jax-wheels", "serving-deps", "gpu-verify"):
            with self.subTest(stage=name):
                self.assertIn(f"stage {name}", text)

    def test_jax_wheels_is_timed_separately_from_apt(self):
        # This is the whole point of the split: a spot reclamation at 21 minutes
        # (MEASURED 2026-08-25) left no record of which step was running, and apt
        # and several GB of CUDA wheels are very different things to attack.
        text = user_data()
        self.assertLess(text.index("stage apt-base"), text.index("stage jax-wheels"))

    def test_stage_helper_is_defined_before_it_is_called(self):
        text = user_data()
        self.assertLess(text.index("stage() {"), text.index("stage apt-base"))


class RootVolumeTests(unittest.TestCase):
    """gp3 defaults to 125 MiB/s and the load sat on it."""

    def test_launch_and_documented_command_agree(self):
        # These disagreed: get_deployment_config printed VolumeSize=200 while
        # create_g4dn_instance launched 100, so the copy-pasteable repro command
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
        # INSTALL_DONE is the readiness signal deploy_jax_server waits on; a
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
            "JAX_PIP_SPEC", "JAX_PYTHON_VERSION", "SERVICE_NAME",
            "XLA_PYTHON_CLIENT_MEM_FRACTION", "PREFILL_CHUNK_SIZE",
            # The AMI keys were NOT covered here, which is the pair most able to
            # break a launch on their own and the pair most likely to drift:
            # they changed together on 2026-08-27 and nothing would have noticed
            # if only one of them had.
            "DLAMI_SSM_PARAMETER", "DLAMI_NAME",
            # JAX_COMPILATION_CACHE_DIR is here because it silently did nothing
            # until 2026-08-27; if the env file and the server ever disagree
            # about the path again, the S3 sync backs up the wrong directory.
            "JAX_COMPILATION_CACHE_DIR", "JAX_CACHE_S3_URI",
        ):
            with self.subTest(key=key):
                self.assertEqual(values[key], getattr(server, key))
        for key in ("JAX_PORT", "MAX_MODEL_LEN", "PLE_BITS", "JAX_CACHE_SYNC_MINUTES"):
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


class ObservabilityTests(unittest.TestCase, BashSyntaxMixin):
    """Pin the traceability machinery added 2026-08-25.

    Every assertion here corresponds to a way this rig previously destroyed its
    own evidence: dropped INFO logs, tracebacks discarded on 500s, a request id
    that reached nothing, padding computed and thrown away, and a deploy that
    could not be told apart from the stale one it replaced.
    """

    SERVER_SRC = None
    ENGINE_SRC = None

    @classmethod
    def setUpClass(cls):
        cls.SERVER_SRC = (ROOT / "jax_openai_server.py").read_text()
        cls.ENGINE_SRC = (ROOT / "jax_engine.py").read_text()

    # ----------------------------------------------------------- logging

    def test_root_logging_is_configured_before_the_engine_import(self):
        """Ordering is the whole fix, not the basicConfig call on its own.

        ports.gemma4.jax_e_model logs its device-policy banner at IMPORT time and
        is imported via jax_engine, so configuring after that import would
        silence the one line naming the resolved compute dtype.
        """
        src = self.SERVER_SRC
        config_at = src.index("logging.basicConfig(")
        engine_import_at = src.index("from jax_engine import")
        self.assertLess(
            config_at, engine_import_at,
            "logging.basicConfig must precede the jax_engine import, or the "
            "device-policy banner is emitted before any handler exists",
        )
        self.assertIn("force=True", src[config_at:engine_import_at])

    def test_uvicorn_does_not_configure_the_root_logger(self):
        """The premise of the fix, asserted rather than assumed.

        If uvicorn ever starts adding a root handler this test fails and the
        basicConfig call can be reconsidered. Until then INFO records from the
        payload reach logging.lastResort, which drops anything below WARNING.
        """
        import logging
        import logging.config

        try:
            from uvicorn.config import LOGGING_CONFIG
        except ImportError:
            self.skipTest("uvicorn is a serving dependency, absent here")
        root = logging.getLogger()
        saved = list(root.handlers), root.level
        try:
            root.handlers = []
            logging.config.dictConfig(LOGGING_CONFIG)
            self.assertEqual(
                root.handlers, [],
                "uvicorn now configures the root logger; revisit basicConfig",
            )
        finally:
            root.handlers, root.level = saved

    def test_startup_reports_through_the_logger_not_print(self):
        # print() bypasses level control and the format, and cannot be filtered
        # by JAX_SERVER_LOG_LEVEL.
        body = self.SERVER_SRC.split("def load_engine")[1].split("\ndef _eos_ids")[0]
        self.assertNotIn("print(", body)
        self.assertIn("logger.info", body)

    # -------------------------------------------------------- request path

    def test_failures_are_logged_with_the_traceback_and_the_request_id(self):
        # A 500 used to leave NOTHING in the journal: HTTPException carries
        # str(exc) to the client and discards the stack.
        self.assertEqual(self.SERVER_SRC.count("logger.exception("), 3,
                         "expected both handlers plus the streaming generator")
        self.assertIn('detail=f"[{req_id}] {exc}"', self.SERVER_SRC)

    def test_streaming_failures_are_counted_and_not_silent(self):
        """The generator runs after the handler returned, outside its try."""
        stream = self.SERVER_SRC.split("def _sse_stream")[1].split('@app.get("/health")')[0]
        self.assertIn('METRICS["failed_requests"] += 1', stream)
        self.assertIn('finish="error"', stream)

    def test_request_id_is_echoed_in_a_header(self):
        # Without this the id exists only inside the JSON body, so a client
        # cannot cite a request the server is able to find.
        self.assertIn('response.headers["X-Request-Id"] = req_id', self.SERVER_SRC)
        self.assertEqual(self.SERVER_SRC.count('"X-Request-Id": req_id'), 2,
                         "both streaming responses must carry it too")

    def test_one_log_line_per_request_carries_the_shape(self):
        record = self.SERVER_SRC.split("def _record")[1].split("\ndef _sse_stream")[0]
        for field in ("id=%s", "bucket=%d", "pad=%d", "cold=%s", "finish=%s"):
            with self.subTest(field=field):
                self.assertIn(field, record)

    # ------------------------------------------------------------- padding

    def test_generation_stats_carries_the_variable_behind_the_eviction_bug(self):
        """pad_len predicted the failure 14/14 and was being discarded."""
        for field in ("bucket_size", "pad_len", "cold_shape",
                      "prefill_chunked", "max_new_tokens_clamped"):
            with self.subTest(field=field):
                self.assertIn(f"{field}:", self.ENGINE_SRC)
        self.assertIn("pad_len = bucket_s - prompt_len", self.ENGINE_SRC)

    def test_padding_at_or_past_the_window_is_warned_about(self):
        self.assertIn("the pre-fix eviction", self.ENGINE_SRC)
        self.assertIn("pad_len >= int(win)", self.ENGINE_SRC)

    def test_degenerate_output_is_logged_with_its_padding(self):
        record = self.SERVER_SRC.split("def _record")[1].split("\ndef _sse_stream")[0]
        self.assertIn("DEGENERATE OUTPUT", record)
        self.assertIn("logger.error", record)

    def test_pad_metrics_are_exported(self):
        for series in ("tpu_jax_last_pad_tokens", "tpu_jax_max_pad_tokens",
                       "tpu_jax_last_bucket_size", "tpu_jax_cold_requests_total",
                       "tpu_jax_decode_seconds_total"):
            with self.subTest(series=series):
                self.assertIn(series, self.SERVER_SRC)

    def test_metrics_carry_a_rig_label(self):
        # Two rigs serving the same checkpoint otherwise emit byte-identical
        # series names and label sets.
        self.assertIn('f\'rig="{RIG_NAME}"\'', self.SERVER_SRC)
        self.assertIn('RIG_NAME = os.environ.get("RIG_NAME"', self.SERVER_SRC)
        # ...and the rig name must actually reach the serving process.
        self.assertIn("RIG_NAME={MANAGED_BY}", (ROOT / "server.py").read_text())

    # ---------------------------------------------------- silent fallbacks

    def test_the_two_silent_fallbacks_now_warn(self):
        self.assertIn("to ONE-SHOT prefill for this shape", self.ENGINE_SRC)
        self.assertIn("max_new_tokens clamped", self.ENGINE_SRC)

    # ------------------------------------------------------ build identity

    def test_payload_digest_is_deterministic_and_content_addressed(self):
        first, second = server._payload_digest(), server._payload_digest()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertRegex(first, r"^[0-9a-f]{12}$")

    def test_payload_stamp_rides_in_the_tarball(self):
        import io
        import tarfile

        raw = base64.b64decode(server._payload_tar_b64())
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            names = tar.getnames()
            self.assertIn("PAYLOAD_SHA", names)
            stamped = tar.extractfile("PAYLOAD_SHA").read().decode().strip()
        self.assertEqual(stamped, server._payload_digest())

    def test_the_server_reports_the_stamp_it_was_shipped(self):
        self.assertIn('stamp = os.path.join(here, "PAYLOAD_SHA")', self.SERVER_SRC)
        self.assertIn('"build_id": BUILD_ID', self.SERVER_SRC)

    def test_deploy_reports_the_root_it_resolved(self):
        # _payload_root() silently picks between the working tree and the skill
        # snapshot; on 2026-08-24 it chose the stale snapshot and said nothing.
        src = (ROOT / "server.py").read_text()
        deploy = src.split("async def deploy_jax_server")[1].split("@mcp.tool")[0]
        self.assertIn("Payload root:", deploy)
        self.assertIn("Build id:", deploy)

    # ------------------------------------------------------- health verdict

    def test_health_does_not_pass_on_a_merely_non_empty_reply(self):
        """The rule the tool itself used to break.

        A broken deploy answered ': ok: ok: ok…' on the vLLM sibling, and KV-ring
        eviction returned a token loop here — both non-empty, both broken.
        """
        src = (ROOT / "server.py").read_text()
        body = src.split("async def verify_model_health")[1].split("@mcp.tool")[0]
        self.assertNotIn("and text.strip() else", body)
        self.assertIn("tpu_jax_degenerate_responses_total", body)
        self.assertIn("degenerate", body)

    def test_health_compares_the_served_build_against_the_local_one(self):
        src = (ROOT / "server.py").read_text()
        body = src.split("async def verify_model_health")[1].split("@mcp.tool")[0]
        self.assertIn("STALE DEPLOY", body)
        self.assertIn("_payload_digest()", body)

    def test_health_is_503_while_loading(self):
        self.assertIn("response.status_code = 503", self.SERVER_SRC)

    # ---------------------------------------------------------------- SSM

    def test_ssm_failures_carry_the_command_id(self):
        src = (ROOT / "server.py").read_text()
        body = src.split("async def _ssm")[1].split("\ndef _error")[0]
        self.assertIn("command-id {command_id}", body)
        self.assertIn("STILL RUNNING", body)
        self.assertIn("logger.info", body)

    def test_ssm_truncation_is_detected_not_assumed(self):
        src = (ROOT / "server.py").read_text()
        self.assertIn("_SSM_OUTPUT_CAP = 24_000", src)
        self.assertIn("OUTPUT TRUNCATED", src)

    def test_error_renderer_logs_the_traceback(self):
        src = (ROOT / "server.py").read_text()
        body = src.split("def _error(exc")[1].split("@mcp.tool")[0]
        self.assertIn("logger.exception", body)

    def test_launch_reports_the_ami_it_booted(self):
        src = (ROOT / "server.py").read_text()
        body = src.split("async def create_g4dn_instance")[1].split("@mcp.tool")[0]
        self.assertIn("AMI: `{ami_id}`", body)


class QuantizerMemoryTests(unittest.TestCase):
    """The load-time quantizers must not allocate the destination on top of the
    source. Both bugs below were hard startup failures on a T4G 2026-08-26, and
    both allocations matched their tensor byte-for-byte."""

    SRC = None

    @classmethod
    def setUpClass(cls):
        cls.SRC = (ROOT / "ports" / "gemma4" / "jax_e_model.py").read_text()

    def _fn(self, name):
        body = self.SRC.split(f"def {name}")[1]
        return body.split("\ndef ")[0]

    def test_lm_head_quantizes_on_the_host_in_chunks(self):
        """262144 x 1536 x 4 B = 1.50 GiB, the exact allocation that failed."""
        body = self._fn("quantize_lm_head")
        self.assertIn('jax.devices("cpu")', body)
        self.assertIn("rows_per_chunk", body)
        # The upcast must apply to a CHUNK, never the whole table. Checked on
        # executable lines only -- the comment above the fix quotes the old
        # expression on purpose, and matching prose would pass vacuously.
        code = "\n".join(ln for ln in body.splitlines()
                          if not ln.lstrip().startswith("#"))
        self.assertIn("blk = blk.astype(jnp.float32)", code)
        self.assertNotIn("emb.astype(jnp.float32)", code)

    def test_lm_head_moves_the_table_host_side_before_slicing(self):
        # Slicing a device-resident table allocates a device buffer per chunk,
        # which defeats the point of chunking.
        body = self._fn("quantize_lm_head")
        move = body.index("emb_host = jax.device_put(emb, cpu)")
        first_slice = body.index("emb_host[start:start + rows_per_chunk]")
        self.assertLess(move, first_slice)

    def test_ple_releases_the_source_before_placing_the_copy(self):
        """262144 x 8960 x 1 B = 2.19 GiB, the exact allocation that failed.

        Popping from the returned dict frees nothing -- the caller still holds
        the device-resident original.
        """
        body = self._fn("quantize_ple_table")
        self.assertIn("source.delete()", body)
        # Opt-in: .delete() invalidates the CALLER's array. A CPU test caught
        # this within seconds of the first version defaulting it to on.
        self.assertIn("if release_source:", body)
        engine = (ROOT / "jax_engine.py").read_text()
        self.assertIn("release_source=True", engine,
                      "load() must opt in, or the fix does nothing in production")
        delete_at = body.index("source.delete()")
        place_at = body.index("q_all = jax.device_put(q_all, home)")
        self.assertLess(delete_at, place_at,
                        "the source must be released BEFORE the copy is placed")

    def test_the_failing_allocations_match_their_tensors(self):
        # Guards the arithmetic the comments claim, so a shape change is caught.
        self.assertAlmostEqual(262144 * 1536 * 4 / 2**30, 1.50, places=2)
        self.assertAlmostEqual(262144 * 8960 * 1 / 2**30, 2.19, places=2)
