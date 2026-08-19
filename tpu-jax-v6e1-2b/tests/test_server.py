"""Smoke and regression tests for the tpu-jax-v6e1-2b MCP server and repo invariants.

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
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import server  # noqa: E402

EXPECTED_DESTRUCTIVE = {
    "destroy_tpu_vm_instance",
    "manage_vllm_docker",
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
            ["v6e", "v5p", "v5e"],
        )

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

    def test_min_chips_for_model_on_v6e(self):
        def chips(m):
            return server._min_chips_for_model(m, "v6e-8")

        self.assertEqual(chips("google/gemma-4-31B-it"), 4)
        self.assertEqual(chips("google/gemma-4-12B-it"), 1)
        self.assertEqual(chips("google/gemma-4-E2B-it"), 1)
        # int4 is ~4x smaller, so these drop back to a single 32GB chip
        self.assertEqual(chips("google/gemma-4-26B-it-qat-q4_0"), 1)
        self.assertEqual(chips("google/gemma-4-31B-it-int4"), 1)

    def test_min_chips_for_model_is_sized_on_the_target_chip(self):
        # A v5e chip has half a v6e's HBM, so the same checkpoint needs more of them.
        # bf16 12B is the case the old fixed-32GB floor waved through onto a v5e-1.
        self.assertEqual(server._min_chips_for_model("google/gemma-4-12B-it", "v6e-1"), 1)
        self.assertEqual(server._min_chips_for_model("google/gemma-4-12B-it", "v5e-1"), 4)
        # Quantized is not an automatic pass either: 26B int4 is still ~13GB.
        self.assertEqual(server._min_chips_for_model("google/gemma-4-26B-it-qat-q4_0", "v5e-1"), 4)
        # The intended v5e-1 payload, and bf16 E2B, both still fit one chip.
        self.assertEqual(server._min_chips_for_model("google/gemma-4-E2B-it-qat-w4a16-ct", "v5e-1"), 1)
        self.assertEqual(server._min_chips_for_model("google/gemma-4-E2B-it", "v5e-1"), 1)
        # Unknown checkpoints are never blocked.
        self.assertEqual(server._min_chips_for_model("some/unknown-model", "v5e-1"), 1)

    def test_quota_ids_are_the_compute_engine_ones(self):
        """The TPU API's metrics do not govern anything on this path.

        They are a different service with disjoint pools, and reading a TPU-API metric
        here answers confidently and wrongly — this project holds 512 v6e chips under
        the TPU API in a region where Compute Engine gave it nothing.
        """
        self.assertEqual(server.GCE_QUOTA_ID, "TPUS-PER-TPU-FAMILY-per-project-region")
        self.assertEqual(server.GCE_SPOT_QUOTA_ID, "PREEMPTIBLE-TPU-V6E-per-project-region")
        for quota_id in (server.GCE_QUOTA_ID, server.GCE_SPOT_QUOTA_ID):
            self.assertNotIn("ForTPUAPI", quota_id)

    def test_create_blocks_model_too_big_for_accelerator(self):
        result = run(
            server.create_tpu_vm_instance(
                accelerator="v6e-1", model_name="google/gemma-4-31B-it", workload="vllm"
            )
        )
        self.assertTrue(result.startswith("❌"))
        self.assertIn("chips", result)

    def test_gce_shapes_are_the_creatable_ones(self):
        self.assertEqual(server._gce_machine_type("v6e-1"), ("ct6e-standard-1t", 1))
        self.assertEqual(server._gce_machine_type("v6e-8"), ("ct6e-standard-8t", 8))
        self.assertEqual(server._gce_machine_type("v5p-8"), ("ct5p-hightpu-4t", 4))
        self.assertIsNone(server._gce_machine_type("v6e-2"))

    def test_v5e_has_no_compute_engine_path(self):
        """ct5lp-* machine types exist in the catalog in 26 zones and cannot be created.

        `gcloud compute instances create --machine-type=ct5lp-hightpu-1t` is refused at
        validation: "This user agent is not allowed to use the machine type". That is
        neither a quota error nor a does-not-exist error, so the refusal has to say so —
        otherwise it reads as something another zone or a quota request could fix.
        """
        for spelling in ("v5e-1", "v5e-8", "v5litepod-1"):
            self.assertIsNone(server._gce_machine_type(spelling))
        msg = server._unsupported_accelerator_message("v5e-1")
        self.assertIn("no Compute Engine provisioning path", msg)
        self.assertIn("not of this zone or your quota", msg)
        # An unrecognised shape must NOT claim a generation-level explanation it lacks.
        self.assertNotIn("no Compute Engine provisioning path", server._unsupported_accelerator_message("v6e-2"))

    def test_provisioning_models_map_to_screaming_case(self):
        """`flex-start` and `FLEX_START` are the same request to different APIs."""
        flags = server._provisioning_flags("flex-start", "2h", "4h")
        self.assertIn("--provisioning-model=FLEX_START", flags)
        self.assertIn("--request-valid-for-duration=2h", flags)
        self.assertIn("--max-run-duration=4h", flags)
        self.assertIn("--instance-termination-action=DELETE", flags)
        self.assertIn("--provisioning-model=SPOT", server._provisioning_flags("spot", "2h", "4h"))
        self.assertIn("--provisioning-model=STANDARD", server._provisioning_flags("on-demand", "2h", "4h"))
        with self.assertRaises(ValueError):
            server._provisioning_flags("preemptible", "2h", "4h")

    def test_only_flex_start_gets_the_queue_knob(self):
        """--request-valid-for-duration is flex-start's wait knob; the others reject it."""
        for model in ("spot", "on-demand"):
            flags = server._provisioning_flags(model, "2h", "4h")
            self.assertNotIn("--request-valid-for-duration=2h", flags)

    def test_reservation_bound_needs_a_reservation(self):
        """gcloud takes RESERVATION_BOUND only with --reservation-affinity=specific, so
        the model on its own is not a complete request."""
        with self.assertRaises(ValueError):
            server._provisioning_flags("reservation-bound", "2h", "4h")
        with mock.patch.object(server, "RESERVATION_NAME", "my-calendar-reservation"):
            flags = server._provisioning_flags("reservation-bound", "2h", "4h")
        self.assertIn("--reservation-affinity=specific", flags)
        self.assertIn("--reservation=my-calendar-reservation", flags)

    def test_ssh_targets_compute_engine_not_the_tpu_api(self):
        """`gcloud compute tpus tpu-vm ssh` cannot reach a ct6e-* instance.

        This is the call site a migration most often leaves behind, because everything
        that uses it — container management, log tailing, journalctl, benchmarks — is
        what you reach for after something has already gone wrong.
        """
        argv, target = server._build_ssh_cmd("uptime", None, "europe-west4-a")
        self.assertEqual(argv[:3], ["gcloud", "compute", "ssh"])
        self.assertNotIn("tpus", argv)
        self.assertNotIn("tpu-vm", argv)
        self.assertEqual(target, server.INSTANCE_NAME)
        argv, target = server._build_ssh_cmd("uptime", "other-vm", "us-central1-b")
        self.assertEqual(target, "other-vm")
        self.assertIn("--zone=us-central1-b", argv)

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
            ROOT / ".claude/skills/tpu-jax-v6e1-2b-management/mcp/startup_script_template.sh"
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


class JaxStartupTemplateTests(unittest.TestCase):
    """The JAX startup script must fail loudly, not print a success marker anyway.

    Regression guard: a hand-rolled precursor of this script omitted `set -e`, its
    pip step failed on an Ubuntu pip too old for --break-system-packages, and it
    still emitted the ready marker. The VM looked provisioned and had no JAX.
    """

    def render(self):
        template = (
            ROOT / ".claude/skills/tpu-jax-v6e1-2b-management/mcp/startup_script_jax_template.sh"
        ).read_text()
        return template.format(
            project_id="test-project",
            zone="europe-west4-a",
            python_version="3.13",
            jax_pip_spec="jax[tpu]",
            jax_pip_extras="numpy scipy",
        )

    def test_renders_without_leftover_placeholders(self):
        rendered = self.render()
        self.assertNotIn("{project_id}", rendered)
        self.assertNotIn("{jax_pip_spec}", rendered)
        self.assertNotIn("{python_version}", rendered)
        self.assertIn("test-project", rendered)
        self.assertIn("python3.13", rendered)

    def test_rendered_script_passes_bash_syntax_check(self):
        rendered = self.render()
        proc = subprocess.run(
            ["bash", "-n", "/dev/stdin"], input=rendered, text=True, capture_output=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_fails_loudly(self):
        """set -e plus an ERR trap, and a FAILED marker distinct from the ready one."""
        rendered = self.render()
        self.assertRegex(rendered, r"set -eu?x?", "script must exit on error")
        self.assertIn("trap", rendered)
        self.assertIn("JAX-BOOTLOADER: FAILED", rendered)

    def test_asserts_a_tpu_is_actually_visible(self):
        """Importing jax succeeds without a TPU backend, so assert on devices()."""
        rendered = self.render()
        self.assertIn("jax.devices()", rendered)
        self.assertRegex(rendered, r"platform\s*==\s*.tpu.")
        self.assertIn("sys.exit(1)", rendered)

    def test_ready_marker_comes_after_the_device_assertion(self):
        rendered = self.render()
        self.assertLess(
            rendered.index("jax.devices()"),
            rendered.index("JAX-BOOTLOADER: TPU environment ready."),
            "ready marker must not precede the TPU check",
        )

    def code(self):
        """Executable lines only — the prose explains what the script avoids."""
        return "\n".join(
            ln for ln in self.render().splitlines() if ln.strip() and not ln.lstrip().startswith("#")
        )

    def test_no_docker_or_hf_token(self):
        code = self.code()
        for forbidden in ("docker", "hf-token", "HF_TOKEN", "secretmanager"):
            self.assertNotIn(forbidden, code, f"bare JAX VM should not run {forbidden}")

    def test_trap_is_installed_before_tracing(self):
        """`set -x` echoes the trap definition, which contains the FAILED marker.

        If tracing is enabled first, that trace line makes a healthy boot look
        like a failure to any log scanner (it did, on the first real run).
        """
        rendered = self.render()
        trap_at = rendered.index("trap ")
        set_x_at = rendered.index("\nset -x")
        self.assertLess(trap_at, set_x_at, "trap must be installed before `set -x`")
        self.assertNotIn("set -eux", rendered, "combined -eux re-introduces the trace leak")

    def test_makes_tpu_logs_group_writable(self):
        """The script runs as root; libtpu's /tmp/tpu_logs would then block users."""
        rendered = self.render()
        self.assertIn("/tmp/tpu_logs", rendered)
        self.assertIn("1777", rendered)

    def test_no_virtualenv(self):
        """Repo standard: pip into a dedicated interpreter, never a venv."""
        code = self.code()
        self.assertNotIn("venv", code)
        self.assertNotIn("virtualenv", code)


class TemplateResolutionTests(unittest.TestCase):
    """Templates must resolve whether server.py runs deployed or from the repo root.

    The templates are hand-maintained inside the skill, beside the *deployed* copy
    of server.py. This module imports the repo-root server.py, where they are not
    siblings — which used to raise FileNotFoundError at VM-creation time.
    """

    def test_vllm_template_renders_from_repo_root(self):
        script = server._get_formatted_startup_script(
            "google/gemma-4-E2B-it", "europe-west4-a", tp_size=1
        )
        self.assertIn("vllm", script)
        proc = subprocess.run(
            ["bash", "-n", "/dev/stdin"], input=script, text=True, capture_output=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_jax_template_renders_from_repo_root(self):
        script = server._get_jax_startup_script("europe-west4-a")
        self.assertIn("JAX-BOOTLOADER", script)
        proc = subprocess.run(
            ["bash", "-n", "/dev/stdin"], input=script, text=True, capture_output=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_missing_template_names_every_location_searched(self):
        with self.assertRaises(RuntimeError) as ctx:
            server._read_template("definitely_not_a_template.sh")
        msg = str(ctx.exception)
        self.assertIn("definitely_not_a_template.sh", msg)
        for directory in server._TEMPLATE_SEARCH_DIRS:
            self.assertIn(directory, msg, "error should name each directory searched")

    def test_deployed_copy_is_a_sibling_of_its_templates(self):
        """The snapshot server.py must ship next to both templates."""
        mcp_dir = ROOT / ".claude/skills/tpu-jax-v6e1-2b-management/mcp"
        for name in ("server.py", "startup_script_template.sh", "startup_script_jax_template.sh"):
            self.assertTrue((mcp_dir / name).is_file(), f"{name} missing from the deployed skill")


class JaxReadyMarkerScanTests(unittest.TestCase):
    """Serial-log scanning must distinguish printed markers from `set -x` traces."""

    def scan(self, serial, marker):
        # Mirror of the closure inside wait_for_jax_ready.
        for line in serial.splitlines():
            body = line.split("startup-script:", 1)[-1].strip()
            if body.startswith("+"):
                continue
            if marker in body:
                return True
        return False

    def test_traced_trap_definition_is_not_a_failure(self):
        serial = (
            "Jul 28 21:22 vm google_metadata_script_runner[1]: startup-script: "
            "+ trap 'rc=$?; echo \"JAX-BOOTLOADER: FAILED\"; exit $rc' ERR"
        )
        self.assertFalse(
            self.scan(serial, "JAX-BOOTLOADER: FAILED"),
            "a traced trap definition must not read as a failure",
        )

    def test_real_failure_is_detected(self):
        serial = "Jul 28 21:22 vm x[1]: startup-script: JAX-BOOTLOADER: FAILED"
        self.assertTrue(self.scan(serial, "JAX-BOOTLOADER: FAILED"))

    def test_real_ready_is_detected(self):
        serial = "Jul 28 21:22 vm x[1]: startup-script: JAX-BOOTLOADER: TPU environment ready."
        self.assertTrue(self.scan(serial, "JAX-BOOTLOADER: TPU environment ready."))

    def test_traced_echo_of_ready_is_ignored(self):
        serial = "Jul 28 21:22 vm x[1]: startup-script: + echo 'JAX-BOOTLOADER: TPU environment ready.'"
        self.assertFalse(self.scan(serial, "JAX-BOOTLOADER: TPU environment ready."))


class TemplateBraceHygieneTests(unittest.TestCase):
    """Startup templates go through str.format(), so stray braces are a landmine.

    Bash brace groups (`{ cmd; }`) and awk programs (`awk '{print $2}'`) both raise
    KeyError at render time — i.e. at VM-creation time, not in CI. This has bitten
    twice; the templates avoid braces entirely rather than escaping them.
    """

    TEMPLATES = {
        "startup_script_template.sh": {
            "project_id", "zone", "model_name", "hf_secret_id", "tp_size",
            "limit_mm_per_prompt_env",
        },
        "startup_script_jax_template.sh": {
            "project_id", "zone", "python_version", "jax_pip_spec", "jax_pip_extras",
        },
        "startup_script_cpu_template.sh": {
            "project_id", "zone", "python_version", "pip_spec",
        },
    }

    def test_only_known_placeholders_appear(self):
        import re
        for name, allowed in self.TEMPLATES.items():
            text = (ROOT / ".claude/skills/tpu-jax-v6e1-2b-management/mcp" / name).read_text()
            found = set(re.findall(r"\{([^}]*)\}", text))
            unexpected = found - allowed
            self.assertEqual(
                unexpected, set(),
                f"{name} has brace groups str.format() will choke on: {unexpected}",
            )

    def test_every_template_renders(self):
        """Render each through the real server helper — the render IS the test."""
        import subprocess
        for render in (
            lambda: server._get_formatted_startup_script("google/gemma-4-E2B-it", "europe-west4-a", tp_size=1),
            lambda: server._get_jax_startup_script("europe-west4-a"),
            lambda: server._get_cpu_debug_startup_script("europe-west4-a"),
        ):
            script = render()
            proc = subprocess.run(["bash", "-n", "/dev/stdin"], input=script,
                                  text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)


class OffTheCloudTpuApiTests(unittest.TestCase):
    """This rig provisions only through Compute Engine. Grep for the old commands.

    One test over one function is not enough here, and that is the whole lesson: a rig
    can carry a test asserting it is off the deprecated API, have that test cover the
    discovery helper, and still have four other tools calling `tpu-vm ssh` behind its
    back plus several Makefile targets. None of them throws — `gcloud compute tpus
    tpu-vm list` returns empty for a ct6e-* instance, with no error and no warning, so
    the tooling simply believes nothing is running. So these scan the whole source text
    rather than any one call site.
    """

    SOURCES = (
        "server.py",
        "Makefile",
        ".claude/skills/tpu-jax-v6e1-2b-management/mcp/startup_script_template.sh",
        ".claude/skills/tpu-jax-v6e1-2b-management/mcp/startup_script_jax_template.sh",
        ".claude/skills/tpu-jax-v6e1-2b-management/mcp/startup_script_cpu_template.sh",
    )

    def code_lines(self, rel):
        """Executable lines only.

        Prose explaining what this rig migrated OFF names the old commands on purpose,
        so docstrings and comments are stripped rather than matched. Strings that are
        NOT docstrings stay in scope — a gcloud argv element is a string literal, and
        dropping those would defeat the whole check.
        """
        text = (ROOT / rel).read_text()
        drop = set()
        if rel.endswith(".py"):
            import ast

            for node in ast.walk(ast.parse(text)):
                body = getattr(node, "body", None)
                if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    drop.update(range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1))
        return [
            ln
            for i, ln in enumerate(text.splitlines(), 1)
            if i not in drop and ln.strip() and not ln.lstrip().startswith("#")
        ]

    def test_no_tpu_vm_or_queued_resource_commands(self):
        for rel in self.SOURCES:
            for i, line in enumerate(self.code_lines(rel), 1):
                for forbidden in ("tpu-vm", "queued-resources", "tpus tpu-vm"):
                    self.assertNotIn(
                        forbidden, line,
                        f"{rel} still calls the Cloud TPU API ({forbidden!r}) at code line {i}: {line.strip()}",
                    )

    def test_no_queued_resource_tools_remain(self):
        names = {t.name for t in run(server.mcp.list_tools())}
        for gone in (
            "create_tpu_queued_resource",
            "manage_queued_resource",
            "destroy_queued_resource",
            "list_queued_resources",
            "describe_queued_resource",
            "find_tpu",
            "query_queued_gemma4",
        ):
            self.assertNotIn(gone, names)

    def test_discovery_reads_the_compute_engine_field_shapes(self):
        """Two field shapes moved and NEITHER throws when stale.

        `state: READY` became `status: RUNNING`, and the external IP moved from
        networkEndpoints[].accessConfig.externalIp to
        networkInterfaces[].accessConfigs[].natIP. A copied-across status check does not
        error; it just sorts every healthy instance to the bottom of a ranking, which
        you notice the day you have two.
        """
        source = (ROOT / "server.py").read_text()
        self.assertIn("networkInterfaces[0].accessConfigs[0].natIP", source)
        self.assertNotIn("networkEndpoints[0].accessConfig.externalIp", source)
        self.assertNotIn("networkEndpoints[0].ipAddress", source)

    def test_cloud_logging_filters_on_gce_instance(self):
        """A ct6e-* instance logs as gce_instance. The old filter returns zero entries
        cleanly, so a stale one reads as "no errors" rather than as a wrong query."""
        source = (ROOT / "server.py").read_text()
        self.assertNotIn('resource.type="tpu_worker"', source.replace("# ", ""))
        self.assertIn('resource.type="gce_instance"', source)

    def test_startup_script_installs_docker_before_pulling(self):
        """The accelerator image ships NO Docker on PATH at first boot.

        The same script worked for months on the TPU API's runtime version, which
        evidently provided it. Ported verbatim it went straight for the pull and died
        100 seconds in with `sudo: docker: command not found`, while the instance
        reported RUNNING the whole time.
        """
        script = server._get_formatted_startup_script("google/gemma-4-E2B-it", "europe-west4-a", tp_size=1)
        install_at = script.index("apt-get install -y docker.io")
        pull_at = script.index("docker pull")
        self.assertLess(install_at, pull_at, "Docker must be installed before the first pull")

    def test_create_passes_the_three_flags_that_fail_late(self):
        """--scopes, --boot-disk-size and --maintenance-policy all fail long after the
        create, so nothing about the failure points back at the flag."""
        source = (ROOT / "server.py").read_text()
        for flag in ("--scopes=cloud-platform", "--maintenance-policy=TERMINATE", "--boot-disk-size="):
            self.assertIn(flag, source)


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
            ("server.py", ".claude/skills/tpu-jax-v6e1-2b-management/mcp/server.py"),
            ("project-setup.sh", ".claude/skills/tpu-jax-v6e1-2b-management/mcp/project-setup.sh"),
            ("requirements.txt", ".claude/skills/tpu-jax-v6e1-2b-management/mcp/requirements.txt"),
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
            "mcp/startup_script_jax_template.sh",
            "mcp/startup_script_cpu_template.sh",
        ):
            self.assertTrue(
                filecmp.cmp(
                    ROOT / ".claude/skills/tpu-jax-v6e1-2b-management" / rel,
                    ROOT / "skills/tpu-jax-v6e1-2b-management" / rel,
                    shallow=False,
                ),
                f"skills/tpu-jax-v6e1-2b-management/{rel} is stale — run `make skill`",
            )


if __name__ == "__main__":
    unittest.main()
