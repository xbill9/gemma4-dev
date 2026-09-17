"""Offline unit tests for the attached-T4 rig's MCP server.

unittest, never pytest. The whole `mcp` module is mocked before `server` is
imported, so nothing here touches the network, a subprocess, the GPU, or a real
filesystem probe.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

RIG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RIG_DIR))


class _FakeMCPServer:
    def __init__(self, name):
        self.name = name

    def tool(self, *args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    async def list_tools(self):
        return []

    def run(self):
        raise AssertionError("mcp.run() must never be called from a test")


_mcpserver_module = MagicMock()
_mcpserver_module.MCPServer = _FakeMCPServer
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.mcpserver"] = _mcpserver_module

import server  # noqa: E402

GB = 10**9


class TestRigIdentity(unittest.TestCase):
    def test_rig_name_matches_directory(self):
        self.assertEqual(server.RIG_NAME, RIG_DIR.name)
        self.assertEqual(server.RIG_NAME, "gpu-vllm-t4-2b")

    def test_server_name_defaults_to_rig_name(self):
        self.assertEqual(server.MCP_SERVER_NAME, server.RIG_NAME)

    def test_slot_3_is_the_gpu_sku_not_an_ec2_family(self):
        """@NAMING.md spells the EC2 instance family only on EC2; everywhere else
        slot 3 is the GPU SKU. This is Compute Engine, so `t4`.

        `gpu-jax-t4-2b` and `gpu-jax-l4-2b` were renamed to `g4dn`/`g6` on
        2026-08-28 under that rule, and this rig must not be "corrected" the same
        way — it is not on EC2, and the machine type has no slot at all.
        """
        self.assertNotIn("g4dn", server.RIG_NAME)
        self.assertNotIn("n1", server.RIG_NAME)
        self.assertIn("-t4-", server.RIG_NAME)


class TestNoCloudControlPlane(unittest.TestCase):
    """The most load-bearing class in this suite.

    This rig was forked from `gpu-vllm-g4dn-2b`, whose server.py is 1274 lines of
    which the majority launches, starts, stops, terminates and SSMs into EC2
    instances. A fork of a cloud rig keeps passing its own tests while describing
    hardware that does not exist, so the test that matters asserts the vocabulary
    is GONE.

    IT DOES NOT FORBID `gcloud`, AND THAT IS DELIBERATE. This rig legitimately
    PRINTS a `gcloud compute disks resize` suggestion in a remedy string — that is
    advice to an operator, not a control plane. What would make it a cloud rig is
    EXECUTING one, so `test_no_cloud_binary_is_ever_executed` checks the argv of
    every subprocess call site instead of banning the word.
    """

    FORBIDDEN = frozenset(
        {
            "boto3",
            "botocore",
            "ec2",
            "ssm",
            "secretsmanager",
            "instance_id",
            "ami",
            "spot",
            "systemd",
            "queued_resource",
            "aws_region",
            "aws_profile",
            "hf_secret_id",
            "resolve_ami",
            "create_g4dn_instance",
            "terminate_g4dn_instance",
            "start_g4dn_instance",
            "check_g4dn_quotas",
            "find_tpu",
            "vllm_patched_image",
            "dlami_ssm_parameter",
            "root_volume_gb",
        }
    )

    # Applied to STRING LITERALS, not identifiers. Naming EC2 in prose the rig
    # prints is provenance — the same thing a docstring does — and banning the word
    # would force the output to be vaguer than the truth. What must never appear in
    # a string is something that indicates a live cloud call.
    FORBIDDEN_IN_STRINGS = frozenset({
        "boto3", "botocore", "instance_id", "queued_resource", "secretsmanager",
    })

    def _identifiers(self):
        """Every name the module's CODE mentions, with docstrings EXCLUDED.

        Checking raw text does not work: this module's docstrings legitimately
        name the EC2 siblings as provenance, and a substring search for "ami"
        hits half the English language. The invariant is about code, so the test
        reads code — and it collects every identifier, because a helper is as much
        of a violation as a tool.
        """
        import ast
        import re

        tree = ast.parse((RIG_DIR / "server.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    node.body = node.body[1:] or [ast.Pass()]
        names, words = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.alias):
                names.add(node.name.split(".")[0])
                names.update(node.name.split("."))
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.update(node.module.split("."))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                words.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", node.value))
        return {n.lower() for n in names}, {w.lower() for w in words}

    def test_source_names_no_cloud_control_plane(self):
        names, _ = self._identifiers()
        found = names & self.FORBIDDEN
        self.assertEqual(found, set(), f"cloud control-plane identifiers in code: {sorted(found)}")

    def test_no_string_the_rig_prints_implies_a_live_cloud_call(self):
        _, words = self._identifiers()
        found = words & self.FORBIDDEN_IN_STRINGS
        self.assertEqual(found, set(), f"cloud-call vocabulary in emitted strings: {sorted(found)}")

    def test_no_cloud_binary_is_ever_executed(self):
        """The rig may print `gcloud ...` as advice; it must never run it.

        Checks the first element of every list passed to a subprocess helper,
        which is the argv[0] that would actually be executed.
        """
        import ast

        tree = ast.parse((RIG_DIR / "server.py").read_text())
        executed = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name not in {"run_command", "create_subprocess_exec"}:
                    continue
                args = node.args
                if args and isinstance(args[0], ast.List) and args[0].elts:
                    head = args[0].elts[0]
                    if isinstance(head, ast.Constant):
                        executed.add(str(head.value))
                elif args and isinstance(args[0], ast.Constant):
                    executed.add(str(args[0].value))
        for binary in executed:
            self.assertNotIn(
                binary,
                {"gcloud", "aws", "kubectl", "docker"},
                f"this rig executes {binary}, which would make it a control plane",
            )

    def test_no_provisioning_tool_exists(self):
        for absent in ("create_instance", "terminate_instance", "find_capacity", "deploy"):
            self.assertFalse(hasattr(server, absent), f"{absent} must not exist here")


class TestPlatformSlot(unittest.TestCase):
    """`gpu`, not `local`, and the distinction is the reason this class exists.

    The GPU is already attached and nothing provisions it, which is `local`'s test
    — but @NAMING.md decides the slot by WHO OWNS THE MACHINE: "SSH into a cloud VM
    you provisioned and it keeps that VM's platform value." This is a GCE VM, so
    the slot is `gpu` and the rig is the first `gpu` rig here with no provisioning
    half.
    """

    def test_platform_slot_is_gpu(self):
        self.assertTrue(server.RIG_NAME.startswith("gpu-"))

    def test_host_platform_is_recorded_as_a_cloud_vm(self):
        import os

        self.assertEqual(os.environ.get("HOST_PLATFORM"), "gce")
        self.assertTrue(os.environ.get("HOST_MACHINE_TYPE", "").startswith("n1-"))

    def test_endpoint_is_loopback_because_there_is_no_discovery_chain(self):
        self.assertIn("127.0.0.1", server.ENDPOINT)


class TestTuringPolicy(unittest.TestCase):
    def test_dtype_is_float16_not_bfloat16(self):
        """Turing has no bf16 datapath. bf16 does NOT fail — PyTorch upconverts
        and vLLM logs the cast and proceeds — so the guard protects against a
        silent cost rather than an error, which is why it must not be relaxed."""
        self.assertEqual(server.DTYPE, "float16")

    def test_kv_cache_dtype_is_not_fp8(self):
        """fp8 has no datapath at all on SM 7.5, unlike on the Ada sibling where
        it exists and is merely unused."""
        self.assertEqual(server.KV_CACHE_DTYPE, "auto")

    def test_no_attention_backend_is_pinned(self):
        """MEASURED on `gpu-vllm-g5g-2b`: vLLM does not recognize
        VLLM_ATTENTION_BACKEND and forces TRITON_ATTN for Gemma 4 anyway, so
        setting it would be dead config.

        AST rather than a substring search, for the same reason as the shell=True
        test: server.py's docstring EXPLAINS why the variable is absent, and a text
        search cannot tell an explanation from a setting.
        """
        import ast

        tree = ast.parse((RIG_DIR / "server.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                        and isinstance(first.value.value, str):
                    node.body = node.body[1:] or [ast.Pass()]
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertNotIn(
                    "VLLM_ATTENTION_BACKEND", node.value,
                    "the backend must not be pinned; vLLM ignores it and forces TRITON_ATTN",
                )

    def test_smem_budget_is_under_the_hard_limit(self):
        """65,536 is the opt-in maximum and the tile arithmetic does not count the
        kernel's accumulators, so budgeting the full limit still overflows."""
        self.assertLess(server.TURING_SMEM_BUDGET, 65536)

    def test_the_sentinel_matches_the_patch_script(self):
        """If these drift, verification looks for a string the patch never writes
        and every failure gets blamed on the wrong thing."""
        source = (RIG_DIR / "patch_triton_turing.py").read_text()
        self.assertIn(f'SENTINEL = "{server._PATCH_SENTINEL}"', source)

    def test_the_patch_marker_names_this_rig(self):
        source = (RIG_DIR / "patch_triton_turing.py").read_text()
        self.assertIn(f'MARKER = "# {server.RIG_NAME}:', source)

    def test_idempotency_keys_on_the_shared_sentinel_not_the_rig_marker(self):
        """site-packages is shared host-wide, unlike a per-rig docker image. A
        check keyed on this rig's marker would not recognise a sibling script's
        clamp and would patch the file twice, halving the tiles again."""
        source = (RIG_DIR / "patch_triton_turing.py").read_text()
        self.assertIn("if SENTINEL in text:", source)


class TestTargetInterpreter(unittest.TestCase):
    """Every probe must ask the interpreter that will SERVE, not this one.

    On this host `python3` is pyenv 3.12.13 with site-packages on a nearly-full
    root disk, while `/usr/bin/python3.13` with PYTHONUSERBASE=/opt1/pyuser has
    the packages and the room. Asking the wrong one produces a confident answer
    about an interpreter that will never serve.
    """

    def test_python_bin_is_configured(self):
        import os

        self.assertEqual(os.environ.get("PYTHON_BIN"), "/usr/bin/python3.13")
        self.assertEqual(server.PYTHON_BIN, "/usr/bin/python3.13")

    def test_child_env_carries_the_user_base(self):
        env = server._child_env()
        self.assertEqual(env["PYTHONUSERBASE"], "/opt1/pyuser")
        self.assertEqual(env["TMPDIR"], "/opt1/tmp")

    def test_no_call_site_hardcodes_python3(self):
        """A bare "python3" in an argv would silently probe the wrong interpreter."""
        import ast

        tree = ast.parse((RIG_DIR / "server.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name not in {"run_command", "create_subprocess_exec"}:
                    continue
                for arg in node.args:
                    elts = arg.elts if isinstance(arg, ast.List) else [arg]
                    for elt in elts:
                        if isinstance(elt, ast.Constant) and elt.value == "python3":
                            self.fail("a call site hardcodes python3; use PYTHON_BIN")

    def test_site_packages_is_derived_from_the_user_base(self):
        site = server._site_packages_dir()
        self.assertIsNotNone(site)
        self.assertIn("/opt1/pyuser", str(site))


def _fake_host(
    *,
    install_free=249 * GB,
    build_free=249 * GB,
    weights_free=249 * GB,
    ram_avail=6.4 * GB,
    ram_total=7.8 * GB,
    swap=0,
    installed=True,
):
    """Patch every filesystem and memory probe.

    PER-PATH DISK, because that is the trap this host sets: three filesystems, and
    a single figure for all of them is what made the first draft of this rig wrong
    in both directions at once.
    """
    free_by_kind = {"site-packages": install_free, "tmp": build_free}

    def fake_disk_free(path):
        text = str(path)
        if "site-packages" in text or "pyuser" in text:
            return int(free_by_kind["site-packages"]), 263 * GB
        if "tmp" in text:
            return int(free_by_kind["tmp"]), 263 * GB
        return int(weights_free), 263 * GB

    mem = {
        "MemTotal": int(ram_total),
        "MemAvailable": int(ram_avail),
        "SwapTotal": int(swap),
        "SwapFree": int(swap),
    }
    return (
        patch.object(server, "_disk_free", side_effect=fake_disk_free),
        patch.object(server, "_meminfo", return_value=mem),
        patch.object(server, "_vllm_files_present", return_value=installed),
    )


class TestCapacity(unittest.TestCase):
    """The arithmetic is the rig, and its subject is three filesystems.

    THE FIRST DRAFT READ `df /`, SAW 4.23 GB AND CONCLUDED "disk binds". It was
    wrong twice: `~/.cache` is a symlink onto a 250 GB volume so the checkpoint had
    room, and `/tmp` is a third filesystem with 3.88 GB that a default pip install
    would have failed in anyway. These tests pin the per-path model.
    """

    def test_weights_and_install_are_measured_on_different_paths(self):
        d, m, i = _fake_host(install_free=1 * GB, weights_free=249 * GB, installed=False)
        with d, m, i:
            c = server._capacity()
        self.assertEqual(c["binding"], "install disk")
        # The weights term is NOT short: a single-filesystem model would have said
        # it was.
        shortfalls = dict((name, amount) for name, amount, _ in c["short"])
        self.assertNotIn("weights disk", shortfalls)

    def test_tmpdir_can_bind_on_its_own(self):
        """The least visible failure on this host: packages have room, pip has
        nowhere to unpack them."""
        d, m, i = _fake_host(install_free=249 * GB, build_free=1 * GB, installed=False)
        with d, m, i:
            c = server._capacity()
        self.assertEqual(c["binding"], "build disk (TMPDIR)")

    def test_an_existing_install_costs_no_disk(self):
        d, m, i = _fake_host(installed=True)
        with d, m, i:
            c = server._capacity()
        self.assertEqual(c["install_need"], 0)
        self.assertEqual(c["build_need"], 0)

    def test_vram_fits_on_this_part_and_the_rig_says_so(self):
        """`gpu-vllm-g4dn-2b` MEASURED 9.8 GiB of weights in 15.0 GiB usable and
        still got a 329,579-token KV pool. A rig that reports VRAM as the blocker
        on a T4 has its arithmetic wrong."""
        d, m, i = _fake_host()
        with d, m, i:
            c = server._capacity()
        self.assertTrue(c["vram_fits"])
        self.assertGreater(c["kv_pool"], 2 * GB)

    def test_host_ram_is_not_a_hard_term(self):
        """vLLM mmaps and copies shard by shard, so peak RSS is far below the
        checkpoint. Asserting weights <= RAM would condemn this host on
        arithmetic nobody has measured."""
        d, m, i = _fake_host(ram_avail=1 * GB, ram_total=2 * GB)
        with d, m, i:
            c = server._capacity()
        self.assertTrue(c["fits"], "host RAM must not veto the budget")

    def test_no_swap_is_recorded_rather_than_assumed(self):
        d, m, i = _fake_host(swap=0)
        with d, m, i:
            c = server._capacity()
        self.assertEqual(c["swap_total"], 0)


class TestCapacityReport(unittest.IsolatedAsyncioTestCase):
    async def test_names_the_binding_term(self):
        d, m, i = _fake_host(install_free=1 * GB, installed=False)
        with d, m, i:
            out = await server.check_host_capacity()
        self.assertIn("❌", out)
        self.assertIn("INSTALL DISK BINDS", out)

    async def test_prints_the_redirections_when_not_installed(self):
        """A printed install command without PYTHONUSERBASE/TMPDIR is a command
        that fails on this host, so the report must carry them."""
        d, m, i = _fake_host(installed=False)
        with d, m, i:
            out = await server.check_host_capacity()
        self.assertIn("PYTHONUSERBASE=/opt1/pyuser", out)
        self.assertIn("TMPDIR=/opt1/tmp", out)

    async def test_presence_of_files_is_never_reported_as_working(self):
        d, m, i = _fake_host(installed=True)
        with d, m, i:
            out = await server.check_host_capacity()
        self.assertIn("not the same as a working install", out)
        self.assertIn("verify_gpu_arch", out)

    async def test_refuses_to_size_kv_from_the_geometry_figure(self):
        d, m, i = _fake_host()
        with d, m, i:
            out = await server.check_host_capacity()
        self.assertIn("Do not size KV from 18 KiB/token", out)
        self.assertIn("329,579", out)


class TestGpuArchProbe(unittest.IsolatedAsyncioTestCase):
    """A CPU torch is the state this host is actually in, so it is the case with
    the most tests."""

    async def test_a_cpu_torch_is_called_out_as_no_answer_about_sm75(self):
        probe = (0, "torch 2.11.0+cpu\ntorch.version.cuda None\narch list: []\n", "")
        with (
            patch.object(server, "run_command", AsyncMock(return_value=(0, "Tesla T4, 7.5", ""))),
            patch.object(server, "_probe", AsyncMock(return_value=probe)),
        ):
            out = await server.verify_gpu_arch()
        self.assertIn("CPU BUILD OF TORCH", out)
        self.assertIn("not an answer about SM 7.5", out)

    async def test_a_working_cuda_torch_reports_half_the_question_done(self):
        probe = (
            0,
            "torch 2.13.0+cu130\ntorch.version.cuda 13.0\narch list: ['sm_75', 'sm_80']\nfp16 matmul ok: True\n",
            "",
        )
        with (
            patch.object(server, "run_command", AsyncMock(return_value=(0, "Tesla T4, 7.5", ""))),
            patch.object(server, "_probe", AsyncMock(return_value=probe)),
        ):
            out = await server.verify_gpu_arch()
        self.assertIn("✅", out)
        self.assertIn("half the question", out)

    async def test_missing_sm75_is_the_rigs_bad_answer(self):
        probe = (0, "torch 2.13.0+cu130\narch list: ['sm_80', 'sm_90']\n", "")
        with (
            patch.object(server, "run_command", AsyncMock(return_value=(0, "Tesla T4, 7.5", ""))),
            patch.object(server, "_probe", AsyncMock(return_value=probe)),
        ):
            out = await server.verify_gpu_arch()
        self.assertIn("No SM 7.5 in the arch list", out)

    async def test_the_48kib_static_limit_is_explained_when_printed(self):
        probe = (
            0,
            "torch 2.13.0+cu130\narch list: ['sm_75']\nshared mem per block (static): 49152\nfp16 matmul ok: True\n",
            "",
        )
        with (
            patch.object(server, "run_command", AsyncMock(return_value=(0, "Tesla T4, 7.5", ""))),
            patch.object(server, "_probe", AsyncMock(return_value=probe)),
        ):
            out = await server.verify_gpu_arch()
        self.assertIn("65,536", out)
        self.assertIn("DEFAULT STATIC", out)


class TestStartRefuses(unittest.IsolatedAsyncioTestCase):
    """Refusing is the decision this rig takes away from the operator."""

    async def test_refuses_when_nothing_is_installed(self):
        d, m, i = _fake_host(installed=False)
        with d, m, i, patch.object(server, "_read_pid", return_value=None):
            out = await server.start_vllm_server()
        self.assertIn("❌", out)
        self.assertIn("not installed", out)

    async def test_refuses_when_a_term_is_short(self):
        d, m, i = _fake_host(weights_free=1 * GB, installed=True)
        with d, m, i, patch.object(server, "_read_pid", return_value=None):
            out = await server.start_vllm_server()
        self.assertIn("Refusing to start", out)
        self.assertIn("weights disk", out)

    async def test_refuses_when_the_clamp_is_merely_unconfirmed(self):
        """FAIL CLOSED. An earlier version refused only on the literal
        "UNPATCHED", so verification that could not import vLLM at all — exactly
        this host's state — returned an unrecognised answer and the serve went
        ahead. A whitelist of known-bad answers lets every unknown-bad answer
        through."""
        d, m, i = _fake_host(installed=True)
        broken = "❌ Could not import `vllm...`: ImportError: libcudart.so.13"
        with (
            d,
            m,
            i,
            patch.object(server, "_read_pid", return_value=None),
            patch.object(server, "verify_turing_patch", AsyncMock(return_value=broken)),
        ):
            out = await server.start_vllm_server()
        self.assertIn("Refusing to start", out)
        self.assertIn("not confirmed present", out)

    async def test_reports_already_running(self):
        with patch.object(server, "_read_pid", return_value=4242):
            out = await server.start_vllm_server()
        self.assertIn("✅", out)
        self.assertIn("4242", out)


class TestPatchVerification(unittest.IsolatedAsyncioTestCase):
    async def test_absent_clamp_names_the_failure_it_prevents(self):
        probe = (0, "__FILE__/x/triton_unified_attention.py\nCLAMP ABSENT\nOCCURRENCES 0\n", "")
        with patch.object(server, "_probe", AsyncMock(return_value=probe)):
            out = await server.verify_turing_patch()
        self.assertIn("UNPATCHED", out)
        self.assertIn("98304", out)

    async def test_double_patch_is_detected(self):
        probe = (0, "__FILE__/x/t.py\nCLAMP PRESENT\nOCCURRENCES 2\n", "")
        with patch.object(server, "_probe", AsyncMock(return_value=probe)):
            out = await server.verify_turing_patch()
        self.assertIn("MORE THAN ONCE", out)

    async def test_single_patch_is_confirmed(self):
        probe = (0, "__FILE__/x/t.py\nCLAMP PRESENT\nOCCURRENCES 1\n", "")
        with patch.object(server, "_probe", AsyncMock(return_value=probe)):
            out = await server.verify_turing_patch()
        self.assertIn("✅ **Patched**", out)


class TestApplyPatchDiagnosis(unittest.IsolatedAsyncioTestCase):
    """"Not installed" and "installed and broken" have nothing in common as
    remedies, and this host is in the second state. An earlier version guessed the
    first and pointed at a disk constraint that does not exist."""

    async def test_broken_install_is_not_reported_as_missing(self):
        failure = (1, "", "ImportError: libcudart.so.13: cannot open shared object file")
        with patch.object(server, "_probe", AsyncMock(return_value=failure)), \
             patch.object(server, "_vllm_files_present", return_value=True):
            out = await server.apply_turing_patch()
        self.assertIn("broken install rather than a missing one", out)
        self.assertIn("verify_gpu_arch", out)
        self.assertNotIn("Nothing is installed", out)

    async def test_missing_install_is_reported_as_missing(self):
        failure = (1, "", "ModuleNotFoundError: No module named 'vllm'")
        with patch.object(server, "_probe", AsyncMock(return_value=failure)), \
             patch.object(server, "_vllm_files_present", return_value=False):
            out = await server.apply_turing_patch()
        self.assertIn("Nothing is installed", out)


class TestStop(unittest.IsolatedAsyncioTestCase):
    async def test_stop_when_not_running(self):
        with patch.object(server, "_read_pid", return_value=None), patch.object(server, "PID_FILE", MagicMock()):
            out = await server.stop_vllm_server()
        self.assertIn("✅", out)
        self.assertIn("Not running", out)

    async def test_stopping_is_not_releasing_capacity(self):
        """The one cost difference from every rig that provisions: the VM and its
        attached GPU are billed whether this process runs or not."""
        with (
            patch.object(server, "_read_pid", return_value=99),
            patch.object(server, "PID_FILE", MagicMock()),
            patch("os.kill"),
        ):
            out = await server.stop_vllm_server()
        self.assertIn("not releasing capacity", out)


class TestQueryModel(unittest.TestCase):
    def test_default_max_tokens_is_generous(self):
        """Gemma 4 emits a thinking block first; a small default truncates
        mid-thought and returns an empty answer."""
        import inspect

        default = inspect.signature(server.query_model).parameters["max_tokens"].default
        self.assertGreaterEqual(default, 512)

    def test_chat_completions_not_raw_completions(self):
        """Raw /v1/completions returns an empty string on -it checkpoints."""
        source = (RIG_DIR / "server.py").read_text()
        self.assertIn("/v1/chat/completions", source)


class TestRunCommand(unittest.IsolatedAsyncioTestCase):
    async def test_missing_binary_returns_127(self):
        rc, _, err = await server.run_command(["definitely-not-a-real-binary-xyz"])
        self.assertEqual(rc, 127)
        self.assertIn("not found", err)

    def test_no_shell_true_anywhere(self):
        """AST, not a substring search: `run_command`'s own docstring says "Never
        shell=True", and a text search cannot tell a rule from a violation."""
        import ast

        tree = ast.parse((RIG_DIR / "server.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "shell":
                        self.fail("a call site passes shell=; every subprocess must be exec")


class TestGetHelp(unittest.IsolatedAsyncioTestCase):
    async def test_lists_tools_and_names_the_ab_twin(self):
        tool = MagicMock()
        tool.name = "check_host_capacity"
        tool.description = "Decide whether this host can install and serve."
        with patch.object(server.mcp, "list_tools", AsyncMock(return_value=[tool])):
            out = await server.get_help()
        self.assertIn("check_host_capacity", out)
        self.assertIn("gpu-vllm-g4dn-2b", out)
        self.assertIn("No provisioning tools", out)


if __name__ == "__main__":
    unittest.main()
