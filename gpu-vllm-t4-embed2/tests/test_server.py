"""Offline unit tests for the EmbeddingGemma 2 T4 rig's MCP server.

unittest, never pytest. The whole `mcp` module is mocked before `server` is
imported, so nothing here touches the network, a subprocess, the GPU, or a real
filesystem probe.
"""

import ast
import json
import os
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
import patch_triton_encoder_turing as clamp  # noqa: E402

GB = 10**9


def _code_only_tree():
    """server.py's AST with every docstring removed — the docstrings explain the
    things the code must not do, and a text search cannot tell the two apart."""
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
    return tree


class TestRigIdentity(unittest.TestCase):
    def test_rig_name_matches_directory(self):
        self.assertEqual(server.RIG_NAME, RIG_DIR.name)
        self.assertEqual(server.RIG_NAME, "gpu-vllm-t4-embed2")

    def test_server_name_defaults_to_rig_name(self):
        self.assertEqual(server.MCP_SERVER_NAME, server.RIG_NAME)

    def test_slot_4_is_the_model_family_not_a_size(self):
        """EmbeddingGemma 2 has no Gemma 4 size — one checkpoint loads as 271M to
        744M parameters by tower — so @NAMING.md gives it `embed2`."""
        self.assertEqual(server.RIG_NAME.split("-")[3], "embed2")
        self.assertEqual(server.MODEL_NAME, "google/embeddinggemma-2")

    def test_registration_agrees_in_every_place(self):
        """.mcp.json is gitignored, so only the committed three are checked."""
        plugin = json.loads((RIG_DIR / ".claude-plugin" / "plugin.json").read_text())
        self.assertEqual(plugin["name"], server.RIG_NAME)
        self.assertEqual(list(plugin["mcpServers"]), [server.RIG_NAME])
        codex = (RIG_DIR / ".codex" / "config.toml").read_text()
        self.assertIn(f"[mcp_servers.{server.RIG_NAME}]", codex)
        local = json.loads((RIG_DIR / ".claude" / "settings.local.json").read_text())
        self.assertEqual(local["enabledMcpjsonServers"], [server.RIG_NAME])

    def test_every_gated_codex_tool_exists(self):
        """A gate on a tool name that does not exist fails open and says nothing."""
        codex = (RIG_DIR / ".codex" / "config.toml").read_text()
        prefix = f"[mcp_servers.{server.RIG_NAME}.tools."
        gated = [line[len(prefix) : -1] for line in codex.splitlines() if line.startswith(prefix)]
        self.assertTrue(gated)
        for name in gated:
            self.assertTrue(hasattr(server, name), f".codex gates {name}, which server.py does not define")


class TestNoCloudControlPlane(unittest.TestCase):
    """The GPU is attached; nothing here may provision, as in gpu-vllm-t4-2b."""

    FORBIDDEN = frozenset({"boto3", "botocore", "ec2", "ssm", "instance_id", "queued_resource", "find_tpu"})

    def test_source_names_no_cloud_control_plane(self):
        names = set()
        for node in ast.walk(_code_only_tree()):
            if isinstance(node, ast.Name):
                names.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                names.add(node.attr.lower())
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name.lower())
        self.assertEqual(names & self.FORBIDDEN, set())

    def test_no_cloud_binary_is_ever_executed(self):
        for node in ast.walk(ast.parse((RIG_DIR / "server.py").read_text())):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name in {"run_command", "create_subprocess_exec", "Popen"} and node.args:
                    head = (
                        node.args[0].elts[0]
                        if isinstance(node.args[0], ast.List) and node.args[0].elts
                        else node.args[0]
                    )
                    if isinstance(head, ast.Constant):
                        self.assertNotIn(head.value, {"gcloud", "aws", "kubectl", "docker"})


class TestEmbeddingModelPolicy(unittest.TestCase):
    def test_dtype_is_float32(self):
        """Google: float16 is incompatible with this model. Turing has no bf16
        datapath and vLLM casts bf16 to fp16 on SM 7.5 without failing, so float32
        is the only dtype that serves what was asked for."""
        self.assertEqual(server.DTYPE, "float32")

    def test_runner_is_pooling(self):
        self.assertEqual(server.RUNNER, "pooling")
        self.assertIn("--runner", server._serve_cmd())

    def test_matryoshka_sizes_are_declared(self):
        """config.json declares none, and without the override vLLM rejects every
        `dimensions` request. Listing the trained sizes makes others a loud 400."""
        overrides = json.loads(server.HF_OVERRIDES)
        self.assertEqual(sorted(overrides["matryoshka_dimensions"]), [128, 256, 512, 768])
        cmd = server._serve_cmd()
        self.assertEqual(json.loads(cmd[cmd.index("--hf-overrides") + 1]), overrides)

    def test_audio_is_off_because_its_extras_are_not_installed(self):
        limits = json.loads(server.LIMIT_MM_PER_PROMPT)
        self.assertEqual(limits["audio"], 0)
        cmd = server._serve_cmd()
        self.assertEqual(json.loads(cmd[cmd.index("--limit-mm-per-prompt") + 1]), limits)

    def test_no_generative_serving_flags(self):
        """No KV cache exists for encoder-only layers; these would be dead config."""
        cmd = server._serve_cmd()
        for flag in ("--kv-cache-dtype", "--max-num-seqs", "--enable-auto-tool-choice", "--reasoning-parser"):
            self.assertNotIn(flag, cmd)

    def test_revision_is_pinned_to_the_measured_snapshot(self):
        cmd = server._serve_cmd()
        self.assertEqual(cmd[cmd.index("--revision") + 1], "914f7f89142e33e77833254d9c9b90c3cef7303b")

    def test_prefixes_are_the_trained_ones_and_keep_their_trailing_space(self):
        self.assertEqual(server.QUERY_PREFIX, "task: search result | query: ")
        self.assertEqual(server.DOCUMENT_PREFIX, "title: none | text: ")

    def test_tpu_env_does_not_carry_the_prefixes(self):
        """`include tpu.env` + `export` in the Makefile would hand literal quotes to
        the environment, where they beat dotenv."""
        env = (RIG_DIR / "tpu.env").read_text()
        self.assertNotRegex(env, r"(?m)^(QUERY|DOCUMENT)_PREFIX=")

    def test_no_chat_completion_path(self):
        for node in ast.walk(_code_only_tree()):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertNotIn("/v1/chat/completions", node.value)


class TestEncoderClamp(unittest.TestCase):
    """The patch targets the ENCODER kernel; the siblings' decoder clamp is absent."""

    UPSTREAM = """def context_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
):
    Lq, Lk, _ = q.shape[-1], k.shape[-1], v.shape[-1]
    BLOCK = get_block_size(q.dtype)
    if Lk >= 512:
        BLOCK = min(BLOCK, 32)

    sm_scale = 1.0
    current_platform
    _fwd_kernel[grid](
        q,
        num_stages=1,
        Lk=Lk,
    )
"""

    def test_the_sentinel_matches_the_patch_script(self):
        self.assertEqual(server._PATCH_SENTINEL, clamp.SENTINEL)

    def test_the_marker_names_this_rig(self):
        self.assertTrue(clamp.MARKER.startswith(f"# {server.RIG_NAME}:"))

    def test_the_decoder_patch_is_not_carried(self):
        self.assertFalse((RIG_DIR / "patch_triton_turing.py").exists())
        self.assertEqual(server._TRITON_MODULE, "vllm.v1.attention.ops.triton_prefill_attention")

    def test_patch_inserts_clamp_and_copy_back_once(self):
        out = clamp.patch(self.UPSTREAM)
        self.assertEqual(out.count(clamp.SENTINEL), 1)
        self.assertIn("BLOCK = min(BLOCK, 16)", out)
        self.assertIn("_turing_out.copy_(o)", out)
        # The clamp sits before sm_scale; the copy-back after the launch.
        self.assertLess(out.index(clamp.SENTINEL), out.index("sm_scale"))
        self.assertGreater(out.index("_turing_out.copy_(o)"), out.index("Lk=Lk,"))
        compile(out.replace("_fwd_kernel[grid]", "_fwd_kernel"), "x", "exec")

    def test_patch_is_idempotent(self):
        once = clamp.patch(self.UPSTREAM)
        self.assertEqual(clamp.patch(once), once)

    def test_patch_refuses_a_restructured_file(self):
        with self.assertRaises(SystemExit) as cm, patch("sys.stderr"):
            clamp.patch(self.UPSTREAM.replace("BLOCK = min(BLOCK, 32)", "BLOCK = 32"))
        self.assertEqual(cm.exception.code, 2)


class TestTargetInterpreter(unittest.TestCase):
    def test_python_bin_is_configured(self):
        self.assertEqual(server.PYTHON_BIN, "/usr/bin/python3.13")

    def test_user_base_is_this_rigs_own_not_the_siblings(self):
        """/opt1/pyuser holds the generative rigs' vLLM 0.29 and their clamp."""
        env = server._child_env()
        self.assertEqual(env["PYTHONUSERBASE"], "/opt1/pyuser-embed2")
        self.assertNotEqual(env["PYTHONUSERBASE"], "/opt1/pyuser")

    def test_no_call_site_hardcodes_python3(self):
        for node in ast.walk(ast.parse((RIG_DIR / "server.py").read_text())):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name in {"run_command", "create_subprocess_exec", "Popen"}:
                    for arg in node.args:
                        for elt in arg.elts if isinstance(arg, ast.List) else [arg]:
                            if isinstance(elt, ast.Constant) and elt.value == "python3":
                                self.fail("a call site hardcodes python3; use PYTHON_BIN")
        self.assertEqual(server._serve_cmd()[0], server.PYTHON_BIN)

    def test_engine_is_launched_with_popen_not_asyncio(self):
        """asyncio kills its children when a one-shot `make serve` interpreter exits."""
        calls = {
            getattr(node.func, "attr", None) or getattr(node.func, "id", None): node
            for node in ast.walk(self._fn("start_vllm_server"))
            if isinstance(node, ast.Call)
        }
        self.assertIn("Popen", calls)
        self.assertNotIn("create_subprocess_exec", calls)
        kw = {k.arg: k.value for k in calls["Popen"].keywords}
        self.assertTrue(kw["start_new_session"].value)

    def _fn(self, name):
        for node in ast.walk(ast.parse((RIG_DIR / "server.py").read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return node
        raise AssertionError(name)


def _fake_host(*, install_free=158 * GB, build_free=158 * GB, weights_free=158 * GB, installed=True):
    def fake_disk_free(path):
        text = str(path)
        if "site-packages" in text or "pyuser" in text:
            return int(install_free), 263 * GB
        if "tmp" in text:
            return int(build_free), 263 * GB
        return int(weights_free), 263 * GB

    mem = {"MemTotal": int(7.8 * GB), "MemAvailable": int(6.4 * GB), "SwapTotal": 0, "SwapFree": 0}
    return (
        patch.object(server, "_disk_free", side_effect=fake_disk_free),
        patch.object(server, "_meminfo", return_value=mem),
        patch.object(server, "_vllm_files_present", return_value=installed),
    )


class TestCapacity(unittest.TestCase):
    def test_float32_doubles_the_loaded_weights(self):
        self.assertEqual(server._loaded_weight_bytes(1_000), 2_000)

    def test_vram_fits_with_room_to_spare(self):
        d, m, i = _fake_host()
        with d, m, i:
            c = server._capacity()
        self.assertTrue(c["fits"])
        self.assertLess(c["vram_need"], c["vram"] / 3)

    def test_install_and_build_disks_bind_separately(self):
        d, m, i = _fake_host(install_free=1 * GB, installed=False)
        with d, m, i:
            self.assertEqual(server._capacity()["binding"], "install disk")
        d, m, i = _fake_host(build_free=1 * GB, installed=False)
        with d, m, i:
            self.assertEqual(server._capacity()["binding"], "build disk (TMPDIR)")

    def test_an_existing_install_costs_no_disk(self):
        d, m, i = _fake_host(install_free=0, build_free=0, installed=True)
        with d, m, i:
            self.assertTrue(server._capacity()["fits"])


class TestCapacityReport(unittest.IsolatedAsyncioTestCase):
    async def test_install_command_targets_this_rigs_user_base(self):
        d, m, i = _fake_host(installed=False)
        with d, m, i:
            out = await server.check_host_capacity()
        self.assertIn("PYTHONUSERBASE=/opt1/pyuser-embed2", out)
        self.assertIn("requirements-lock.txt", out)

    async def test_names_host_ram_as_the_term_to_watch(self):
        d, m, i = _fake_host()
        with d, m, i:
            out = await server.check_host_capacity()
        self.assertIn("Host RAM is the term to watch", out)


class TestGpuArchProbe(unittest.IsolatedAsyncioTestCase):
    async def _run(self, body, rc=0):
        with (
            patch.object(server, "run_command", AsyncMock(return_value=(0, "Tesla T4, 7.5", ""))),
            patch.object(server, "_probe", AsyncMock(return_value=(rc, body, ""))),
        ):
            return await server.verify_gpu_arch()

    async def test_old_vllm_is_named_as_the_wrong_user_base(self):
        out = await self._run(
            "torch 2.13.0+cu130\narch list: ['sm_75']\nfp32 matmul ok: True\nvllm 0.29.0\n"
            "EmbeddingGemma2Model registered: False\n"
        )
        self.assertIn("does not know `EmbeddingGemma2Model`", out)

    async def test_old_transformers_is_named(self):
        out = await self._run("ModuleNotFoundError: No module named 'transformers.models.embedding_gemma2'")
        self.assertIn("5.19.0", out)

    async def test_a_working_stack_is_half_the_question(self):
        out = await self._run(
            "torch 2.13.0+cu130\narch list: ['sm_75', 'sm_80']\nfp32 matmul ok: True\n"
            "EmbeddingGemma2Model registered: True\ntransformers embedding_gemma2: present\n"
        )
        self.assertIn("✅", out)
        self.assertIn("apply_encoder_patch", out)


class TestStartRefuses(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pid = patch.object(server, "_read_pid", return_value=None)
        self.pid.start()

    def tearDown(self):
        self.pid.stop()

    async def test_refuses_when_nothing_is_installed(self):
        with patch.object(server, "_vllm_files_present", return_value=False):
            out = await server.start_vllm_server()
        self.assertIn("not installed", out)

    async def test_refuses_when_the_clamp_is_merely_unconfirmed(self):
        """An import failure is neither "patched" nor "UNPATCHED"; it must refuse."""
        d, m, i = _fake_host()
        with (
            d,
            m,
            i,
            patch.object(server, "verify_encoder_patch", AsyncMock(return_value="❌ Could not import")),
            patch.object(server.subprocess, "Popen") as popen,
        ):
            out = await server.start_vllm_server()
        self.assertIn("Refusing to start", out)
        popen.assert_not_called()

    async def test_reports_already_running(self):
        with patch.object(server, "_read_pid", return_value=4242):
            out = await server.start_vllm_server()
        self.assertIn("Already running", out)


class TestPatchVerification(unittest.IsolatedAsyncioTestCase):
    async def _run(self, out):
        with patch.object(server, "_probe", AsyncMock(return_value=(0, out, ""))):
            return await server.verify_encoder_patch()

    async def test_absent_clamp_names_the_failure(self):
        self.assertIn("69632", await self._run("__FILE__x\nCLAMP ABSENT\nOCCURRENCES 0\n"))

    async def test_double_patch_is_detected(self):
        self.assertIn("MORE THAN ONCE", await self._run("__FILE__x\nCLAMP PRESENT\nOCCURRENCES 2\n"))

    async def test_single_patch_is_confirmed(self):
        self.assertIn("✅ **Patched**", await self._run("__FILE__x\nCLAMP PRESENT\nOCCURRENCES 1\n"))


class TestEmbeddingTools(unittest.IsolatedAsyncioTestCase):
    async def test_embed_text_prepends_the_query_prefix(self):
        fake = AsyncMock(return_value=[[1.0] + [0.0] * 767])
        with patch.object(server, "_embed", fake):
            out = await server.embed_text(["hello"])
        self.assertEqual(fake.call_args.args[0], ["task: search result | query: hello"])
        self.assertIn("dim 768", out)

    async def test_embed_text_document_and_raw(self):
        fake = AsyncMock(return_value=[[1.0]])
        with patch.object(server, "_embed", fake):
            await server.embed_text(["d"], kind="document")
            self.assertEqual(fake.call_args.args[0], ["title: none | text: d"])
            await server.embed_text(["r"], kind="raw")
            self.assertEqual(fake.call_args.args[0], ["r"])
        self.assertIn("❌", await server.embed_text(["x"], kind="chat"))

    async def test_rank_documents_orders_by_dot_product(self):
        fake = AsyncMock(return_value=[[1.0, 0.0], [0.2, 0.98], [0.9, 0.44]])
        with patch.object(server, "_embed", fake):
            out = await server.rank_documents("q", ["far", "near"])
        self.assertLess(out.index("near"), out.index("far"))
        self.assertTrue(fake.call_args.args[0][1].startswith("title: none | text: "))

    async def test_verify_embeddings_passes_on_a_sane_server(self):
        q = [1.0] + [0.0] * 767
        rel = [0.8, 0.6] + [0.0] * 766
        irr = [0.1] + [0.0] * 766 + [0.99]
        fake = AsyncMock(side_effect=[[q, rel, irr], [[1.0] + [0.0] * 255]])
        with patch.object(server, "_embed", fake):
            out = await server.verify_embeddings()
        self.assertTrue(out.startswith("✅"), out)

    async def test_verify_embeddings_fails_on_the_wrong_dimension(self):
        v = [1.0] + [0.0] * 383
        fake = AsyncMock(side_effect=[[v, v, v], [v[:256]]])
        with patch.object(server, "_embed", fake):
            out = await server.verify_embeddings()
        self.assertTrue(out.startswith("❌"))


class TestRunCommand(unittest.IsolatedAsyncioTestCase):
    async def test_missing_binary_returns_127(self):
        rc, _, _ = await server.run_command(["/nonexistent/binary-xyz"])
        self.assertEqual(rc, 127)

    def test_no_shell_true_anywhere(self):
        for path in (RIG_DIR / "server.py", RIG_DIR / "patch_triton_encoder_turing.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.keyword) and node.arg == "shell":
                    self.fail(f"shell= keyword in {path.name}")


class TestGetHelp(unittest.IsolatedAsyncioTestCase):
    async def test_says_it_is_not_a_chat_model(self):
        with patch.object(server.mcp, "list_tools", AsyncMock(return_value=[])):
            out = await server.get_help()
        self.assertIn("not a chat model", out)
        self.assertIn("Not an A/B twin", out)


if __name__ == "__main__":
    os.chdir(RIG_DIR)
    unittest.main()
