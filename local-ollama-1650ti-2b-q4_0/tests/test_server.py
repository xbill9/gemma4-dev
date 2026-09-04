"""Offline unit tests for the local Ollama rig's MCP server.

unittest, never pytest. The whole `mcp` module is mocked before `server` is
imported, so nothing here touches the network, a subprocess, or the GPU.

Because `mcp` is a MagicMock, `mcp.list_tools()` needs an explicit AsyncMock —
see the get_help test.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

RIG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RIG_DIR))

# Mock the mcp package before importing server. A bare MagicMock is NOT enough
# here: `@mcp.tool()` would then return a MagicMock instead of the decorated
# coroutine, and every tool test fails with "'MagicMock' object can't be
# awaited" — which reads as a broken server and is really a broken fake.


class _FakeFastMCP:
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


_fastmcp_module = MagicMock()
_fastmcp_module.FastMCP = _FakeFastMCP
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.fastmcp"] = _fastmcp_module

import server  # noqa: E402


class TestRigIdentity(unittest.TestCase):
    """The registered name must equal the directory, or two loaded rigs are
    indistinguishable at the call site (root CLAUDE.md)."""

    def test_rig_name_matches_directory(self):
        self.assertEqual(server.RIG_NAME, RIG_DIR.name)
        self.assertEqual(server.RIG_NAME, "local-ollama-1650ti-2b-q4_0")

    def test_server_name_defaults_to_rig_name(self):
        self.assertEqual(server.MCP_SERVER_NAME, server.RIG_NAME)


class TestNoProvisioning(unittest.TestCase):
    """A `local` rig that grows capacity-finding machinery has the wrong name."""

    FORBIDDEN = ("find_tpu", "queued_resource", "gcloud", "boto3", "tpu_zones_status",
                 "create_g5g_instance", "_ssm")

    def test_source_has_no_provisioning(self):
        source = (RIG_DIR / "server.py").read_text()
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        for token in self.FORBIDDEN:
            with self.subTest(token=token):
                self.assertNotIn(token + "(", code)


class TestArtifactIdentity(unittest.TestCase):
    """The whole point of the pair is that the DAEMON chooses the artifact.

    Pointing this rig at the Hub GGUF through a Modelfile would make it a
    llama.cpp rig wearing an Ollama name and delete the only difference the A/B
    can see. `MODEL_NAME` must stay an Ollama tag.
    """

    def test_model_name_is_an_ollama_tag_not_a_path(self):
        self.assertNotIn("/", server.MODEL_NAME)
        self.assertIn(":", server.MODEL_NAME)
        self.assertFalse(server.MODEL_NAME.endswith(".gguf"))

    def test_server_never_creates_or_substitutes_a_model(self):
        """The rig DOES ship a Modelfile now — modelfiles/text-only.Modelfile, which
        drops the projector layer from Ollama's own blob. That is a layer removal,
        not a substitution, and it is done out of band by `make text-only`.

        What must never appear is server.py building a model at runtime, or any
        path to a .gguf on disk: pointing this rig at Google's Hub file would make
        it a llama.cpp rig wearing an Ollama name (see TestProjectorFree)."""
        source = (RIG_DIR / "server.py").read_text()
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        for token in ("/api/create", ".gguf"):
            with self.subTest(token=token):
                self.assertNotIn(token, code)


class TestProjectorFree(unittest.TestCase):
    """The rig serves the projector-free variant, and that is worth 1150 MiB.

    MEASURED 2026-09-04: the stock tag declares vision+audio, so the daemon passes
    --mmproj unconditionally and sits at 2762 MiB of 4096. The variant is 1612 MiB
    — 6 MiB below the llama.cpp sibling. Regressing MODEL_NAME to the stock tag
    would silently give the memory back, and `ollama ps` would not show it.
    """

    MODELFILE = RIG_DIR / "modelfiles" / "text-only.Modelfile"

    def test_model_name_is_the_text_only_variant(self):
        self.assertTrue(server.MODEL_NAME.endswith("-text"), server.MODEL_NAME)

    def test_modelfile_has_exactly_one_from(self):
        """Two FROM lines is what pulls the projector back in — that is the whole
        difference between the stock tag's Modelfile and this one."""
        froms = [ln for ln in self.MODELFILE.read_text().splitlines()
                 if ln.strip().upper().startswith("FROM ")]
        self.assertEqual(len(froms), 1, froms)

    def test_modelfile_uses_ollamas_own_blob_not_the_hub_file(self):
        """Substituting Google's file would make this a llama.cpp rig wearing an
        Ollama name. Dropping a layer from Ollama's own blob does not."""
        text = self.MODELFILE.read_text()
        self.assertIn("/.ollama/models/blobs/sha256-", text)
        self.assertNotIn(".gguf", text)

    def test_modelfile_keeps_the_renderer_parser_and_sampling(self):
        """These are what make the variant comparable to the stock tag: change any
        of them and the A/B measures sampling, not the projector."""
        text = self.MODELFILE.read_text()
        for token in ("RENDERER gemma4", "PARSER gemma4",
                      "temperature 1", "top_k 64", "top_p 0.95"):
            with self.subTest(token=token):
                self.assertIn(token, text)


class TestDaemonEnv(unittest.TestCase):
    """gpu-ollama-g5g-2b-q4_0 lost a provisioning cycle to a missing $HOME:
    `ollama serve` exits 1 without it and crash-looped 13 times in two minutes
    while a readiness poll sat on a dead port and then carried on anyway."""

    def test_home_is_present_in_the_environment(self):
        env = server._daemon_env()
        self.assertTrue(env.get("HOME"), "ollama serve exits 1 with $HOME unset")

    def test_cuda_variant_is_pinned(self):
        """Unpinned, the driver version selects cuda_v13, which may JIT every
        kernel from PTX at load — recording codegen as engine throughput."""
        self.assertEqual(server._daemon_env()["OLLAMA_LLM_LIBRARY"], "cuda_v12")

    def test_keep_alive_never_unloads(self):
        """Ollama unloads after 5 minutes idle by default, so a sweep that pauses
        between cells pays a reload and records it as latency."""
        self.assertTrue(float(server._daemon_env()["OLLAMA_KEEP_ALIVE"]) < 0)

    def test_endpoint_is_local_and_not_the_llamacpp_port(self):
        """8080 is local-llamacpp-1650ti-2b-q4_0's port; sharing it would make
        the pair mutually exclusive and, worse, silently swap which one answered."""
        self.assertIn("127.0.0.1", server.ENDPOINT)
        self.assertNotIn(":8080", server.ENDPOINT)


class TestStartDaemon(unittest.IsolatedAsyncioTestCase):
    async def test_refuses_when_binary_missing(self):
        with patch.object(server, "OLLAMA_BIN", "/nonexistent/ollama"):
            out = await server.start_daemon()
        self.assertIn("❌", out)
        self.assertIn("ollama", out)

    async def test_reports_already_running(self):
        with patch.object(server, "OLLAMA_BIN", __file__), \
             patch.object(server, "_read_pid", return_value=4242):
            out = await server.start_daemon()
        self.assertIn("✅", out)
        self.assertIn("4242", out)


class TestStopDaemon(unittest.IsolatedAsyncioTestCase):
    async def test_stop_when_not_running(self):
        # PID_FILE is a PosixPath, whose methods are read-only — patch the
        # module attribute, not the method on the instance.
        with patch.object(server, "_read_pid", return_value=None), \
             patch.object(server, "PID_FILE", MagicMock()):
            out = await server.stop_daemon()
        self.assertIn("✅", out)
        self.assertIn("Not running", out)


class TestRunCommand(unittest.IsolatedAsyncioTestCase):
    async def test_missing_binary_returns_127(self):
        rc, _, err = await server.run_command(["definitely-not-a-real-binary-xyz"])
        self.assertEqual(rc, 127)
        self.assertIn("not found", err)


def _json_response(payload, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    return resp


def _client_returning(resp, method="post", other=None):
    """A stand-in httpx.AsyncClient.

    BOTH verbs are stubbed even when a test only cares about one, because
    verify_model_resident makes two calls — GET /api/ps for residency and POST
    /api/show for the capabilities it needs to interpret the VRAM gap. Leaving
    the second unstubbed returns a bare MagicMock and fails with "object can't be
    awaited", which reads as a broken tool and is really a broken fake."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    fallback = other if other is not None else _json_response({})
    client.get = AsyncMock(return_value=resp if method == "get" else fallback)
    client.post = AsyncMock(return_value=resp if method == "post" else fallback)
    return client


class TestVerifyModelResident(unittest.IsolatedAsyncioTestCase):
    """There is no N_GPU_LAYERS here, so partial offload cannot be prevented by
    configuration — only detected. A 48%/52% split is a silent throughput cliff."""

    async def test_partial_offload_is_reported_as_failure(self):
        resp = _json_response({"models": [
            {"name": "gemma4:e2b-it-qat", "size": 4_000_000_000,
             "size_vram": 2_000_000_000, "context_length": 8192}]})
        with patch.object(server.httpx, "AsyncClient",
                          return_value=_client_returning(resp, "get")), \
             patch.object(server, "run_command", AsyncMock(return_value=(0, "", ""))):
            out = await server.verify_model_resident()
        self.assertIn("PARTIAL OFFLOAD", out)

    async def test_full_offload_passes(self):
        resp = _json_response({"models": [
            {"name": "gemma4:e2b-it-qat", "size": 1_700_000_000,
             "size_vram": 1_700_000_000, "context_length": 8192}]})
        with patch.object(server.httpx, "AsyncClient",
                          return_value=_client_returning(resp, "get")), \
             patch.object(server, "run_command", AsyncMock(return_value=(0, "", ""))):
            out = await server.verify_model_resident()
        self.assertIn("fully on GPU", out)

    async def test_no_model_loaded_is_not_an_error(self):
        """Ollama loads on first request, so this is the state after start_daemon."""
        resp = _json_response({"models": []})
        with patch.object(server.httpx, "AsyncClient",
                          return_value=_client_returning(resp, "get")):
            out = await server.verify_model_resident()
        self.assertNotIn("❌", out)
        self.assertIn("loads on first request", out)


class TestProjectorDetection(unittest.IsolatedAsyncioTestCase):
    """The expensive mistake is serving a projector-carrying model by accident.

    Nothing announces it: `ollama ps` does not count the mmproj, so the memory is
    simply gone. Both tools therefore read the DECLARED CAPABILITIES rather than
    trusting the tag name — a tag can be renamed, a layer cannot be hidden.
    """

    def test_helper_reads_capabilities_not_the_tag_name(self):
        self.assertTrue(server._has_projector(["completion", "vision", "thinking"]))
        self.assertTrue(server._has_projector(["completion", "audio"]))
        self.assertFalse(server._has_projector(["completion", "tools", "thinking"]))
        self.assertFalse(server._has_projector([]))
        self.assertFalse(server._has_projector(None))

    async def test_model_info_flags_a_projector_carrying_model(self):
        resp = _json_response({
            "details": {"family": "gemma4", "parameter_size": "4.6B",
                        "quantization_level": "Q4_0"},
            "model_info": {"gemma4.context_length": 131072},
            "capabilities": ["completion", "vision", "audio", "tools", "thinking"]})
        with patch.object(server.httpx, "AsyncClient", return_value=_client_returning(resp)):
            out = await server.model_info()
        self.assertIn("⚠️", out)
        self.assertIn("carries the projector", out)
        self.assertIn("make text-only", out)

    async def test_model_info_confirms_the_projector_free_variant(self):
        resp = _json_response({
            "details": {"family": "gemma4", "parameter_size": "4.6B",
                        "quantization_level": "Q4_0"},
            "model_info": {"gemma4.context_length": 131072},
            "capabilities": ["completion", "tools", "thinking"]})
        with patch.object(server.httpx, "AsyncClient", return_value=_client_returning(resp)):
            out = await server.model_info()
        self.assertIn("✅", out)
        self.assertIn("Projector-free", out)
        self.assertIn("1612", out)

    async def test_residency_interprets_the_gap_for_a_projector_free_model(self):
        """The normal gap is CUDA context plus compute buffers and scales with slot
        count (181 MiB at 1 slot, 291 at 32). ~1150 MiB is an mmproj, and on a
        model that should not have one it means the wrong tag answered."""
        ps = _json_response({"models": [
            {"name": "gemma4:e2b-it-qat-text", "size": 1_500_000_000,
             "size_vram": 1_500_000_000, "context_length": 8192}]})
        show = _json_response({"capabilities": ["completion", "tools", "thinking"]})
        with patch.object(server.httpx, "AsyncClient",
                          return_value=_client_returning(ps, "get", other=show)), \
             patch.object(server, "run_command",
                          AsyncMock(return_value=(0, "1, llama-server, 1612 MiB", ""))):
            out = await server.verify_model_resident()
        self.assertIn("fully on GPU", out)
        self.assertIn("181 MiB at 1 slot", out)
        self.assertIn("1150 MiB means a projector-carrying model is loaded", out)

    async def test_residency_flags_a_projector_when_one_is_declared(self):
        ps = _json_response({"models": [
            {"name": "gemma4:e2b-it-qat", "size": 1_700_000_000,
             "size_vram": 1_700_000_000, "context_length": 8192}]})
        show = _json_response({"capabilities": ["completion", "vision", "audio"]})
        with patch.object(server.httpx, "AsyncClient",
                          return_value=_client_returning(ps, "get", other=show)), \
             patch.object(server, "run_command",
                          AsyncMock(return_value=(0, "1, llama-server, 2762 MiB", ""))):
            out = await server.verify_model_resident()
        self.assertIn("⚠️", out)
        self.assertIn("1154 MiB", out)


class TestQueryModelReasoning(unittest.IsolatedAsyncioTestCase):
    """Gemma 4 emits a thinking block and Ollama can DISCARD it.

    MEASURED 2026-09-04: at num_predict=128 the block has not closed, so
    /api/generate returns done_reason "length" with both `content` and
    `thinking` empty — 128 tokens generated and zero characters returned. The
    llama.cpp sibling at least exposes the partial thought in reasoning_content.
    """

    async def test_empty_content_is_not_reported_as_success(self):
        resp = _json_response({
            "message": {"role": "assistant", "content": "", "thinking": ""},
            "done_reason": "length", "eval_count": 128,
            "eval_duration": 1_800_000_000})
        with patch.object(server.httpx, "AsyncClient", return_value=_client_returning(resp)):
            out = await server.query_model("hi", max_tokens=128)
        self.assertNotIn("✅", out)
        self.assertIn("No answer text", out)
        self.assertIn("discarded", out)

    async def test_answer_reports_thinking_was_suppressed(self):
        resp = _json_response({
            "message": {"role": "assistant", "content": "TPU v1, TPU v2, TPU v3",
                        "thinking": "x" * 916},
            "done_reason": "stop", "eval_count": 278,
            "eval_duration": 4_000_000_000})
        with patch.object(server.httpx, "AsyncClient", return_value=_client_returning(resp)):
            out = await server.query_model("hi")
        self.assertIn("✅", out)
        self.assertIn("TPU v1", out)
        self.assertIn("916 chars of thinking", out)

    async def test_server_side_gauge_is_labelled_as_such(self):
        """gpu-ollama-g5g-2b-q4_0: "do not publish 3x vLLM off this"."""
        resp = _json_response({
            "message": {"role": "assistant", "content": "ok", "thinking": ""},
            "done_reason": "stop", "eval_count": 128,
            "eval_duration": 1_828_674_000})
        with patch.object(server.httpx, "AsyncClient", return_value=_client_returning(resp)):
            out = await server.query_model("hi")
        self.assertIn("SERVER-SIDE", out)
        self.assertIn("70.0", out)

    def test_default_max_tokens_is_generous(self):
        import inspect
        default = inspect.signature(server.query_model).parameters["max_tokens"].default
        self.assertGreaterEqual(default, 512,
                                "a small default truncates mid-thought and returns nothing")


class TestRateHelpers(unittest.TestCase):
    def test_decode_rate_from_nanoseconds(self):
        self.assertAlmostEqual(
            server._decode_rate({"eval_count": 128, "eval_duration": 1_828_674_000}),
            69.99, places=1)

    def test_missing_fields_return_none(self):
        self.assertIsNone(server._decode_rate({}))
        self.assertIsNone(server._prefill_rate({"prompt_eval_count": 21}))


class TestSweepHarness(unittest.TestCase):
    """The two fixes this rig's first sweep needed, pinned so they cannot regress.

    Both were MEASURED against the daemon's own slot log on 2026-09-04 and both
    silently measure the prompt cache instead of prefill when wrong.
    """

    def _sweep_source(self):
        return (RIG_DIR / "sweep.py").read_text()

    def test_cache_prompt_is_not_relied_on(self):
        """Ollama drops unknown OpenAI fields in translation, so cache_prompt
        never reaches llama-server: an 8-token request carrying it still logged
        "cached n_tokens = 660" of 661."""
        import re
        source = self._sweep_source()
        match = re.search(r"^DEFEAT_PROMPT_CACHE = (\w+)$", source, re.M)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "False")

    def test_prompt_rng_is_not_seeded(self):
        """A constant seed makes every process emit the SAME shuffled prompts,
        and Ollama's prompt cache outlives the client: a second run returned a
        652-token prompt in 0.15 s against 2.17 s cold — prefill "measured" at
        16000 t/s."""
        self.assertIn("_RNG = random.Random()", self._sweep_source())


class TestGetHelp(unittest.IsolatedAsyncioTestCase):
    async def test_lists_tools(self):
        tool = MagicMock()
        tool.name = "gpu_status"
        tool.description = "Report the local GPU: name, compute capability, VRAM."
        with patch.object(server.mcp, "list_tools", AsyncMock(return_value=[tool])):
            out = await server.get_help()
        self.assertIn("gpu_status", out)
        self.assertIn("control plane", out)


if __name__ == "__main__":
    unittest.main()
