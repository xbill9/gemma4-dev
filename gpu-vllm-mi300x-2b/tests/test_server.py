"""Offline unit tests for the gpu-vllm-mi300x-2b MCP server.

unittest, never pytest. The whole `mcp` package is mocked before `server` is
imported, so nothing here reaches the DigitalOcean API, opens an SSH
connection, starts a container, or needs a token.

A bare MagicMock is NOT enough as the fake: `@mcp.tool()` would then return a
MagicMock instead of the decorated coroutine, and every tool test fails with
"'MagicMock' object can't be awaited" — which reads as a broken server and is
really a broken fake. So `tool()` is a pass-through decorator and the real
functions survive.
"""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


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
sys.modules["mcp.types"] = MagicMock()

import server  # noqa: E402

ACTIVE = {
    "id": 601142018,
    "name": "debian-gpu-mi300x1-192gb-devcloud-atl1",
    "status": "active",
    "size_slug": "gpu-mi300x1-192gb-devcloud",
    "size": {"vcpus": 20, "memory": 245760, "disk": 720, "price_hourly": 1.99},
    "region": {"slug": "atl1"},
    "networks": {
        "v4": [
            {"type": "private", "ip_address": "10.0.0.2"},
            {"type": "public", "ip_address": "203.0.113.7"},
        ]
    },
    "tags": ["gemma"],
}
OFF = dict(ACTIVE, id=987654321, name="mi300-off", status="off", networks={"v4": []})


def run(coro):
    return asyncio.run(coro)


class TestServeArgv(unittest.TestCase):
    """The docker argv is the rig's whole deployment contract."""

    def test_official_image_omits_the_serve_subcommand(self):
        """vllm/vllm-openai-rocm has ENTRYPOINT ["vllm","serve"] already."""
        with patch.object(server, "VLLM_IMAGE", "vllm/vllm-openai-rocm:nightly-rocm100"):
            argv = server._serve_argv()
        image_at = argv.index("vllm/vllm-openai-rocm:nightly-rocm100")
        self.assertEqual(argv[image_at + 1], server.VLLM_MODEL)
        self.assertNotIn("serve", argv)

    def test_vendor_image_needs_the_serve_subcommand(self):
        """rocm/vllm has no entrypoint, so `vllm serve` must be spelled out."""
        vendor = "rocm/vllm:rocm7.13.0_gfx94X-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1"
        with patch.object(server, "VLLM_IMAGE", vendor):
            argv = server._serve_argv()
        image_at = argv.index(vendor)
        self.assertEqual(argv[image_at + 1 : image_at + 4], ["vllm", "serve", server.VLLM_MODEL])

    def test_gemma4_parsers_are_always_passed(self):
        argv = server._serve_argv()
        for flag, value in [
            ("--reasoning-parser", "gemma4"),
            ("--tool-call-parser", "gemma4"),
        ]:
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn("--enable-auto-tool-choice", argv)

    def test_audio_is_disabled(self):
        """No ROCm vLLM image ships vllm[audio], so audio must stay 0."""
        argv = server._serve_argv()
        limits = json.loads(argv[argv.index("--limit-mm-per-prompt") + 1])
        self.assertEqual(limits["audio"], 0)

    def test_render_groups_become_group_add_flags(self):
        with patch.object(server, "RENDER_GIDS", "44,991"):
            argv = server._serve_argv()
        pairs = [argv[i + 1] for i, part in enumerate(argv) if part == "--group-add"]
        self.assertEqual(pairs, ["44", "991"])

    def test_devices_are_mapped_in(self):
        argv = server._serve_argv()
        devices = [argv[i + 1] for i, part in enumerate(argv) if part == "--device"]
        self.assertEqual(sorted(devices), ["/dev/dri", "/dev/kfd"])


class TestImageClassification(unittest.TestCase):
    def test_official_images(self):
        self.assertTrue(server._is_official_image("vllm/vllm-openai-rocm:nightly-rocm100"))
        self.assertTrue(server._is_official_image("vllm/vllm-openai:latest"))

    def test_vendor_images(self):
        self.assertFalse(server._is_official_image("rocm/vllm:rocm10.0.0_x_vllm_0.27.0"))
        self.assertFalse(server._is_official_image("rocm/vllm"))


class TestResolve(unittest.TestCase):
    def test_resolves_by_name_and_by_id(self):
        with patch.object(server, "_droplets", AsyncMock(return_value=[ACTIVE, OFF])):
            self.assertEqual(run(server._resolve("601142018"))["name"], ACTIVE["name"])
            self.assertEqual(run(server._resolve(ACTIVE["name"]))["id"], ACTIVE["id"])

    def test_unknown_name_lists_what_is_tagged(self):
        with patch.object(server, "_droplets", AsyncMock(return_value=[ACTIVE])):
            with self.assertRaises(RuntimeError) as caught:
                run(server._resolve("nope"))
        self.assertIn(ACTIVE["name"], str(caught.exception))

    def test_no_tagged_droplets_names_the_tag(self):
        with patch.object(server, "_droplets", AsyncMock(return_value=[])):
            with self.assertRaises(RuntimeError) as caught:
                run(server._resolve("anything"))
        self.assertIn(server.DROPLET_TAG, str(caught.exception))


class TestReachable(unittest.TestCase):
    def test_active_with_public_ip(self):
        ip, why_not = run(server._reachable(ACTIVE))
        self.assertEqual(ip, "203.0.113.7")
        self.assertIsNone(why_not)

    def test_powered_off_says_so(self):
        ip, why_not = run(server._reachable(OFF))
        self.assertIsNone(ip)
        self.assertIn("start_droplet", why_not)

    def test_active_without_ip_is_still_booting(self):
        ip, why_not = run(server._reachable(dict(ACTIVE, networks={"v4": []})))
        self.assertIsNone(ip)
        self.assertIn("public IPv4", why_not)


class TestRocmSmiParsing(unittest.TestCase):
    """rocm-smi exits 0 when it fails, so the parsed output is the only signal."""

    def test_parses_a_card(self):
        raw = json.dumps(
            {"card0": {"Card Series": "AMD Instinct MI300X VF", "GPU use (%)": "0", "GPU memory use (%)": "3"}}
        )
        table = server._summarize_rocm_smi(raw)
        self.assertIn("MI300X", table)
        self.assertIn("1 GPU(s)", table)

    def test_empty_output_is_not_a_healthy_gpu(self):
        self.assertIsNone(server._summarize_rocm_smi(""))
        self.assertIsNone(server._summarize_rocm_smi("{}"))
        self.assertIsNone(server._summarize_rocm_smi("Driver not initialized"))


class TestGpuStatus(unittest.TestCase):
    def test_missing_kfd_says_reboot_rather_than_debug(self):
        calls = [(0, "", "Driver not initialized"), (0, "missing", "")]
        with patch.object(server, "_remote", AsyncMock(side_effect=calls)):
            out = run(server.gpu_status("mi300"))
        self.assertIn("/dev/kfd", out)
        self.assertIn("reboot_droplet", out)

    def test_healthy_card_reports_a_table(self):
        raw = json.dumps({"card0": {"Card Series": "AMD Instinct MI300X VF", "GPU use (%)": "0"}})
        with patch.object(server, "_remote", AsyncMock(return_value=(0, raw, ""))):
            out = run(server.gpu_status("mi300"))
        self.assertTrue(out.startswith("✅"))


class TestServingStatus(unittest.TestCase):
    def test_no_container_points_at_deploy(self):
        with patch.object(server, "_remote", AsyncMock(return_value=(0, "", ""))):
            out = run(server.serving_status("mi300"))
        self.assertIn("deploy_vllm", out)

    def test_up_but_not_ready_is_distinguishable_from_serving(self):
        models = json.dumps({"object": "list", "data": []})
        with patch.object(server, "_remote", AsyncMock(return_value=(0, "Up 2 minutes|img", ""))):
            with patch.object(server, "_curl_endpoint", AsyncMock(return_value=(0, models, ""))):
                out = run(server.serving_status("mi300"))
        self.assertIn("not answering yet", out)

    def test_serving_reports_the_model(self):
        models = json.dumps({"object": "list", "data": [{"id": server.VLLM_MODEL}]})
        with patch.object(server, "_remote", AsyncMock(return_value=(0, "Up 9 minutes|img", ""))):
            with patch.object(server, "_curl_endpoint", AsyncMock(return_value=(0, models, ""))):
                out = run(server.serving_status("mi300"))
        self.assertTrue(out.startswith("✅"))
        self.assertIn(server.VLLM_MODEL, out)


class TestQueryModel(unittest.TestCase):
    def test_reports_answer_and_reasoning(self):
        body = json.dumps(
            {
                "choices": [{"message": {"content": "CDNA 3.", "reasoning": "thinking about it"}}],
                "usage": {"completion_tokens": 51, "completion_tokens_details": {"reasoning_tokens": 32}},
            }
        )
        with patch.object(server, "_curl_endpoint", AsyncMock(return_value=(0, body, ""))):
            out = run(server.query_model("mi300", "what architecture?"))
        self.assertIn("CDNA 3.", out)
        self.assertIn("32 tokens", out)

    def test_non_json_is_an_error_not_a_crash(self):
        with patch.object(server, "_curl_endpoint", AsyncMock(return_value=(0, "<html>502", ""))):
            out = run(server.query_model("mi300", "hello"))
        self.assertTrue(out.startswith("❌"))


class TestCheckImage(unittest.TestCase):
    """The tool that makes a bad 62 GB pull a free question."""

    def test_missing_convertor_is_called_out(self):
        probe = "vllm 0.27.1.dev5\ntorch 2.12.0\ntransformers 5.16.1\nconvertor None\narchs ['gfx942']"
        with patch.object(server, "_remote", AsyncMock(return_value=(0, probe, ""))):
            out = run(server.check_image("mi300", "rocm/vllm:vllm_0.27.0"))
        self.assertTrue(out.startswith("❌"))
        self.assertIn("Gemma4ModelArchConfigConvertor", out)

    def test_good_image_passes(self):
        probe = (
            "vllm 0.29.1rc1\ntorch 2.12.0\ntransformers 5.17.0\n"
            "convertor <class '...Gemma4ModelArchConfigConvertor'>\narchs ['gfx942']"
        )
        with patch.object(server, "_remote", AsyncMock(return_value=(0, probe, ""))):
            out = run(server.check_image("mi300"))
        self.assertTrue(out.startswith("✅"))

    def test_missing_arch_fails_even_with_the_convertor(self):
        probe = "convertor <class 'Gemma4ModelArchConfigConvertor'>\narchs []"
        with patch.object(server, "_remote", AsyncMock(return_value=(0, probe, ""))):
            out = run(server.check_image("mi300"))
        self.assertTrue(out.startswith("❌"))


class TestVerifyCapabilities(unittest.TestCase):
    def test_counts_four_probes_and_never_probes_audio(self):
        text = json.dumps({"choices": [{"message": {"content": "CDNA 3 architecture."}}]})
        think = json.dumps(
            {
                "choices": [{"message": {"reasoning": "step by step"}}],
                "usage": {"completion_tokens_details": {"reasoning_tokens": 610}},
            }
        )
        tools = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city": "Reykjavik"}'}}]
                        },
                    }
                ]
            }
        )
        vision = json.dumps({"choices": [{"message": {"content": "alternating red and blue squares"}}]})
        responses = [(0, text, ""), (0, think, ""), (0, tools, ""), (0, vision, "")]
        with patch.object(server, "_curl_endpoint", AsyncMock(side_effect=responses)):
            out = run(server.verify_capabilities("mi300"))
        self.assertIn("4/4", out)
        self.assertIn("Audio is not probed", out)

    def test_a_failed_probe_is_reported_not_swallowed(self):
        text = json.dumps({"choices": [{"message": {"content": ""}}]})
        responses = [(0, text, ""), (0, "junk", ""), (0, "junk", ""), (0, "junk", "")]
        with patch.object(server, "_curl_endpoint", AsyncMock(side_effect=responses)):
            out = run(server.verify_capabilities("mi300"))
        self.assertIn("0/4", out)


class TestAnalyzeLogs(unittest.TestCase):
    def test_dead_endpoint_returns_the_raw_tail(self):
        with patch.object(server, "_remote", AsyncMock(return_value=(0, "ERROR engine died", ""))):
            with patch.object(server, "_curl_endpoint", AsyncMock(return_value=(7, "", "connection refused"))):
                out = run(server.analyze_logs("mi300"))
        self.assertIn("raw tail", out)
        self.assertIn("ERROR engine died", out)

    def test_no_logs_is_not_an_analysis(self):
        with patch.object(server, "_remote", AsyncMock(return_value=(0, "", ""))):
            out = run(server.analyze_logs("mi300"))
        self.assertIn("serving_status", out)


class TestStopDropletWarnsAboutBilling(unittest.TestCase):
    def test_billing_warning_is_in_the_output(self):
        with patch.object(server, "_resolve", AsyncMock(return_value=ACTIVE)):
            with patch.object(server, "_api", AsyncMock(return_value={"action": {"id": 1, "status": "in-progress"}})):
                out = run(server.stop_droplet("mi300"))
        self.assertIn("Billing continues", out)


class TestRunCommandNeverUsesAShell(unittest.TestCase):
    def test_missing_binary_is_reported_not_raised(self):
        code, out, err = run(server.run_command(["definitely-not-a-real-binary-xyz"]))
        self.assertEqual(code, 127)
        self.assertIn("not found", err)


class TestGetHelp(unittest.TestCase):
    def test_names_the_rig_and_the_image_caveat(self):
        with patch.object(server.mcp, "list_tools", AsyncMock(return_value=[])):
            out = run(server.get_help())
        self.assertIn(server.MCP_SERVER_NAME, out)
        self.assertIn(server.VLLM_IMAGE, out)
        self.assertIn("check_image", out)


if __name__ == "__main__":
    unittest.main()
