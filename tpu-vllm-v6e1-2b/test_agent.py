import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Mocking FastMCP and other dependencies before importing server
mock_mcp = MagicMock()
sys.modules["mcp"] = MagicMock()
sys.modules["mcp.server"] = MagicMock()
sys.modules["mcp.server.fastmcp"] = MagicMock()
sys.modules["mcp.server.fastmcp"].FastMCP = MagicMock(return_value=mock_mcp)


# Mock decorative tools
def mock_decorator(*args, **kwargs):
    def wrapper(func):
        return func

    return wrapper


mock_mcp.tool = mock_decorator
mock_mcp.resource = mock_decorator

sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.storage"] = MagicMock()
sys.modules["google.cloud.logging"] = MagicMock()
sys.modules["google.cloud.secretmanager"] = MagicMock()

# Now import the functions to test
from server import (  # noqa: E402
    MODEL_NAME,
    PROVISIONING_MODELS,
    RESOURCE_ID,
    TPU_QUOTA_ID,
    TPU_SPOT_QUOTA_ID,
    _create_queued_resource,
    _discover_vllm_node,
    _lookup_tpu_rate,
    _parse_topology,
    _provisioning_flags,
    _quota_id_for,
    _resolve_node_id,
    _status_model,
    estimate_deployment_cost,
    get_help,
    get_metrics,
    get_model_details,
    get_vllm_deployment_config,
    query_queued_gemma4_with_stats,
    save_hf_token,
    verify_model_health,
)


class TestDevOpsAgent(unittest.IsolatedAsyncioTestCase):
    def test_model_name_default(self):
        """Verify the default model is Gemma 4."""
        self.assertEqual(MODEL_NAME, "google/gemma-4-E2B-it")

    @patch("server.get_secret", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_get_vllm_deployment_config(self, mock_run_command, mock_get_secret):
        """Test TPU deployment config generation."""
        mock_get_secret.return_value = "dummy-hf-token"
        # Mock run_command to prevent actual gcloud calls during this test
        mock_run_command.return_value = 0, "", ""

        config = await get_vllm_deployment_config(service_name="test-vllm", model_name="google/gemma-4-E2B-it")
        self.assertIn("gcloud alpha compute tpus tpu-vm create test-vllm", config)
        self.assertIn("--accelerator-type=v6e-1", config)
        self.assertIn("--version=v2-alpha-tpuv6e", config)

        self.assertIn("vllm/vllm-tpu:nightly", config)
        self.assertIn("google/gemma-4-E2B-it", config)

    @patch("server.get_vllm_client", new_callable=AsyncMock)
    @patch("server.discover_vllm_url", new_callable=AsyncMock)
    async def test_verify_model_health_success(self, mock_discover_url, mock_get_client):
        """Test successful model health check."""
        mock_discover_url.return_value = "http://test-url:8000"
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "READY"

        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await verify_model_health()
        self.assertIn("✅ Model health check PASSED.", result)
        self.assertIn("READY", result)

    @patch("server.get_vllm_client", new_callable=AsyncMock)
    async def test_query_queued_gemma4_with_stats_success(self, mock_get_client):
        """Test query with performance metrics."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_stream = AsyncMock()

        # Create chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"

        mock_stream.__aiter__.return_value = [chunk1, chunk2]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        result = await query_queued_gemma4_with_stats("Hi")
        self.assertIn("Hello world", result)
        self.assertIn("Performance Stats", result)
        self.assertIn("TTFT", result)

    @patch("server.discover_vllm_url", new_callable=AsyncMock)
    @patch("httpx.AsyncClient", autospec=True)
    async def test_get_model_details_success(self, mock_async_client, mock_discover_url):
        """Test retrieving model stats."""
        mock_discover_url.return_value = "http://test-url:8000"

        # Mock httpx.AsyncClient and its get method
        mock_client_instance = mock_async_client.return_value.__aenter__.return_value

        mock_models_response = MagicMock()
        mock_models_response.json.return_value = {"data": [{"id": "test-model", "max_model_len": 4096}]}
        mock_models_response.status_code = 200

        mock_version_response = MagicMock()
        mock_version_response.json.return_value = {"version": "test-version"}
        mock_version_response.status_code = 200

        mock_health_response = MagicMock()
        mock_health_response.status_code = 200

        mock_metrics_response = MagicMock()
        mock_metrics_response.text = "vllm_requests_running 1"
        mock_metrics_response.status_code = 200

        mock_client_instance.get.side_effect = [
            mock_models_response,
            mock_version_response,
            mock_health_response,
            mock_metrics_response,
        ]

        result = await get_model_details()
        self.assertIn("### 🧩 Model & vLLM Engine Details", result)
        self.assertIn("test-model", result)
        self.assertIn("test-version", result)
        self.assertIn("Healthy", result)
        self.assertIn("vllm_requests_running", result)

    @patch("server.discover_vllm_url", new_callable=AsyncMock)
    @patch("httpx.AsyncClient", autospec=True)
    async def test_get_metrics_success(self, mock_async_client, mock_discover_url):
        """Test retrieving raw metrics from /metrics endpoint successfully."""
        mock_discover_url.return_value = "http://test-url:8000"

        mock_client_instance = mock_async_client.return_value.__aenter__.return_value
        mock_metrics_response = MagicMock()
        mock_metrics_response.text = "vllm_requests_running 1\nprometheus_metric_example 42"
        mock_metrics_response.status_code = 200
        mock_client_instance.get.return_value = mock_metrics_response

        result = await get_metrics()
        self.assertIn("vllm_requests_running 1", result)
        self.assertIn("prometheus_metric_example 42", result)

    @patch("server.discover_vllm_url", new_callable=AsyncMock)
    async def test_get_metrics_no_url(self, mock_discover_url):
        """Test get_metrics when no active vLLM service URL is discovered."""
        mock_discover_url.return_value = None
        result = await get_metrics()
        self.assertIn("is serving vLLM", result)

    @patch("server.secretmanager.SecretManagerServiceClient")
    @patch("server.get_secret", new_callable=AsyncMock)  # Mock get_secret to prevent actual calls
    async def test_save_hf_token(self, mock_get_secret, mock_secret_client):
        """Test saving HF token to Secret Manager."""
        mock_instance = mock_secret_client.return_value
        mock_instance.add_secret_version.return_value.name = "projects/test-project/secrets/hf-token/versions/1"

        # Mock get_secret to simulate secret existence check.
        # First call: simulate secret not found (raises exception)
        # Second call: simulate secret found (returns a dummy secret)
        mock_instance.get_secret.side_effect = [Exception("Secret not found"), MagicMock()]
        mock_instance.create_secret.return_value = None  # Mock create_secret if it doesn't exist

        # Test successful save (secret is created and version added)
        result = await save_hf_token("test-token")
        self.assertIn("✅ Token saved.", result)
        mock_instance.create_secret.assert_called_once()
        mock_instance.add_secret_version.assert_called_once()

    async def test_get_help(self):
        """Test that get_help returns formatted help text containing key configuration parameters."""
        fake_tools = [
            SimpleNamespace(name="zeta_tool", description="Does the zeta thing.\nIgnored second line."),
            SimpleNamespace(name="alpha_tool", description="Does the alpha thing."),
        ]
        with patch("server.mcp.list_tools", new_callable=AsyncMock) as mock_list_tools:
            mock_list_tools.return_value = fake_tools
            result = await get_help()

        self.assertIn("### 🛠️ TPU Gemma 4 SRE Agent Help & Configuration", result)
        self.assertIn("GOOGLE_CLOUD_PROJECT", result)
        self.assertIn("MODEL_NAME", result)
        self.assertIn("ACCELERATOR_TYPE", result)
        self.assertIn("Available MCP Tools", result)

        # The catalog is generated from the live registry, so registered tools must appear...
        self.assertIn("- **`alpha_tool`**: Does the alpha thing.", result)
        self.assertIn("- **`zeta_tool`**: Does the zeta thing.", result)
        # ...sorted by name, with only the docstring summary line.
        self.assertLess(result.index("alpha_tool"), result.index("zeta_tool"))
        self.assertNotIn("Ignored second line.", result)

    async def test_get_help_tolerates_missing_docstring(self):
        """A tool with no docstring should not break the generated catalog."""
        with patch("server.mcp.list_tools", new_callable=AsyncMock) as mock_list_tools:
            mock_list_tools.return_value = [SimpleNamespace(name="bare_tool", description=None)]
            result = await get_help()

        self.assertIn("- **`bare_tool`**: No description.", result)


class TestProvisioningModel(unittest.IsolatedAsyncioTestCase):
    """Covers the flex-start / spot / on-demand split on the Queued Resource path."""

    def test_flex_start_flags_bound_the_run(self):
        flags = _provisioning_flags("flex-start")
        self.assertIn("--provisioning-model=flex-start", flags)
        self.assertIn("--max-run-duration=4h", flags)
        self.assertNotIn("--spot", flags)

    def test_spot_flags(self):
        flags = _provisioning_flags("spot")
        self.assertIn("--spot", flags)
        # gcloud documents --max-run-duration as flex-start only; passing it here is rejected.
        self.assertNotIn("--max-run-duration=4h", flags)
        self.assertNotIn("--provisioning-model=flex-start", flags)

    def test_on_demand_flags_select_nothing(self):
        flags = _provisioning_flags("on-demand")
        self.assertEqual(flags, ["--labels=purpose=on-demand"])

    def test_spot_uses_the_preemptible_quota(self):
        """Spot capacity is metered separately, so a spot zone sweep must read another id."""
        self.assertEqual(_quota_id_for("spot"), TPU_SPOT_QUOTA_ID)
        self.assertEqual(_quota_id_for("flex-start"), TPU_QUOTA_ID)
        self.assertEqual(_quota_id_for("on-demand"), TPU_QUOTA_ID)
        self.assertNotEqual(TPU_QUOTA_ID, TPU_SPOT_QUOTA_ID)

    def test_status_rows_are_attributed_to_a_model(self):
        """Untagged rows predate the feature and all recorded flex-start attempts."""
        self.assertEqual(_status_model(" Timed out waiting 3 minutes. "), "flex-start")
        self.assertEqual(_status_model(" [spot] Creation failed. "), "spot")
        self.assertEqual(_status_model(" [on-demand] Creation failed. "), "on-demand")

    @patch("server.get_secret", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_create_passes_spot_flag_to_gcloud(self, mock_run_command, mock_get_secret):
        mock_get_secret.return_value = "dummy-hf-token"
        mock_run_command.return_value = (0, "", "")

        ok, msg = await _create_queued_resource("qr-test", "us-east5-b", "spot")

        self.assertTrue(ok, msg)
        cmd = mock_run_command.call_args[0][0]
        self.assertIn("--spot", cmd)
        self.assertNotIn("--max-run-duration=4h", cmd)
        self.assertIn("--valid-until-duration=4h", cmd)
        self.assertIn("--accelerator-type=v6e-1", cmd)
        # The preemption warning is the point of the message — don't let it get dropped.
        self.assertIn("Preemptible", msg)

    @patch("server.get_secret", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_create_defaults_to_flex_start(self, mock_run_command, mock_get_secret):
        mock_get_secret.return_value = "dummy-hf-token"
        mock_run_command.return_value = (0, "", "")

        ok, _ = await _create_queued_resource("qr-test", "us-east5-b")

        self.assertTrue(ok)
        cmd = mock_run_command.call_args[0][0]
        self.assertIn("--provisioning-model=flex-start", cmd)
        self.assertIn("--max-run-duration=4h", cmd)
        self.assertNotIn("--spot", cmd)

    @patch("server.get_secret", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_create_rejects_unknown_model_before_calling_gcloud(self, mock_run_command, mock_get_secret):
        ok, msg = await _create_queued_resource("qr-test", "us-east5-b", "preemptible")

        self.assertFalse(ok)
        self.assertIn("unknown provisioning_model", msg)
        mock_run_command.assert_not_called()
        mock_get_secret.assert_not_called()

    async def test_cost_rejects_unknown_model(self):
        result = await estimate_deployment_cost(provisioning_model="preemptible")
        self.assertIn("Unknown provisioning_model", result)


def _sku(description, usage_type, price, region="us-east5"):
    """A Billing Catalog SKU, trimmed to the fields the lookup reads."""
    units, nanos = int(price), round((price - int(price)) * 1e9)
    return {
        "description": description,
        "serviceRegions": [region],
        "category": {"usageType": usage_type},
        "pricingInfo": [
            {"pricingExpression": {"usageUnit": "h", "tieredRates": [{"unitPrice": {"units": units, "nanos": nanos}}]}}
        ],
    }


# Shapes and rates copied from the live catalog for us-east5 on 2026-08-07. The decoys are
# the real reason this matcher needs testing: five other SKUs describe the same chip in the
# same region, and three of them share the OnDemand usage type. "Capacity Optimized TpuV6e"
# is the one only the ^ anchor rejects — it is OnDemand, in region, and quotes a different
# rate from the plain SKU.
FAKE_SKUS = [
    _sku("TpuV6e running in Columbus", "OnDemand", 2.70),
    _sku("DWS Defined Duration V6e running in Columbus", "OnDemand", 1.35),
    _sku("TpuV6e attached to Spot Preemptible VMs running in Columbus", "Preemptible", 1.4033),
    _sku("Capacity Optimized TpuV6e running in Columbus", "OnDemand", 2.70),
    _sku("Reserved V6e TPU in Columbus in Calendar Mode", "OnDemand", 1.89),
    _sku("Commitment v1: TpuV6e running in Columbus for 1 Year", "Commit1Yr", 1.89),
    _sku("TpuV6e running in Tokyo", "OnDemand", 3.24, region="asia-northeast1"),
]


class TestCostFromLivePricing(unittest.IsolatedAsyncioTestCase):
    """estimate_deployment_cost reads the Billing Catalog; it must never invent a rate."""

    def setUp(self):
        patcher = patch("server._fetch_compute_skus", new_callable=AsyncMock)
        self.mock_fetch = patcher.start()
        self.mock_fetch.return_value = (FAKE_SKUS, "")
        self.addCleanup(patcher.stop)

    async def test_each_model_selects_its_own_sku(self):
        expected = {"on-demand": 2.70, "flex-start": 1.35, "spot": 1.4033}
        for model, rate in expected.items():
            rate_found, unit, desc = await _lookup_tpu_rate("v6e", model, "us-east5")
            assert rate_found is not None, f"no SKU matched for {model}"
            self.assertAlmostEqual(rate_found, rate, places=4, msg=f"{model} -> {desc}")
            self.assertEqual(unit, "h")

    async def test_reserved_and_committed_skus_are_not_mistaken_for_on_demand(self):
        """All four share a region and three share the OnDemand usage type."""
        _, _, desc = await _lookup_tpu_rate("v6e", "on-demand", "us-east5")
        self.assertEqual(desc, "TpuV6e running in Columbus")

    async def test_region_is_respected(self):
        rate, _, _ = await _lookup_tpu_rate("v6e", "on-demand", "asia-northeast1")
        assert rate is not None
        self.assertAlmostEqual(rate, 3.24, places=4)

    async def test_missing_sku_reports_instead_of_guessing(self):
        rate, _, detail = await _lookup_tpu_rate("v6e", "spot", "europe-west4")
        self.assertIsNone(rate)
        result = await estimate_deployment_cost(tpu_type="v6e", provisioning_model="spot", region="europe-west4")
        self.assertIn("No published rate found", result)
        self.assertIn("europe-west4", detail or "")
        # A fabricated dollar figure is the exact failure this replaced — there must be none.
        self.assertNotIn("Estimated Cost", result)

    async def test_unreachable_catalog_does_not_produce_a_number(self):
        self.mock_fetch.return_value = (None, "could not get an access token")
        result = await estimate_deployment_cost(provisioning_model="spot")
        self.assertIn("could not get an access token", result)
        self.assertNotIn("Estimated Cost", result)

    async def test_cost_multiplies_rate_by_chips_and_hours(self):
        result = await estimate_deployment_cost(
            hours=2.0, tpu_type="v6e", topology="2x4", provisioning_model="on-demand", region="us-east5"
        )
        self.assertIn("$43.20", result)  # 2.70 * 8 chips * 2h
        self.assertIn("TpuV6e running in Columbus", result)

    async def test_ranking_matches_published_rates(self):
        """On v6e in us-east5 flex-start is the cheapest — spot lands just above it.

        That is the reverse of the v5e ordering this rig was forked from, which is the
        point: the ranking is a fact about the catalog for one chip in one region, never
        a rule about the provisioning models. Assert what is published, not what is
        assumed.
        """
        rates = {}
        for model in PROVISIONING_MODELS:
            rate, _, _ = await _lookup_tpu_rate("v6e", model, "us-east5")
            assert rate is not None, f"no SKU matched for {model}"
            rates[model] = rate
        self.assertLess(rates["flex-start"], rates["spot"])
        self.assertLess(rates["spot"], rates["on-demand"])

    def test_topology_parsing(self):
        self.assertEqual(_parse_topology("1x1"), 1)
        self.assertEqual(_parse_topology("2x4"), 8)
        self.assertEqual(_parse_topology("2x2x4"), 16)
        self.assertIsNone(_parse_topology("v6e-1"))

    async def test_bad_topology_is_rejected_not_defaulted(self):
        """It used to eval() this and silently fall back to 8 chips on failure."""
        result = await estimate_deployment_cost(topology="__import__('os')")
        self.assertIn("Could not read a chip count", result)


def _node(name, ip="10.0.0.1", state="READY", external=True):
    """A tpu-vm list entry as gcloud prints it."""
    endpoint = {"ipAddress": ip}
    if external:
        endpoint["accessConfig"] = {"externalIp": ip}
    return {
        "name": f"projects/p/locations/us-east5-b/nodes/{name}",
        "state": state,
        "acceleratorType": "v6e-1",
        "networkEndpoints": [endpoint],
    }


class TestNodeDiscovery(unittest.IsolatedAsyncioTestCase):
    """Discovery has to see a node whether or not a Queued Resource created it."""

    def _router(self, nodes, qrs=None):
        """Routes the two gcloud list calls discovery makes; describe always misses."""

        async def _run(cmd, timeout=60):
            if "queued-resources" in cmd:
                if "describe" in cmd:
                    return 1, "", "NOT_FOUND"
                return 0, json.dumps(qrs or []), ""
            if "tpu-vm" in cmd and "list" in cmd:
                return 0, json.dumps(nodes), ""
            return 1, "", f"unexpected command: {cmd}"

        return _run

    async def test_finds_standalone_spot_vm_with_no_queued_resource(self):
        """The regression: a hand-provisioned spot VM used to read as 'no TPU at all'."""
        with (
            patch("server.run_command", new=AsyncMock(side_effect=self._router([_node("tpu-2B-v6e1-devops-agent")]))),
            patch("server._probe_vllm", new=AsyncMock(return_value="google/gemma-4-E2B-it")),
        ):
            node = await _discover_vllm_node()
        assert node is not None
        self.assertEqual(node.name, "tpu-2B-v6e1-devops-agent")
        self.assertEqual(node.url, "http://10.0.0.1:8000")
        self.assertTrue(node.serving)

    async def test_this_rigs_node_wins_over_a_siblings(self):
        """Rigs share the zone, so our own name is probed first."""
        nodes = [_node("some-other-rig-node", ip="10.0.0.9"), _node(f"{RESOURCE_ID}-node", ip="10.0.0.2")]
        with (
            patch("server.run_command", new=AsyncMock(side_effect=self._router(nodes))),
            patch("server._probe_vllm", new=AsyncMock(return_value="google/gemma-4-E2B-it")) as probe,
        ):
            node = await _discover_vllm_node()
        assert node is not None
        self.assertEqual(node.name, f"{RESOURCE_ID}-node")
        probe.assert_awaited_once_with("http://10.0.0.2:8000")

    async def test_our_booting_node_is_returned_unprobed_but_a_siblings_is_not(self):
        """A node of ours is pollable while vLLM boots; someone else's is not ours to use."""
        nodes = [_node(f"{RESOURCE_ID}-node", ip="10.0.0.2")]
        with (
            patch("server.run_command", new=AsyncMock(side_effect=self._router(nodes))),
            patch("server._probe_vllm", new=AsyncMock(return_value=None)),
        ):
            mine = await _discover_vllm_node()
        assert mine is not None
        self.assertEqual(mine.url, "http://10.0.0.2:8000")
        self.assertFalse(mine.serving)

        with (
            patch(
                "server.run_command",
                new=AsyncMock(side_effect=self._router([_node("some-other-rig-node", ip="10.0.0.9")])),
            ),
            patch("server._probe_vllm", new=AsyncMock(return_value=None)),
        ):
            theirs = await _discover_vllm_node()
        self.assertIsNone(theirs)

    async def test_node_with_no_ip_is_skipped(self):
        nodes = [_node("pending-node", state="CREATING")]
        nodes[0]["networkEndpoints"] = [{}]
        with (
            patch("server.run_command", new=AsyncMock(side_effect=self._router(nodes))),
            patch("server._probe_vllm", new=AsyncMock(return_value="m")),
        ):
            self.assertIsNone(await _discover_vllm_node())

    async def test_resolve_node_id_falls_back_to_a_standalone_vm(self):
        """No queued resource by that id, but a TPU VM with the name is right there."""
        with patch("server.run_command", new=AsyncMock(side_effect=self._router([_node("legacy-vm")]))):
            self.assertEqual(await _resolve_node_id("legacy-vm"), "legacy-vm")

        # ...and by the <id>-node form the queued-resource path would have produced.
        with patch("server.run_command", new=AsyncMock(side_effect=self._router([_node("legacy-node")]))):
            self.assertEqual(await _resolve_node_id("legacy"), "legacy-node")

    async def test_resolve_node_id_prefers_the_queued_resource_node(self):
        async def _run(cmd, timeout=60):
            if "queued-resources" in cmd and "describe" in cmd:
                return 0, f"{RESOURCE_ID}-node", ""
            return 1, "", "should not be reached"

        with patch("server.run_command", new=AsyncMock(side_effect=_run)):
            self.assertEqual(await _resolve_node_id(RESOURCE_ID), f"{RESOURCE_ID}-node")

    async def test_resolve_node_id_last_resort_is_the_serving_node(self):
        """Default resource_id still reaches a running deployment named by an old convention."""
        with (
            patch("server.run_command", new=AsyncMock(side_effect=self._router([_node("tpu-2B-v6e1-devops-agent")]))),
            patch("server._probe_vllm", new=AsyncMock(return_value="google/gemma-4-E2B-it")),
        ):
            self.assertEqual(await _resolve_node_id(RESOURCE_ID), "tpu-2B-v6e1-devops-agent")

    async def test_resolve_node_id_gives_up_when_nothing_serves(self):
        with (
            patch("server.run_command", new=AsyncMock(side_effect=self._router([_node("some-other-rig-node")]))),
            patch("server._probe_vllm", new=AsyncMock(return_value=None)),
        ):
            self.assertIsNone(await _resolve_node_id(RESOURCE_ID))


if __name__ == "__main__":
    unittest.main()
