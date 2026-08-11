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
import server  # noqa: E402
from server import (  # noqa: E402
    INSTANCE_NAME,
    MACHINE_TYPE,
    MODEL_NAME,
    PROVISIONING_MODELS,
    TPU_QUOTA_ID,
    TPU_SPOT_QUOTA_ID,
    _create_tpu_instance,
    _discover_vllm_node,
    _lookup_tpu_rate,
    _parse_topology,
    _provisioning_flags,
    _quota_id_for,
    _resolve_node_id,
    _status_model,
    destroy_tpu_instance,
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
        self.assertIn("gcloud compute instances create test-vllm", config)
        self.assertIn(f"--machine-type={MACHINE_TYPE}", config)
        self.assertIn("--image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e", config)
        self.assertIn("--maintenance-policy=TERMINATE", config)
        # The TPU API's vocabulary must not leak into a Compute Engine command line.
        self.assertNotIn("--accelerator-type=", config)
        self.assertNotIn("--version=v2-alpha-tpuv6e", config)
        self.assertNotIn("tpus tpu-vm", config)

        self.assertIn("vllm/vllm-tpu:nightly", config)
        self.assertIn("google/gemma-4-E2B-it", config)

        # The image ships no Docker, so a startup script that goes straight to `docker run`
        # fails on its first command while the instance still reports RUNNING.
        self.assertIn("docker.io", config)
        self.assertLess(config.find("docker.io"), config.find("docker run"))

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
    """Covers the provisioning-model split on the Compute Engine path."""

    def test_every_model_maps_to_a_screaming_case_gcloud_value(self):
        """`instances create` validates this enum client-side; the TPU API's spelling fails."""
        expected = {
            "flex-start": "FLEX_START",
            "spot": "SPOT",
            "on-demand": "STANDARD",
            "reservation-bound": "RESERVATION_BOUND",
        }
        for model, gcloud_value in expected.items():
            flags = _provisioning_flags(model)
            self.assertIn(f"--provisioning-model={gcloud_value}", flags, model)
            self.assertNotIn(f"--provisioning-model={model}", flags, model)

    def test_flex_start_carries_the_wait_knob(self):
        flags = _provisioning_flags("flex-start")
        self.assertIn("--request-valid-for-duration=2h", flags)
        self.assertIn("--max-run-duration=4h", flags)
        # --valid-until-duration is the TPU API's spelling and does not exist here.
        self.assertFalse(any(f.startswith("--valid-until-duration") for f in flags))

    def test_run_bound_is_not_flex_start_only_here(self):
        """The TPU API restricts --max-run-duration to flex-start. Compute Engine does not."""
        for model in ("spot", "on-demand"):
            flags = _provisioning_flags(model)
            self.assertIn("--max-run-duration=4h", flags, model)
            self.assertIn("--instance-termination-action=DELETE", flags, model)

    def test_reservation_bound_has_no_run_bound(self):
        """It runs for its reservation's duration — a stop flag would contradict that."""
        flags = _provisioning_flags("reservation-bound")
        self.assertFalse(any(f.startswith("--max-run-duration") for f in flags))

    def test_reservation_bound_targets_a_specific_reservation(self):
        """--reservation-affinity defaults to `any`; RESERVATION_BOUND requires `specific`."""
        flags = _provisioning_flags("reservation-bound", reservation_name="cal-v6e-1")
        self.assertIn("--reservation-affinity=specific", flags)
        self.assertIn("--reservation=cal-v6e-1", flags)

    def test_reservation_name_is_ignored_by_the_other_models(self):
        """A reservation is only consumable through RESERVATION_BOUND."""
        for model in ("flex-start", "spot", "on-demand"):
            flags = _provisioning_flags(model, reservation_name="cal-v6e-1")
            self.assertFalse(any(f.startswith("--reservation") for f in flags), model)

    def test_no_reservation_name_emits_no_half_formed_flag(self):
        """An empty --reservation= is a worse failure than an incomplete model gcloud rejects."""
        flags = _provisioning_flags("reservation-bound")
        self.assertNotIn("--reservation=", flags)
        self.assertFalse(any(f.startswith("--reservation-affinity") for f in flags))

    async def test_reservation_bound_without_a_name_aborts_before_gcloud(self):
        """The server-side rejection lands after the startup script is rendered and uploaded."""
        with patch("server.run_command", new=AsyncMock()) as run:
            ok, msg = await server._create_tpu_instance("i", "europe-west4-a", "reservation-bound")
        self.assertFalse(ok)
        self.assertIn("reservation", msg.lower())
        run.assert_not_awaited()

    def test_every_model_labels_its_owning_rig(self):
        for model in PROVISIONING_MODELS:
            flags = _provisioning_flags(model)
            self.assertTrue(any(f.startswith("--labels=rig=") for f in flags), model)

    def test_durations_are_overridable(self):
        flags = _provisioning_flags("flex-start", max_run_duration="1h", request_valid_for="30m")
        self.assertIn("--max-run-duration=1h", flags)
        self.assertIn("--request-valid-for-duration=30m", flags)

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

    @patch("server.run_command", new_callable=AsyncMock)
    async def test_destroy_needs_no_force_flag(self, mock_run_command):
        """An instance is one object, so teardown does not need the QR path's --force.

        A Queued Resource delete required --force because an ACTIVE resource owns a node the
        API refuses to drop out from under it. Nothing here owns anything else.
        """
        mock_run_command.return_value = (0, "deleted", "")

        await destroy_tpu_instance("inst-test", zone="europe-west4-a")

        cmd = mock_run_command.call_args[0][0]
        self.assertIn("instances", cmd)
        self.assertIn("delete", cmd)
        self.assertIn("--zone=europe-west4-a", cmd)
        self.assertNotIn("--force", cmd)

    @patch("server.get_secret", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_create_uses_screaming_case_provisioning_model(self, mock_run_command, mock_get_secret):
        """Regression guard: `instances create` rejects the TPU API's lowercase spelling."""
        mock_get_secret.return_value = "dummy-hf-token"
        mock_run_command.return_value = (0, "", "")

        ok, msg = await _create_tpu_instance("inst-test", "us-east5-b", "spot")

        self.assertTrue(ok, msg)
        cmd = mock_run_command.call_args[0][0]
        self.assertIn("--provisioning-model=SPOT", cmd)
        self.assertNotIn("--provisioning-model=spot", cmd)
        # The machine type is the accelerator request; ACCELERATOR_TYPE must never be passed.
        self.assertIn("--machine-type=ct6e-standard-1t", cmd)
        self.assertNotIn("--accelerator-type=v6e-1", cmd)
        self.assertNotIn("--runtime-version=v2-alpha-tpuv6e", cmd)
        # The preemption warning is the point of the message — don't let it get dropped.
        self.assertIn("Preemptible", msg)

    @patch("server.get_secret", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_spot_and_on_demand_also_get_a_run_bound(self, mock_run_command, mock_get_secret):
        """Unlike the TPU API, --max-run-duration is not flex-start-only on Compute Engine."""
        mock_get_secret.return_value = "dummy-hf-token"
        mock_run_command.return_value = (0, "", "")

        for model in ("spot", "on-demand"):
            await _create_tpu_instance("inst-test", "us-east5-b", model)
            cmd = mock_run_command.call_args[0][0]
            self.assertIn("--max-run-duration=4h", cmd, model)
            self.assertIn("--instance-termination-action=DELETE", cmd, model)

    @patch("server.get_secret", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_create_defaults_to_flex_start(self, mock_run_command, mock_get_secret):
        mock_get_secret.return_value = "dummy-hf-token"
        mock_run_command.return_value = (0, "", "")

        ok, _ = await _create_tpu_instance("inst-test", "us-east5-b")

        self.assertTrue(ok)
        cmd = mock_run_command.call_args[0][0]
        self.assertIn("--provisioning-model=FLEX_START", cmd)
        # The flex-start wait knob, replacing the TPU API's --valid-until-duration.
        self.assertIn("--request-valid-for-duration=2h", cmd)
        self.assertIn("--max-run-duration=4h", cmd)
        # Both are load-bearing: no scope means the boot-time secret fetch fails, and a TPU
        # instance cannot live-migrate.
        self.assertIn("--scopes=cloud-platform", cmd)
        self.assertIn("--maintenance-policy=TERMINATE", cmd)

    @patch("server.get_secret", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_create_labels_the_instance_with_its_rig(self, mock_run_command, mock_get_secret):
        """Teardown tells ours from a sibling's by label — an instance name does not encode its owner."""
        mock_get_secret.return_value = "dummy-hf-token"
        mock_run_command.return_value = (0, "", "")

        await _create_tpu_instance("inst-test", "us-east5-b")

        cmd = mock_run_command.call_args[0][0]
        self.assertTrue(any(c.startswith("--labels=rig=") for c in cmd), cmd)

    @patch("server.get_secret", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_create_rejects_unknown_model_before_calling_gcloud(self, mock_run_command, mock_get_secret):
        ok, msg = await _create_tpu_instance("inst-test", "us-east5-b", "preemptible")

        self.assertFalse(ok)
        self.assertIn("unknown provisioning_model", msg)
        mock_run_command.assert_not_called()
        mock_get_secret.assert_not_called()

    @patch("server.get_secret", new_callable=AsyncMock)
    @patch("server.run_command", new_callable=AsyncMock)
    async def test_create_timeout_is_not_reported_as_a_failure(self, mock_run_command, mock_get_secret):
        """A flex-start create that times out client-side may still produce a billing VM."""
        mock_get_secret.return_value = "dummy-hf-token"
        mock_run_command.return_value = (1, "", "Timeout after 590s")

        ok, msg = await _create_tpu_instance("inst-test", "us-east5-b")

        self.assertFalse(ok)
        self.assertIn("⏳", msg)
        self.assertNotIn("❌", msg)

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
        for model in ("flex-start", "spot", "on-demand"):
            rate, _, _ = await _lookup_tpu_rate("v6e", model, "us-east5")
            assert rate is not None, f"no SKU matched for {model}"
            rates[model] = rate
        self.assertLess(rates["flex-start"], rates["spot"])
        self.assertLess(rates["spot"], rates["on-demand"])

    async def test_reservation_bound_reports_no_rate_rather_than_guessing(self):
        """It is in PROVISIONING_MODELS but has no catalog SKU — must not fall through to on-demand."""
        rate, _, note = await _lookup_tpu_rate("v6e", "reservation-bound", "us-east5")
        self.assertIsNone(rate)
        self.assertIn("reservation", (note or "").lower())

    def test_topology_parsing(self):
        self.assertEqual(_parse_topology("1x1"), 1)
        self.assertEqual(_parse_topology("2x4"), 8)
        self.assertEqual(_parse_topology("2x2x4"), 16)
        self.assertIsNone(_parse_topology("v6e-1"))

    async def test_bad_topology_is_rejected_not_defaulted(self):
        """It used to eval() this and silently fall back to 8 chips on failure."""
        result = await estimate_deployment_cost(topology="__import__('os')")
        self.assertIn("Could not read a chip count", result)


def _node(name, ip="10.0.0.1", status="RUNNING", external=True, rig=None):
    """A `compute instances list` entry as gcloud prints it.

    Deliberately a different shape from the sibling rig's fixture, and the differences are
    the ones that bite: `status`/`RUNNING` rather than `state`/`READY`, and
    `networkInterfaces[].accessConfigs[].natIP` rather than
    `networkEndpoints[].accessConfig.externalIp`. Copying that fixture over would make
    discovery look tested while silently matching nothing.
    """
    iface = {"networkIP": ip}
    if external:
        iface["accessConfigs"] = [{"natIP": ip}]
    return {
        "name": f"projects/p/zones/us-east5-b/instances/{name}",
        "status": status,
        "machineType": "projects/p/zones/us-east5-b/machineTypes/ct6e-standard-1t",
        "labels": {"rig": rig} if rig else {},
        "networkInterfaces": [iface],
    }


class TestNodeDiscovery(unittest.IsolatedAsyncioTestCase):
    """Discovery lists Compute Engine instances, not TPU VM nodes."""

    def _router(self, nodes, qrs=None):
        """Routes the gcloud list calls discovery makes; describe always misses."""

        async def _run(cmd, timeout=60):
            if "queued-resources" in cmd:
                if "describe" in cmd:
                    return 1, "", "NOT_FOUND"
                return 0, json.dumps(qrs or []), ""
            if "instances" in cmd and "list" in cmd:
                return 0, json.dumps(nodes), ""
            if "instances" in cmd and "describe" in cmd:
                return 1, "", "NOT_FOUND"
            return 1, "", f"unexpected command: {cmd}"

        return _run

    async def test_a_ct6e_instance_is_not_a_tpu_vm_node(self):
        """The structural difference: `tpus tpu-vm list` cannot see what this rig creates.

        If discovery ever regresses to the sibling's helper it will find nothing, so this
        pins that the instances API is the one being asked.
        """
        seen = []

        async def _run(cmd, timeout=60):
            seen.append(cmd)
            if "queued-resources" in cmd:
                return 0, "[]", ""
            if "instances" in cmd and "list" in cmd:
                return 0, json.dumps([_node("gce-vllm-v6e1-2b")]), ""
            return 1, "", "unexpected"

        with (
            patch("server.run_command", new=AsyncMock(side_effect=_run)),
            patch("server._probe_vllm", new=AsyncMock(return_value="google/gemma-4-E2B-it")),
        ):
            await _discover_vllm_node()

        self.assertFalse(any("tpu-vm" in c for c in seen), f"discovery must not use tpu-vm: {seen}")

    async def test_flex_start_is_metered_by_the_preemptible_quota(self):
        """flex-start spends the PREEMPTIBLE pool, not the family quota.

        This is documented, not inferred — the Compute Engine provisioning-models page says
        "When you create a Flex-start VM, preemptible quota is consumed." It is easy to get
        backwards because flex-start is not preemptible in behaviour, and this file did have
        it backwards: flex-start was grouped with on-demand against the family quota. Since
        this rig DEFAULTS to flex-start, that sent the default path at the wrong metric.

        Only on-demand draws on the family quota.
        """
        seen = []

        async def _fake(service, quota_id):
            seen.append((service, quota_id))
            return []

        for model, expected in [
            ("flex-start", server.GCE_SPOT_QUOTA_ID),
            ("spot", server.GCE_SPOT_QUOTA_ID),
            ("on-demand", server.GCE_QUOTA_ID),
        ]:
            seen.clear()
            with patch("server._get_zones_with_available_quota_list", new=AsyncMock(side_effect=_fake)):
                await server.get_zones_with_available_quota(service="compute.googleapis.com", provisioning_model=model)
            self.assertEqual(seen[-1][1], expected, f"{model} must meter against {expected}")

    async def test_the_preemptible_quota_id_is_region_scoped(self):
        """The per-zone spelling exists but carries no entries in this project.

        `PREEMPTIBLE-TPU-V6E-per-project-zone` returns a bare default of -1 and no per-zone
        rows, while the per-region id holds the real values. Defaulting to the per-zone one
        made every quota lookup return nothing.
        """
        self.assertTrue(
            server.GCE_SPOT_QUOTA_ID.endswith("per-project-region"),
            f"expected a region-scoped preemptible id, got {server.GCE_SPOT_QUOTA_ID}",
        )

    async def test_no_remote_tool_reaches_the_vm_through_tpu_vm_ssh(self):
        """Discovery was pinned off the TPU API; the SSH tools were not, and had regressed.

        `manage_vllm_docker`, `run_vllm_benchmark`, `get_vllm_docker_logs` and
        `get_tpu_system_logs` all shelled to `gcloud compute tpus tpu-vm ssh` after the fork.
        That cannot reach a ct6e instance — it is not a TPU VM node — so all four failed with
        a not-found against a VM that was plainly RUNNING. The old
        `test_a_ct6e_instance_is_not_a_tpu_vm_node` did not catch it because it only
        inspected the discovery path.
        """
        seen: list[list[str]] = []

        async def _run(cmd, timeout=60):
            seen.append(cmd)
            return 0, "", ""

        with (
            patch("server.run_command", new=AsyncMock(side_effect=_run)),
            patch("server._resolve_node_id", new=AsyncMock(return_value="gce-vllm-v6e1-2b")),
        ):
            await server.manage_vllm_docker("gce-vllm-v6e1-2b", action="status")
            await server.get_vllm_docker_logs("gce-vllm-v6e1-2b")
            await server.get_tpu_system_logs("gce-vllm-v6e1-2b")

        self.assertTrue(seen, "no commands were issued")
        for cmd in seen:
            self.assertNotIn("tpu-vm", cmd, f"must not SSH via the TPU API: {cmd}")
            self.assertEqual(cmd[:3], ["gcloud", "compute", "ssh"], f"expected `compute ssh`: {cmd}")

    async def test_docker_commands_install_docker_when_the_image_lacks_it(self):
        """The CE image ships no Docker, so a recovery tool that assumes it dies as the boot did.

        `manage_vllm_docker start` is what you reach for when the startup script failed — and
        the way it failed, on the first instance this rig created, was `docker: command not
        found`. Without the prelude the recovery reproduces the original failure exactly.

        `get_tpu_system_logs` is deliberately excluded: journalctl is not a Docker command.
        """
        seen: list[list[str]] = []

        async def _run(cmd, timeout=60):
            seen.append(cmd)
            return 0, "", ""

        with (
            patch("server.run_command", new=AsyncMock(side_effect=_run)),
            patch("server._resolve_node_id", new=AsyncMock(return_value="gce-vllm-v6e1-2b")),
        ):
            await server.manage_vllm_docker("gce-vllm-v6e1-2b", action="start")
            docker_cmd = seen[-1][-1]
            seen.clear()
            await server.get_tpu_system_logs("gce-vllm-v6e1-2b")
            journal_cmd = seen[-1][-1]

        self.assertIn("apt-get install -y -qq docker.io", docker_cmd)
        self.assertTrue(docker_cmd.startswith("if ! command -v docker"), docker_cmd)
        self.assertNotIn("docker.io", journal_cmd, "journalctl must not carry the Docker prelude")

    async def test_the_startup_script_installs_docker_before_pulling(self):
        """The boot-time fix for the same fact, at its own layer.

        The pull loop's five retries are not a recovery path when the binary is absent — they
        just spend 100 seconds discovering the same thing five times, then fail the script
        while the instance goes on reporting RUNNING.
        """
        script = await server._get_formatted_startup_script("google/gemma-4-E2B-it", zone="europe-west4-a")

        install_at = script.find("apt-get install -y -qq docker.io")
        pull_at = script.find("docker pull vllm/vllm-tpu:nightly")
        self.assertNotEqual(install_at, -1, "startup script must install Docker")
        self.assertNotEqual(pull_at, -1, "startup script must still pull the image")
        self.assertLess(install_at, pull_at, "Docker must be installed before the pull is attempted")

        # The template is rendered with str.format(); a stray brace in the added block would
        # raise at render time and break every deploy. Checking for surviving *placeholders*
        # rather than for braces at all — the rendered script legitimately contains the JSON
        # value {"image":4,"audio":1}, which is output, not an unfilled slot.
        import re as _re

        self.assertEqual([], _re.findall(r"\{[a-z_][a-z0-9_]*\}", script))

    async def test_finds_a_running_instance(self):
        with (
            patch("server.run_command", new=AsyncMock(side_effect=self._router([_node("hand-made-vm")]))),
            patch("server._probe_vllm", new=AsyncMock(return_value="google/gemma-4-E2B-it")),
        ):
            node = await _discover_vllm_node()
        assert node is not None
        self.assertEqual(node.name, "hand-made-vm")
        self.assertEqual(node.url, "http://10.0.0.1:8000")
        self.assertTrue(node.serving)

    async def test_this_rigs_instance_wins_over_a_siblings(self):
        """Rigs share the zone, so our own name is probed first."""
        nodes = [_node("some-other-rig", ip="10.0.0.9"), _node(INSTANCE_NAME, ip="10.0.0.2")]
        with (
            patch("server.run_command", new=AsyncMock(side_effect=self._router(nodes))),
            patch("server._probe_vllm", new=AsyncMock(return_value="google/gemma-4-E2B-it")) as probe,
        ):
            node = await _discover_vllm_node()
        assert node is not None
        self.assertEqual(node.name, INSTANCE_NAME)
        probe.assert_awaited_once_with("http://10.0.0.2:8000")

    async def test_our_booting_instance_is_returned_unprobed_but_a_siblings_is_not(self):
        """One of ours is pollable while vLLM boots; someone else's is not ours to use."""
        with (
            patch("server.run_command", new=AsyncMock(side_effect=self._router([_node(INSTANCE_NAME, ip="10.0.0.2")]))),
            patch("server._probe_vllm", new=AsyncMock(return_value=None)),
        ):
            mine = await _discover_vllm_node()
        assert mine is not None
        self.assertEqual(mine.url, "http://10.0.0.2:8000")
        self.assertFalse(mine.serving)

        with (
            patch(
                "server.run_command",
                new=AsyncMock(side_effect=self._router([_node("some-other-rig", ip="10.0.0.9")])),
            ),
            patch("server._probe_vllm", new=AsyncMock(return_value=None)),
        ):
            theirs = await _discover_vllm_node()
        self.assertIsNone(theirs)

    async def test_instance_with_no_ip_is_skipped(self):
        nodes = [_node("pending", status="PROVISIONING", external=False)]
        nodes[0]["networkInterfaces"] = [{}]
        with (
            patch("server.run_command", new=AsyncMock(side_effect=self._router(nodes))),
            patch("server._probe_vllm", new=AsyncMock(return_value="m")),
        ):
            self.assertIsNone(await _discover_vllm_node())

    async def test_resolve_node_id_returns_the_instance_of_that_exact_name(self):
        """No derived -node suffix on this path: the id you ask for is the name you get."""

        async def _run(cmd, timeout=60):
            if "instances" in cmd and "describe" in cmd:
                return 0, "legacy-vm", ""
            return 1, "", "should not be reached"

        with patch("server.run_command", new=AsyncMock(side_effect=_run)):
            self.assertEqual(await _resolve_node_id("legacy-vm"), "legacy-vm")

    async def test_resolve_node_id_last_resort_is_the_serving_instance(self):
        """Default name still reaches a running deployment named by hand."""
        with (
            patch("server.run_command", new=AsyncMock(side_effect=self._router([_node("hand-made-vm")]))),
            patch("server._probe_vllm", new=AsyncMock(return_value="google/gemma-4-E2B-it")),
        ):
            self.assertEqual(await _resolve_node_id(INSTANCE_NAME), "hand-made-vm")

    async def test_resolve_node_id_gives_up_when_nothing_serves(self):
        with (
            patch("server.run_command", new=AsyncMock(side_effect=self._router([_node("some-other-rig")]))),
            patch("server._probe_vllm", new=AsyncMock(return_value=None)),
        ):
            self.assertIsNone(await _resolve_node_id(INSTANCE_NAME))


if __name__ == "__main__":
    unittest.main()
