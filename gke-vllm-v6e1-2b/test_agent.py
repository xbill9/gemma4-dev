import asyncio
import base64
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import yaml

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
    MODEL_NAME,
    PROVISIONING_MODELS,
    _discover_vllm_node,
    _lookup_tpu_rate,
    _parse_topology,
    _vllm_serve_flags,
    estimate_deployment_cost,
    get_help,
    get_metrics,
    get_model_details,
    query_queued_gemma4_with_stats,
    save_hf_token,
    verify_model_health,
)


class TestDevOpsAgent(unittest.IsolatedAsyncioTestCase):
    def test_model_name_default(self):
        """Verify the default model is Gemma 4."""
        self.assertEqual(MODEL_NAME, "google/gemma-4-E2B-it")

    async def test_get_vllm_deployment_config_emits_a_manifest(self):
        """The config tool prints Kubernetes YAML, not a `docker run` one-liner.

        The twin rig emits a shell command to paste on a VM. There is no VM to paste it on
        here, and the vocabulary of the twin's version — machine type, image family,
        maintenance policy — has no place in a manifest.
        """
        config = await server.get_vllm_deployment_config(model_name=MODEL_NAME)
        self.assertIn("kind: Deployment", config)
        self.assertIn("kind: Service", config)
        self.assertIn("google.com/tpu", config)
        self.assertIn(MODEL_NAME, config)
        self.assertIn("vllm/vllm-tpu:nightly", config)
        for vm_ism in ("gcloud compute instances create", "--image-family=", "--maintenance-policy=", "tpus tpu-vm"):
            self.assertNotIn(vm_ism, config, f"{vm_ism} is Compute Engine vocabulary")

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
        self.assertIn("No vLLM Service is answering", result)

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


class TestNodePoolProvisioning(unittest.IsolatedAsyncioTestCase):
    """The node-pool flag mapping — third vocabulary for the same four ideas."""

    def test_single_host_pool_sends_no_tpu_topology(self):
        """Verified against the API on 2026-08-25, and it cost this rig its first create.

            400: TPU topology can't be specified with single-host TPU slice pool

        ct6e-standard-1t at one node IS the slice. --tpu-topology is a MULTI-host flag, and
        1x1 is not a small version of it. GKE still labels the node
        cloud.google.com/gke-tpu-topology=1x1, so the value is real as a selector and
        refused as a create flag — which is why this needs a test rather than a comment.
        """
        with patch.object(server, "GKE_TPU_TOPOLOGY", ""):
            self.assertEqual(server._topology_flags(), [])

    def test_multi_host_pool_sends_topology_and_compact_placement(self):
        with patch.object(server, "GKE_TPU_TOPOLOGY", "4x4"):
            flags = server._topology_flags()
        self.assertIn("--tpu-topology=4x4", flags)
        self.assertIn("--placement-type=COMPACT", flags)

    def test_every_model_maps_to_node_pool_flags(self):
        """No --provisioning-model flag anywhere: a node pool spells all four differently.

        `queued-resources create` takes --provisioning-model=flex-start, `instances create`
        takes FLEX_START, and a node pool takes a bare --flex-start. Passing either of the
        other two spellings here is a hard error, so the mapping lives in exactly one place.
        """
        for model in PROVISIONING_MODELS:
            flags = server._node_pool_provisioning_flags(model, num_nodes="1")
            self.assertFalse(
                any(f.startswith("--provisioning-model") for f in flags),
                f"{model} must not emit --provisioning-model on the GKE path: {flags}",
            )
        self.assertIn("--spot", server._node_pool_provisioning_flags("spot", "1"))
        self.assertIn("--flex-start", server._node_pool_provisioning_flags("flex-start", "1"))
        self.assertNotIn("--spot", server._node_pool_provisioning_flags("on-demand", "1"))

    def test_flex_start_is_an_autoscaling_shape_not_just_a_flag(self):
        """Flex-start creates the pool at zero nodes and lets the autoscaler pull capacity."""
        flags = server._node_pool_provisioning_flags("flex-start", num_nodes="1")
        self.assertIn("--num-nodes=0", flags)
        self.assertIn("--total-max-nodes=1", flags)
        self.assertIn("--enable-autoscaling", flags)

    def test_a_node_pool_has_no_self_destruct(self):
        """The twin rig's instances carry --max-run-duration + DELETE. A node pool cannot.

        This is the single most expensive difference between the two paths: on Compute
        Engine a forgotten demo VM stops itself, and here nothing does. If a run-duration
        flag ever appears in this mapping it is being copied from the twin and will be
        rejected by gcloud — but the real point is that teardown is a manual step.
        """
        for model in PROVISIONING_MODELS:
            flags = server._node_pool_provisioning_flags(model, "1")
            self.assertFalse(
                any("max-run-duration" in f or "instance-termination-action" in f for f in flags),
                f"{model} emitted an instance-scheduling flag: {flags}",
            )

    def test_reservation_bound_targets_a_specific_reservation(self):
        flags = server._node_pool_provisioning_flags("reservation-bound", "1", reservation_name="my-res")
        self.assertIn("--reservation-affinity=specific", flags)
        self.assertIn("--reservation=my-res", flags)

    def test_no_reservation_name_emits_no_half_formed_flag(self):
        with patch.object(server, "RESERVATION_NAME", ""):
            flags = server._node_pool_provisioning_flags("reservation-bound", "1")
        self.assertFalse(any(f.startswith("--reservation=") for f in flags), flags)

    async def test_create_rejects_unknown_model_before_calling_gcloud(self):
        mock_run = AsyncMock(return_value=(0, "", ""))
        with patch("server.run_command", new=mock_run):
            result = await server.create_tpu_node_pool(provisioning_model="preemptible")
        self.assertTrue(result.startswith("❌"))
        mock_run.assert_not_called()

    async def test_create_pool_refuses_without_a_cluster(self):
        """A node pool cannot exist on its own — this is the whole shape of the GKE path."""
        with patch("server._cluster_exists", new=AsyncMock(return_value=False)):
            result = await server.create_tpu_node_pool()
        self.assertIn("create_gke_cluster", result)


class TestOffTheOtherControlPlanes(unittest.IsolatedAsyncioTestCase):
    """This rig provisions through GKE — and unlike last time, the whole module is checked."""

    def test_no_tool_shells_to_instances_create_or_tpu_vm(self):
        """The twin rig learned this the expensive way: it asserted 'we are off the TPU API'
        about the code as a whole and tested it on ONE function, while four tools quietly
        shelled to `tpus tpu-vm ssh`. So this reads the module source rather than one call
        path: no `compute instances create`, no `tpu-vm`, no `queued-resources` anywhere.
        """
        with open(server.__file__) as f:
            source = f.read()
        for forbidden in ('"instances",\n        "create"', '"tpu-vm"', '"queued-resources"'):
            self.assertNotIn(forbidden, source, f"server.py still reaches for {forbidden}")

    def test_kubectl_is_always_pinned_to_this_rigs_context(self):
        """An unpinned kubectl inherits the machine's current context.

        That is global state shared with every cluster this workstation has ever fetched
        credentials for, so an unpinned `kubectl delete` is one stale context away from
        acting on somebody else's cluster.
        """
        argv = server._kubectl(["get", "pods"])
        self.assertEqual(argv[0], "kubectl")
        self.assertTrue(argv[1].startswith("--context=gke_"), argv)
        self.assertIn(server.GKE_CLUSTER_NAME, argv[1])

    def test_secret_access_names_the_project(self):
        """gcloud's DEFAULT project on this workstation is an expired qwiklabs lab, so a
        secrets call without --project fails with a permission error naming a project that
        appears nowhere in this rig. The shell path always passed it; the MCP tool did not.
        """
        seen = []

        async def _run(cmd, timeout=60):
            seen.append(cmd)
            return 0, "tok", ""

        async def _go():
            with patch("server.run_command", new=AsyncMock(side_effect=_run)):
                await server.get_secret("hf-token")

        asyncio.run(_go())
        self.assertTrue(any(a == f"--project={server.PROJECT_ID}" for a in seen[0]), seen[0])


class TestManifest(unittest.IsolatedAsyncioTestCase):
    """The manifest is the deployment. These pin what makes it reach the chip."""

    def setUp(self):
        self.manifest = server._render_vllm_manifest()
        self.docs = list(yaml.safe_load_all(self.manifest))
        self.deployment = next(d for d in self.docs if d["kind"] == "Deployment")
        self.pod_spec = self.deployment["spec"]["template"]["spec"]

    def test_pod_is_bound_to_a_tpu_node_and_asks_for_one_chip(self):
        """Drop either half and the pod schedules onto the e2 system node and fails there,
        which reads as a vLLM problem rather than a placement one."""
        selector = self.pod_spec["nodeSelector"]
        self.assertEqual(selector["cloud.google.com/gke-tpu-accelerator"], server.GKE_TPU_ACCELERATOR)
        self.assertEqual(selector["cloud.google.com/gke-tpu-topology"], server.TPU_TOPOLOGY)
        limits = self.pod_spec["containers"][0]["resources"]["limits"]
        self.assertEqual(str(limits["google.com/tpu"]), "1")

    def test_pod_tolerates_the_tpu_taint(self):
        """GKE taints TPU nodes google.com/tpu=present:NoSchedule."""
        keys = [t.get("key") for t in self.pod_spec.get("tolerations", [])]
        self.assertIn("google.com/tpu", keys)

    def test_manifest_serves_the_same_flags_as_the_sibling_rigs(self):
        """THE COMPARISON DEPENDS ON THIS. tpu-vllm-v6e1-2b, gce-vllm-v6e1-2b and this rig
        differ in slot 1 of their names and nothing else that matters, so a serving flag
        that drifts here silently turns a provisioning A/B into a serving A/B.
        """
        args = " ".join(self.deployment["spec"]["template"]["spec"]["containers"][0]["args"])
        for flag in _vllm_serve_flags().split(" --"):
            token = flag.strip().split(" ")[0].lstrip("-")
            if token:
                self.assertIn(token, args, f"{token} is in _vllm_serve_flags but not in the manifest")
        self.assertIn(f"--max-model-len={server.MAX_MODEL_LEN}", args)
        self.assertIn(f"--tensor-parallel-size={server.TENSOR_PARALLEL_SIZE}", args)

    def test_startup_probe_covers_the_load_and_compile(self):
        """A Running pod is not a served model: image pull, weight load and XLA precompile
        took about ten minutes on the first real deployment. Too short a probe and the
        kubelet restarts the pod partway through, forever."""
        probe = self.pod_spec["containers"][0]["startupProbe"]
        budget = probe["periodSeconds"] * probe["failureThreshold"]
        self.assertGreaterEqual(budget, 900, f"startup budget is only {budget}s")

    def test_the_template_carries_no_dollar_in_prose(self):
        """envsubst tolerates a dollar sign in a comment; string.Template raises on it, so
        the MCP path breaks while the shell path renders happily. That happened."""
        with open(server.MANIFEST_TEMPLATE) as f:
            for i, line in enumerate(f, 1):
                if line.lstrip().startswith("#"):
                    self.assertNotIn("$", line, f"line {i} puts a dollar sign in a comment")


class TestDeployment(unittest.IsolatedAsyncioTestCase):
    async def test_token_never_reaches_the_process_table(self):
        """`kubectl create secret --from-literal=token=...` puts the HF token in argv, where
        any user on the machine can read it out of ps. It goes through a 0600 file instead.
        """
        applied = []
        seen_argv = []

        async def _run(cmd, timeout=60):
            seen_argv.append(cmd)
            return 0, "configured", ""

        async def _apply(manifest, cluster_name="", location=""):
            applied.append(manifest)
            return True, "configured"

        with (
            patch("server.run_command", new=AsyncMock(side_effect=_run)),
            patch("server.get_secret", new=AsyncMock(return_value="hf_secret_value")),
            patch("server._kubectl_apply", new=AsyncMock(side_effect=_apply)),
        ):
            result = await server.deploy_vllm()

        self.assertTrue(result.startswith("🚀"), result)
        self.assertFalse(
            any("hf_secret_value" in " ".join(c) for c in seen_argv),
            "the token appeared in a command line",
        )
        self.assertTrue(any("kind: Secret" in m for m in applied))
        self.assertTrue(any(base64.b64encode(b"hf_secret_value").decode() in m for m in applied))

    async def test_cost_never_claims_a_node_pool_stops_itself(self):
        """The twin rig tells you flex-start "self-terminates at --max-run-duration", because a
        Compute Engine instance carries one. A node pool does not. That sentence survived the
        port and was live in the cost output for one run — a confident wrong statement about
        money, which is the one kind this rig's estimator is written to avoid."""
        with (
            patch("server._lookup_tpu_rate", new=AsyncMock(return_value=(1.35, "h", "DWS Defined Duration V6e"))),
            patch("server._parse_topology", return_value=1),
        ):
            out = await server.estimate_deployment_cost(provisioning_model="flex-start", hours=1)
        self.assertNotIn("self-terminates", out)
        self.assertIn("destroy_tpu_node_pool", out)

    async def test_deploy_aborts_when_the_secret_is_missing(self):
        with (
            patch("server._ensure_cluster_credentials", new=AsyncMock(return_value=(True, ""))),
            patch("server.get_secret", new=AsyncMock(return_value=None)),
            patch("server._kubectl_apply", new=AsyncMock(return_value=(True, ""))) as apply_mock,
        ):
            result = await server.deploy_vllm()
        self.assertTrue(result.startswith("❌"))
        apply_mock.assert_not_called()

    async def test_destroy_only_touches_the_named_cluster(self):
        """The sibling's manage_queued_resource deletes every QR in the zone that is not the
        named primary. Nothing here sweeps: one rig's teardown must not remove another's."""
        seen = []

        async def _run(cmd, timeout=60):
            seen.append(cmd)
            return 0, "", ""

        with patch("server.run_command", new=AsyncMock(side_effect=_run)):
            await server.destroy_gke_cluster(cluster_name="only-this-one", location="europe-west4-a")
        self.assertIn("only-this-one", seen[0])
        self.assertNotIn("--all", " ".join(seen[0]))
        self.assertFalse(any("list" in c for c in seen), "teardown must not enumerate other clusters")


class TestServiceDiscovery(unittest.IsolatedAsyncioTestCase):
    """The endpoint is a Service. A node IP is the wrong answer, not a fallback."""

    @staticmethod
    def _svc(ip=None, svc_type="LoadBalancer"):
        ingress = [{"ip": ip}] if ip else []
        return {"spec": {"type": svc_type}, "status": {"loadBalancer": {"ingress": ingress}}}

    @staticmethod
    def _pod(ready=True, phase="Running"):
        return {
            "items": [
                {
                    "metadata": {"name": "vllm-gemma4-abc"},
                    "spec": {"nodeName": "gke-tpu-1"},
                    "status": {"phase": phase, "containerStatuses": [{"ready": ready}]},
                }
            ]
        }

    def _router(self, svc, pods):
        async def _run(cmd, timeout=60):
            if "svc" in cmd:
                return 0, json.dumps(svc), ""
            if "pods" in cmd:
                return 0, json.dumps(pods), ""
            return 1, "", f"unexpected: {cmd}"

        return _run

    async def test_serving_endpoint_comes_from_the_load_balancer(self):
        with (
            patch(
                "server.run_command", new=AsyncMock(side_effect=self._router(self._svc("34.91.103.13"), self._pod()))
            ),
            patch("server._probe_vllm", new=AsyncMock(return_value="google/gemma-4-E2B-it")),
        ):
            node = await _discover_vllm_node()
        assert node is not None
        self.assertEqual(node.url, "http://34.91.103.13:8000")
        self.assertTrue(node.serving)

    async def test_a_service_with_no_external_address_is_not_serving(self):
        """ClusterIP, or a LoadBalancer still being assigned. Returning the address anyway
        would hand callers something that hangs rather than fails."""
        with patch(
            "server.run_command",
            new=AsyncMock(side_effect=self._router(self._svc(None, "ClusterIP"), self._pod(ready=False))),
        ):
            node = await _discover_vllm_node()
        assert node is not None
        self.assertFalse(node.serving)
        self.assertEqual(node.url, "")

    async def test_a_pod_that_is_up_but_not_answering_is_returned_unprobed(self):
        """Ten minutes of pull, load and compile sit between Running and served, so callers
        need something to poll rather than None."""
        with (
            patch(
                "server.run_command",
                new=AsyncMock(side_effect=self._router(self._svc("34.1.1.1"), self._pod(ready=False))),
            ),
            patch("server._probe_vllm", new=AsyncMock(return_value=None)),
        ):
            node = await _discover_vllm_node()
        assert node is not None
        self.assertFalse(node.serving)
        self.assertEqual(node.url, "http://34.1.1.1:8000")

    async def test_discovery_never_reads_a_node_ip(self):
        """A GKE node DOES appear in `gcloud compute instances list`, so the twin rig's
        discovery call succeeds here and returns the wrong object — the node, not the
        Service. That is a quieter failure than the twin's was, and this pins it out."""
        seen = []

        async def _run(cmd, timeout=60):
            seen.append(cmd)
            if "svc" in cmd:
                return 0, json.dumps(self._svc("34.2.2.2")), ""
            return 0, json.dumps(self._pod()), ""

        with (
            patch("server.run_command", new=AsyncMock(side_effect=_run)),
            patch("server._probe_vllm", new=AsyncMock(return_value="m")),
        ):
            await _discover_vllm_node()
        flat = [" ".join(c) for c in seen]
        self.assertFalse(any("compute" in c and "instances" in c for c in flat), flat)
        self.assertTrue(all(c.startswith("kubectl") for c in flat), flat)


class TestQuotaMapping(unittest.IsolatedAsyncioTestCase):
    """Kept from the Compute Engine twin: GKE node pools spend the same CE pools."""

    async def test_flex_start_is_metered_by_the_preemptible_quota(self):
        """Documented, not inferred: "When you create a Flex-start VM, preemptible quota is
        consumed." Easy to get backwards, and this rig's docs did, twice."""
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
        self.assertTrue(
            server.GCE_SPOT_QUOTA_ID.endswith("per-project-region"),
            f"expected a region-scoped preemptible id, got {server.GCE_SPOT_QUOTA_ID}",
        )

    async def test_zones_come_from_the_machine_type_not_the_quota(self):
        """Both Compute Engine TPU quotas are REGIONAL, so neither can produce a zone list —
        the same reason the twin sweeps by machine type. Quota is a ceiling, not an
        allocation: a zone with 1536 chips of quota and no hardware fails like a zone with none.
        """
        seen = []

        async def _run(cmd, timeout=60):
            seen.append(cmd)
            return 0, "", ""

        with patch("server.run_command", new=AsyncMock(side_effect=_run)):
            await server._zones_with_machine_type("ct6e-standard-1t")
        self.assertIn("machine-types", seen[0])


if __name__ == "__main__":
    unittest.main()
