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
        self.assertIn("--accelerator-type=v5litepod-1", config)
        self.assertIn("--version=v2-alpha-tpuv5-lite", config)

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
        self.assertIn("No ACTIVE Queued Resource with a reachable vLLM service found", result)

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


if __name__ == "__main__":
    unittest.main()
