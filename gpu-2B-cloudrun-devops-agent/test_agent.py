import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

from server import mcp


class TestDevOpsAgent(unittest.IsolatedAsyncioTestCase):
    EXPECTED_TOOLS = {
        "cloudrun_analyze_cloud_logging",
        "cloudrun_analyze_gpu_logs",
        "cloudrun_check_gpu_quotas",
        "cloudrun_deploy",
        "cloudrun_destroy",
        "cloudrun_get_deployment_config",
        "cloudrun_get_endpoint",
        "cloudrun_get_endpoint_url",
        "cloudrun_get_gpu_deployment_config",
        "cloudrun_get_help",
        "cloudrun_get_huggingface_model_copy_instructions",
        "cloudrun_get_huggingfacehub_download_path",
        "cloudrun_get_metrics",
        "cloudrun_get_model_details",
        "cloudrun_get_system_status",
        "cloudrun_get_vertex_ai_model_copy_instructions",
        "cloudrun_list_bucket_models",
        "cloudrun_list_vertex_models",
        "cloudrun_query",
        "cloudrun_query_gemma4",
        "cloudrun_query_gemma4_with_stats",
        "cloudrun_run_benchmark",
        "cloudrun_save_hf_token",
        "cloudrun_status",
        "cloudrun_suggest_sre_remediation",
        "cloudrun_update_scaling",
        "cloudrun_verify_model_health",
    }

    async def test_tools_registered(self):
        """The registered tool names must match EXPECTED_TOOLS exactly.

        Exact-match (not assertIn) so that adding, removing, or renaming a tool without updating
        the contract fails the test.
        """
        tools = {t.name for t in await mcp.list_tools()}
        self.assertEqual(self.EXPECTED_TOOLS, tools)

    @patch("server.subprocess.run")
    def test_update_cloudrun_scaling(self, mock_run):
        """Test the cloudrun_update_scaling tool with mock subprocess."""
        from server import cloudrun_update_scaling

        # Setup mock behavior
        mock_result = MagicMock()
        mock_result.stdout = "Scaling updated successful"
        mock_run.return_value = mock_result

        result = cloudrun_update_scaling(min_instances=1, max_instances=2, service_name="test-service")

        # Verify result
        self.assertIn("Successfully updated scaling for test-service to min=1, max=2", result)
        self.assertIn("Scaling updated successful", result)

        # Verify subprocess call
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gcloud")
        self.assertEqual(cmd[1], "beta")
        self.assertEqual(cmd[2], "run")
        self.assertEqual(cmd[3], "services")
        self.assertEqual(cmd[4], "update")
        self.assertEqual(cmd[5], "test-service")
        self.assertIn("--min-instances=1", cmd)
        self.assertIn("--max-instances=2", cmd)
        # Bounds alone do not change the scaling mode, so a manual-scaling pin would survive.
        self.assertIn("--scaling=auto", cmd)

    @patch("server.subprocess.run")
    async def test_deploy_cloudrun(self, mock_run):
        """Test the cloudrun_deploy tool with mock subprocess."""
        from server import cloudrun_deploy

        # Setup mock behavior
        mock_result = MagicMock()
        mock_result.stdout = "Deployment successful"
        mock_run.return_value = mock_result

        result = await cloudrun_deploy(
            service_name="test-service",
            model_path="test-model",
            bucket_name="test-bucket",
        )

        # Verify result
        self.assertIn("Successfully deployed test-service", result)
        self.assertIn("Deployment successful", result)

        # Verify subprocess call
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gcloud")
        self.assertEqual(cmd[1], "beta")
        self.assertEqual(cmd[2], "run")
        self.assertEqual(cmd[3], "deploy")
        self.assertEqual(cmd[4], "test-service")
        self.assertIn("--image=vllm/vllm-openai:v0.26.0-cu129", cmd)
        # Without --scaling=auto a stale manual-scaling setting is inherited and the service 503s.
        self.assertIn("--scaling=auto", cmd)
        self.assertIn("--set-env-vars=VLLM_ENABLE_CUDA_COMPATIBILITY=0", cmd)
        self.assertIn(
            "--add-volume=name=model-volume,type=cloud-storage,bucket=test-bucket,readonly=true",
            cmd,
        )
        self.assertIn(
            "--args=--model=/mnt/models/test-model,--dtype=bfloat16,--max-model-len=16384,--disable-chunked-mm-input,--gpu-memory-utilization=0.95,--kv-cache-dtype=fp8,--tensor-parallel-size=1,--max-num-seqs=8,--enable-chunked-prefill,--max-num-batched-tokens=4096,--enable-auto-tool-choice,--tool-call-parser=gemma4,--reasoning-parser=gemma4,--async-scheduling,--limit-mm-per-prompt={},--host=0.0.0.0,--port=8000",
            cmd,
        )

    @patch("server.subprocess.run")
    async def test_deploy_cloudrun_hf(self, mock_run):
        """Test the cloudrun_deploy tool pulling directly from Hugging Face."""
        from server import cloudrun_deploy

        # Setup mock behavior
        mock_result = MagicMock()
        mock_result.stdout = "Deployment successful"
        mock_run.return_value = mock_result

        result = await cloudrun_deploy(
            service_name="test-service",
            model_path="google/gemma-4-E2B-it",
            bucket_name="test-bucket",
        )

        # Verify result
        self.assertIn("Successfully deployed test-service", result)
        self.assertIn("Deployment successful", result)

        # Verify subprocess call does not use FUSE volume and sets secrets
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertIn("--set-secrets=HF_TOKEN=hf-token:latest", cmd)
        self.assertNotIn(
            "--add-volume=name=model-volume,type=cloud-storage,bucket=test-bucket,readonly=true",
            cmd,
        )
        self.assertIn(
            "--args=--model=google/gemma-4-E2B-it,--dtype=bfloat16,--max-model-len=16384,--disable-chunked-mm-input,--gpu-memory-utilization=0.95,--kv-cache-dtype=fp8,--tensor-parallel-size=1,--max-num-seqs=8,--enable-chunked-prefill,--max-num-batched-tokens=4096,--enable-auto-tool-choice,--tool-call-parser=gemma4,--reasoning-parser=gemma4,--async-scheduling,--limit-mm-per-prompt={},--host=0.0.0.0,--port=8000",
            cmd,
        )

    @patch("server.subprocess.run")
    def test_destroy_cloudrun(self, mock_run):
        """Test the cloudrun_destroy tool with mock subprocess."""
        from server import cloudrun_destroy

        # Setup mock behavior
        mock_result = MagicMock()
        mock_result.stdout = "Deletion successful"
        mock_run.return_value = mock_result

        result = cloudrun_destroy(service_name="test-service")

        # Verify result
        self.assertIn("Successfully destroyed test-service", result)
        self.assertIn("Deletion successful", result)

        # Verify subprocess call
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gcloud")
        self.assertEqual(cmd[1], "run")
        self.assertEqual(cmd[2], "services")
        self.assertEqual(cmd[3], "delete")
        self.assertEqual(cmd[4], "test-service")
        self.assertIn("--quiet", cmd)

    @patch("server.subprocess.run")
    def test_status_cloudrun(self, mock_run):
        """Test the cloudrun_status tool with mock subprocess."""
        from server import cloudrun_status

        # Setup mock behavior
        mock_result = MagicMock()
        mock_result.stdout = "status: ready\nurl: http://test-url"
        mock_run.return_value = mock_result

        result = cloudrun_status(service_name="test-service")

        # Verify result
        self.assertIn("Status for test-service", result)
        self.assertIn("status: ready", result)

        # Verify subprocess call
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gcloud")
        self.assertEqual(cmd[1], "run")
        self.assertEqual(cmd[2], "services")
        self.assertEqual(cmd[3], "describe")
        self.assertEqual(cmd[4], "test-service")
        self.assertIn(
            "--format=yaml(status.conditions,status.latestCreatedRevisionName,status.url)",
            cmd,
        )

    @patch("server.discover_vllm_url")
    def test_get_vllm_url_caches_per_service(self, mock_discover):
        """The URL cache is keyed by service name, so distinct services don't collide."""
        import server

        server.invalidate_vllm_url_cache()
        mock_discover.side_effect = lambda name: f"https://{name}.run.app"

        self.assertEqual(server.get_vllm_url("service-a"), "https://service-a.run.app")
        self.assertEqual(server.get_vllm_url("service-b"), "https://service-b.run.app")
        # Second lookup of service-a is served from the cache, not re-discovered.
        self.assertEqual(server.get_vllm_url("service-a"), "https://service-a.run.app")
        self.assertEqual(mock_discover.call_count, 2)

        server.invalidate_vllm_url_cache()

    @patch("server.discover_vllm_url")
    @patch("server.subprocess.run")
    def test_destroy_cloudrun_invalidates_url_cache(self, mock_run, mock_discover):
        """Destroying a service must drop its cached URL so queries don't hit a dead endpoint."""
        import server

        server.invalidate_vllm_url_cache()
        mock_discover.return_value = "https://test-service.run.app"
        mock_run.return_value = MagicMock(stdout="Deletion successful")

        # Prime the cache, then destroy the service.
        self.assertEqual(server.get_vllm_url("test-service"), "https://test-service.run.app")
        server.cloudrun_destroy(service_name="test-service")
        self.assertNotIn("test-service", server._VLLM_URL_CACHE)

        server.invalidate_vllm_url_cache()

    @patch("server.discover_vllm_url")
    @patch("server.subprocess.run")
    async def test_deploy_cloudrun_invalidates_url_cache(self, mock_run, mock_discover):
        """A deploy can mint a new URL, so the cached one must be dropped."""
        import server

        server.invalidate_vllm_url_cache()
        mock_discover.return_value = "https://old-url.run.app"
        mock_run.return_value = MagicMock(stdout="Deployment successful")

        self.assertEqual(server.get_vllm_url("test-service"), "https://old-url.run.app")
        await server.cloudrun_deploy(service_name="test-service", model_path="test-model")
        self.assertNotIn("test-service", server._VLLM_URL_CACHE)

        server.invalidate_vllm_url_cache()

    @patch("server.discover_vllm_url")
    def test_get_cloudrun_endpoint_honours_service_name(self, mock_discover):
        """cloudrun_get_endpoint_url resolves the requested service, not just the default."""
        import server

        server.invalidate_vllm_url_cache()
        mock_discover.side_effect = lambda name: f"https://{name}.run.app"

        result = server.cloudrun_get_endpoint_url(service_name="other-service")
        self.assertIn("https://other-service.run.app", result)
        self.assertIn("✅", result)
        mock_discover.assert_called_once_with("other-service")

        server.invalidate_vllm_url_cache()

    @patch("server.discover_vllm_url")
    def test_get_cloudrun_endpoint_reports_discovery_failure(self, mock_discover):
        """Discovery failures come back as an error string, not an exception."""
        import server

        server.invalidate_vllm_url_cache()
        mock_discover.return_value = None

        result = server.cloudrun_get_endpoint_url(service_name="missing-service")
        self.assertIn("🔴", result)
        self.assertIn("missing-service", result)

        server.invalidate_vllm_url_cache()

    @patch("server.subprocess.run")
    def test_vllm_base_url_applies_only_to_default_service(self, mock_run):
        """VLLM_BASE_URL pins one specific endpoint, so it must not answer for other service names.

        Every shipped MCP client config sets VLLM_BASE_URL. If discovery honoured it for any
        service_name, every per-service tool would report the default service's URL and health
        under whatever name the caller passed.
        """
        import server

        pinned = "https://pinned-default.run.app"
        mock_run.return_value = MagicMock(returncode=0, stdout="https://other-service.run.app\n")

        with patch("server.VLLM_BASE_URL", pinned):
            self.assertEqual(server.discover_vllm_url(server.DEFAULT_SERVICE_NAME), pinned)
            # A non-default service is still discovered via gcloud rather than handed the pin.
            self.assertEqual(server.discover_vllm_url("other-service"), "https://other-service.run.app")

    @patch("server.get_active_model_name")
    @patch("server.get_vllm_client")
    @patch("server.get_auth_token")
    @patch("server.get_vllm_url")
    async def test_run_benchmark_aborts_when_warmup_fails(
        self, mock_url, mock_token, mock_client_factory, mock_model_name
    ):
        """A cold service must produce a clear message, not an all-zeros results table."""
        import server

        mock_url.return_value = "https://test-service.run.app"
        mock_token.return_value = "token"
        mock_client_factory.return_value = MagicMock()
        mock_model_name.return_value = "test-model-name"

        mock_http = MagicMock()

        async def failing_post(*args, **kwargs):
            raise RuntimeError("connect timeout")

        mock_http.post = failing_post
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("server.httpx.AsyncClient", return_value=mock_http):
            result = await server.cloudrun_run_benchmark(num_prompts=2, max_concurrency=1)

        self.assertIn("Benchmark aborted", result)
        self.assertIn("connect timeout", result)
        # The warmup targets chat completions: /v1/completions returns an empty token on -it models.
        self.assertIn("/v1/chat/completions", result)

    def _mock_benchmark_http(self, latencies):
        """Builds an httpx.AsyncClient stand-in whose POSTs succeed with a fixed latency sequence.

        latencies are consumed in order and applied as real sleeps, so the values the benchmark
        measures with perf_counter are ordered the way the sequence is.
        """
        import asyncio as _asyncio

        pending = list(latencies)

        async def post(*args, **kwargs):
            await _asyncio.sleep(pending.pop(0) if pending else 0.0)
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {"usage": {"completion_tokens": 10}}
            return response

        mock_http = MagicMock()
        mock_http.post = post
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        return mock_http

    @patch("server.get_active_model_name")
    @patch("server.get_vllm_client")
    @patch("server.get_auth_token")
    @patch("server.get_vllm_url")
    async def test_run_benchmark_clamps_non_positive_concurrency(
        self, mock_url, mock_token, mock_client_factory, mock_model_name
    ):
        """max_concurrency=0 must not build a Semaphore(0) sweep level, which would never return."""
        import server

        mock_url.return_value = "https://test-service.run.app"
        mock_token.return_value = "token"
        mock_client_factory.return_value = MagicMock()
        mock_model_name.return_value = "test-model-name"

        with patch("server.httpx.AsyncClient", return_value=self._mock_benchmark_http([])):
            # Without the clamp this call hangs forever rather than failing.
            result = await asyncio.wait_for(server.cloudrun_run_benchmark(num_prompts=2, max_concurrency=0), timeout=10)

        self.assertIn("GPU Benchmark Results", result)
        self.assertIn("| 1 |", result)

    @patch("server.get_active_model_name")
    @patch("server.get_vllm_client")
    @patch("server.get_auth_token")
    @patch("server.get_vllm_url")
    async def test_run_benchmark_p95_is_nearest_rank_not_max(
        self, mock_url, mock_token, mock_client_factory, mock_model_name
    ):
        """P95 of 20 samples is the 19th slowest, not the slowest.

        Truncating (int(n * 0.95)) indexes the last element whenever n * 0.95 is a whole number,
        which is exactly the default num_prompts=20 — reporting the max as the P95.
        """
        import server

        mock_url.return_value = "https://test-service.run.app"
        mock_token.return_value = "token"
        mock_client_factory.return_value = MagicMock()
        mock_model_name.return_value = "test-model-name"

        # 1 warmup + 20 sweep requests at concurrency 1. One clear outlier at the end.
        latencies = [0.0] + [0.01] * 19 + [0.60]

        with patch("server.httpx.AsyncClient", return_value=self._mock_benchmark_http(latencies)):
            with patch("server.open", mock_open()):
                result = await server.cloudrun_run_benchmark(num_prompts=20, max_concurrency=1)

        # Table row: | Concurrency | Success Rate | Req/s | Tokens/s | Avg Latency | P95 Latency |
        row = next(line for line in result.splitlines() if line.startswith("| 1 |"))
        p95 = float(row.split("|")[6].strip().rstrip("s"))
        self.assertLess(p95, 0.5, f"P95 picked up the outlier — it reported the max. Row: {row}")

    @patch("server.get_active_model_name")
    @patch("server.get_vllm_client")
    @patch("server.get_auth_token")
    @patch("server.get_vllm_url")
    async def test_run_benchmark_survives_unwritable_csv(
        self, mock_url, mock_token, mock_client_factory, mock_model_name
    ):
        """The sweep has already run when the CSV is written, so a write failure must not lose it."""
        import server

        mock_url.return_value = "https://test-service.run.app"
        mock_token.return_value = "token"
        mock_client_factory.return_value = MagicMock()
        mock_model_name.return_value = "test-model-name"

        with patch("server.httpx.AsyncClient", return_value=self._mock_benchmark_http([])):
            with patch("server.open", side_effect=PermissionError("read-only file system")):
                result = await server.cloudrun_run_benchmark(num_prompts=2, max_concurrency=1)

        # Results still reported, failure surfaced as a string rather than raised.
        self.assertIn("GPU Benchmark Results", result)
        self.assertIn("could not be saved", result)
        self.assertIn("read-only file system", result)

    def test_default_service_name_is_env_driven(self):
        """SERVICE_NAME must reach the server, so a pinned VLLM_BASE_URL can name the same service."""
        import importlib

        import server

        with patch.dict(os.environ, {"SERVICE_NAME": "custom-service"}):
            reloaded = importlib.reload(server)
            self.assertEqual(reloaded.DEFAULT_SERVICE_NAME, "custom-service")

        # Restore the module-level default for the rest of the suite.
        importlib.reload(server)

    async def test_resources_registered(self):
        """Verify that the expected resources are registered with MCPServer."""
        resources = [str(r.uri) for r in await mcp.list_resources()]
        self.assertIn("config://vllm-deployment-template", resources)

    def test_get_huggingface_model_copy_instructions(self):
        """Test the output of the Hugging Face model copy instructions tool."""
        from server import cloudrun_get_huggingface_model_copy_instructions

        instructions = cloudrun_get_huggingface_model_copy_instructions("test/slug", "test-bucket")
        self.assertIn("test/slug", instructions)
        self.assertIn("test-bucket", instructions)
        self.assertIn("slug", instructions)
        self.assertIn("huggingface-cli download test/slug", instructions)

    def test_get_vertex_ai_model_copy_instructions(self):
        """Test the output of the Vertex AI model copy instructions tool."""
        from server import cloudrun_get_vertex_ai_model_copy_instructions

        instructions = cloudrun_get_vertex_ai_model_copy_instructions("gemma-4-E2B-it")
        self.assertIn("gemma-4-E2B-it", instructions)
        self.assertIn("Vertex AI Model Garden", instructions)
        self.assertIn("gcloud storage cp", instructions)

    @patch("server.storage.Client")
    def test_list_bucket_models_mock(self, mock_storage_client):
        """Test the output of the GCS bucket listing tool with mocks."""
        from server import cloudrun_list_bucket_models

        # Setup mock behavior
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.name = "gemma-4-E2B-it/config.json"
        mock_blob.size = 1024 * 1024 * 5  # 5 MB
        mock_bucket.list_blobs.return_value = [mock_blob]
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        result = cloudrun_list_bucket_models("mock-bucket")
        self.assertIn("mock-bucket", result)
        self.assertIn("gemma-4-E2B-it/config.json", result)
        self.assertIn("5.00 MB", result)

    @patch("server.secretmanager.SecretManagerServiceClient")
    async def test_save_hf_token(self, mock_client_class):
        """Test cloudrun_save_hf_token tool saves token to Secret Manager."""
        from server import cloudrun_save_hf_token

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.add_secret_version.return_value = MagicMock(name="projects/test/secrets/hf-token/versions/1")

        result = await cloudrun_save_hf_token("test-token")
        self.assertIn("Token saved", result)

    @patch("server.subprocess.run")
    def test_check_gpu_quotas(self, mock_run):
        """Test cloudrun_check_gpu_quotas tool formats metrics correctly."""
        from server import cloudrun_check_gpu_quotas

        mock_result = MagicMock()
        mock_result.stdout = json.dumps(
            {
                "quotas": [
                    {"metric": "NVIDIA_L4_GPUS", "limit": 1.0, "usage": 0.0},
                    {"metric": "CPUS", "limit": 24.0, "usage": 4.0},
                ]
            }
        )
        mock_run.return_value = mock_result

        result = cloudrun_check_gpu_quotas(region="us-east4")
        self.assertIn("GPU Quotas for region `us-east4`", result)
        self.assertIn("NVIDIA_L4_GPUS", result)
        self.assertNotIn("CPUS", result)  # Non-GPU metrics should be filtered out

        # Verify command arguments
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gcloud")
        self.assertEqual(cmd[1], "compute")
        self.assertEqual(cmd[2], "regions")
        self.assertEqual(cmd[3], "describe")
        self.assertEqual(cmd[4], "us-east4")
        self.assertIn("--format=json(quotas)", cmd)

    async def test_get_help(self):
        """Test cloudrun_get_help returns correct tool and region information."""
        from server import cloudrun_get_help

        result = await cloudrun_get_help()
        self.assertIn("Cloud Run Gemma 4 SRE Agent Help", result)
        self.assertIn("cloudrun_deploy", result)
        self.assertIn("NVIDIA L4", result)
        self.assertIn("us-east4", result)

    @patch("server.get_vllm_client")
    @patch("server.get_active_model_name")
    async def test_verify_model_health(self, mock_model_name, mock_client_factory):
        """Test cloudrun_verify_model_health parses model response and calculates latency."""
        from server import cloudrun_verify_model_health

        mock_model_name.return_value = "test-model-name"
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        mock_message.content = "Yes, the model is active and running."
        mock_choice.message = mock_message
        mock_choice.message.content = "Yes, the model is active and running."
        mock_completion.choices = [mock_choice]

        # Async mock for client.chat.completions.create
        async def mock_create(*args, **kwargs):
            return mock_completion

        mock_chat.create = mock_create
        mock_client.chat = MagicMock()
        mock_client.chat.completions = mock_chat
        mock_client_factory.return_value = mock_client

        result = await cloudrun_verify_model_health()
        self.assertIn("Model health check PASSED", result)
        self.assertIn("test-model-name", result)
        self.assertIn("Yes, the model is active and running.", result)

    @patch("server.get_vllm_client")
    @patch("server.get_active_model_name")
    async def test_query_gemma4(self, mock_model_name, mock_client_factory):
        """Test cloudrun_query_gemma4 queries the model via chat completions."""
        from server import cloudrun_query_gemma4

        mock_model_name.return_value = "test-model-name"
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        mock_message.content = "Response from Gemma"
        mock_choice.message = mock_message
        mock_choice.message.content = "Response from Gemma"
        mock_completion.choices = [mock_choice]

        async def mock_create(*args, **kwargs):
            return mock_completion

        mock_chat.create = mock_create
        mock_client.chat = MagicMock()
        mock_client.chat.completions = mock_chat
        mock_client_factory.return_value = mock_client

        result = await cloudrun_query_gemma4("Hello")
        self.assertEqual(result, "Response from Gemma")

    @patch("server.get_vllm_client")
    @patch("server.get_active_model_name")
    async def test_query_gemma4_with_stats(self, mock_model_name, mock_client_factory):
        """Test cloudrun_query_gemma4_with_stats collects performance metrics."""
        from server import cloudrun_query_gemma4_with_stats

        mock_model_name.return_value = "test-model-name"
        mock_client = MagicMock()
        mock_chat = MagicMock()

        # We need mock chunks to simulate streaming
        class MockChunk:
            def __init__(self, content):
                mock_delta = MagicMock()
                mock_delta.content = content
                mock_choice = MagicMock()
                mock_choice.delta = mock_delta
                self.choices = [mock_choice]

        chunks = [MockChunk("Hello"), MockChunk(" world!")]

        # Async generator mock
        async def mock_create_stream(*args, **kwargs):
            async def async_gen():
                for chunk in chunks:
                    yield chunk

            return async_gen()

        mock_chat.create = mock_create_stream
        mock_client.chat = MagicMock()
        mock_client.chat.completions = mock_chat
        mock_client_factory.return_value = mock_client

        result = await cloudrun_query_gemma4_with_stats("Hello")
        self.assertIn("Performance Stats", result)
        self.assertIn("test-model-name", result)
        self.assertIn("Hello world!", result)

    @patch("server.get_vllm_client")
    @patch("server.get_vllm_url")
    @patch("server.get_auth_token")
    @patch("server.httpx.AsyncClient")
    async def test_get_model_details(
        self, mock_httpx_client_class, mock_auth_token, mock_vllm_url, mock_client_factory
    ):
        """Test cloudrun_get_model_details formats models list and health status."""
        from server import cloudrun_get_model_details

        mock_vllm_url.return_value = "http://test-url"
        mock_auth_token.return_value = "mock-token"

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_models_response = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "test-model-id"
        mock_model.object = "model"
        mock_model.owned_by = "google"
        mock_models_response.data = [mock_model]

        async def mock_list():
            return mock_models_response

        mock_client.models.list = mock_list
        mock_client_factory.return_value = mock_client

        # Mock HTTPX response
        mock_httpx_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_httpx_client.get = mock_get
        mock_httpx_client.__aenter__.return_value = mock_httpx_client
        mock_httpx_client_class.return_value = mock_httpx_client

        result = await cloudrun_get_model_details()
        self.assertIn("Model Details (http://test-url)", result)
        self.assertIn("test-model-id", result)
        self.assertIn("Healthy", result)


if __name__ == "__main__":
    unittest.main()
