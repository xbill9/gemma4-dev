"""Offline regression tests for the G4dn MCP server.

No AWS, no network, no GPU, no docker.

This rig sits between two siblings and inherits a DIFFERENT half of each, so the
copy-paste hazards run in both directions at once:

  * from `gpu-vllm-g6-2b` (the fork parent, x86_64 + Ada SM 8.9): the x86_64 AMI
    filter and the no-build bootstrap are RIGHT and carry. The dtype policy is
    WRONG -- Ada has bf16 and fp8, Turing has neither.
  * from `gpu-vllm-g5g-2b` (Turing, aarch64): the dtype policy and the Triton
    shared-memory ceiling are RIGHT and carry. The 67-minute from-source build,
    the CUDA toolkit, Rust and the prebuilt AMI are WRONG -- the published amd64
    manifest already carries SM 7.5.

The sharpest tests here are the patch ones. The clamp is the only reason this rig
can serve at all, and every way it can fail is SILENT: a marker that drifts from
the patch script, a payload that stops round-tripping, an anchor that stops
matching. Each of those produces a launch that reports success and dies ten
minutes later at engine start.
"""

import ast
import base64
import filecmp
import gzip
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import patch_triton_turing  # noqa: E402
import server  # noqa: E402

# A faithful miniature of vLLM v0.28.0's `unified_attention` launcher,
# transcribed from the real file on 2026-08-29 (v0.28.0 and main are
# byte-identical for it). The ORDER is the point: tile constants are assigned,
# then copied into `tile_size`, then the kernel launches. A clamp inserted at
# the launch site would be too late.
UPSTREAM_SHAPE = '''from typing import Any

from vllm.platforms import current_platform


def _get_tile_size(head_size, sliding_window, element_size, is_prefill):
    return 32 if is_prefill else 16


def unified_attention(q, out, use_td, use_3d, block_size):
    head_size = q.shape[2]
    BLOCK_M = 16

    launch_num_warps: int | None = None
    launch_num_stages: int | None = None

    TILE_SIZE_PREFILL = _get_tile_size(head_size, 0, q.element_size(), True)
    TILE_SIZE_DECODE = _get_tile_size(head_size, 0, q.element_size(), False)

    if use_td:
        TILE_SIZE_PREFILL = min(TILE_SIZE_PREFILL, block_size)
        TILE_SIZE_DECODE = min(TILE_SIZE_DECODE, block_size)

    grid: tuple[Any, ...]
    if not use_3d:
        grid = (1, 2)
        tile_size = TILE_SIZE_PREFILL
    else:
        grid = (1, 2, 3)
        tile_size = TILE_SIZE_DECODE

    launch_kwargs: dict[str, int] = {}
    if launch_num_warps is not None:
        launch_kwargs["num_warps"] = launch_num_warps
    if launch_num_stages is not None:
        launch_kwargs["num_stages"] = launch_num_stages

    kernel_unified_attention[grid](out, q, TILE_SIZE=tile_size, **launch_kwargs)
'''


def run(coro):
    import asyncio

    return asyncio.run(coro)


class ToolCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = {tool.name: tool for tool in run(server.mcp.list_tools())}

    def test_catalog(self):
        expected = {
            "create_g4dn_instance", "list_g4dn_instances", "start_g4dn_instance",
            "stop_g4dn_instance", "terminate_g4dn_instance", "verify_gpu_arch",
            "verify_triton_patch", "get_install_progress", "get_vllm_logs",
            "get_endpoint", "verify_model_health", "query_model", "save_hf_token",
            "check_g4dn_quotas", "get_deployment_config", "get_help",
        }
        self.assertEqual(set(self.tools), expected)

    def test_annotations(self):
        destructive = {
            name for name, tool in self.tools.items() if tool.annotations.destructiveHint
        }
        self.assertEqual(destructive, {"stop_g4dn_instance", "terminate_g4dn_instance"})
        for name, tool in self.tools.items():
            self.assertTrue(tool.title, name)
            self.assertTrue(tool.description, name)
            self.assertIsNotNone(tool.annotations, name)

    def test_codex_gates_name_tools_that_exist(self):
        """A gate on a tool name that does not exist FAILS OPEN and says nothing.

        On the JAX rig's G5g -> G6 fork, .codex/config.toml kept the old
        `*_g5g_*` tool names, so every destructive tool was ungated under Codex
        while appearing to be gated. This fork renames them again, to `*_g4dn_*`.
        """
        text = (ROOT / ".codex" / "config.toml").read_text()
        marker = f"[mcp_servers.{server.RIG_NAME}.tools."
        gated = {
            line[len(marker):].rstrip("]").strip()
            for line in text.splitlines()
            if line.startswith(marker)
        }
        self.assertTrue(gated, "no tools are gated at all")
        self.assertTrue(
            gated <= set(self.tools),
            f"gates name tools that do not exist: {sorted(gated - set(self.tools))}",
        )
        self.assertIn("terminate_g4dn_instance", gated)
        self.assertIn("stop_g4dn_instance", gated)

    def test_launch_defaults_to_spot_and_has_no_serving_mode(self):
        for name in ("create_g4dn_instance", "get_deployment_config"):
            schema = self.tools[name].inputSchema["properties"]
            self.assertTrue(schema["spot"]["default"], name)
            # The G5g rig's build/stock choice does not exist here: SM 7.5 is in
            # the published amd64 image, so `build` has nothing to compile and
            # `stock` nothing to fail at on the ARCH axis. The Turing patch is
            # not a mode -- it is unconditional, because there is no configuration
            # in which serving unpatched works.
            self.assertNotIn("serving", schema, name)


class G4dnTopologyTests(unittest.TestCase):
    def test_sizes_match_the_aws_product_page(self):
        expected = {
            "g4dn.xlarge": (1, 16),
            "g4dn.2xlarge": (1, 32),
            "g4dn.4xlarge": (1, 64),
            "g4dn.8xlarge": (1, 128),
            "g4dn.12xlarge": (4, 192),
            "g4dn.16xlarge": (1, 256),
            "g4dn.metal": (8, 384),
        }
        self.assertEqual(set(server._G4DN_SIZES), set(expected))
        for instance_type, (gpus, ram) in expected.items():
            self.assertTrue(server._is_g4dn(instance_type))
            self.assertEqual(server._gpu_count(instance_type), gpus)
            self.assertEqual(server._host_memory_gb(instance_type), ram)

    def test_gpu_count_is_not_monotonic_in_size(self):
        """g4dn.16xlarge is SINGLE-GPU while 12xlarge has four and metal eight.

        Anything that infers GPU count from the size suffix gets this wrong, and
        a wrong tensor-parallel size fails at engine start.
        """
        self.assertEqual(server._gpu_count("g4dn.12xlarge"), 4)
        self.assertEqual(server._gpu_count("g4dn.16xlarge"), 1)
        self.assertEqual(server._gpu_count("g4dn.metal"), 8)

    def test_vcpu_is_not_derived_from_host_ram_the_g5g_way(self):
        """G4dn is 4 GiB/vCPU; G5g was 2. The inherited `RAM // 2` doubles here."""
        self.assertEqual(server._vcpu_count("g4dn.xlarge"), 4)
        self.assertEqual(server._vcpu_count("g4dn.2xlarge"), 8)
        self.assertEqual(server._vcpu_count("g4dn.metal"), 96)
        self.assertNotEqual(
            server._vcpu_count("g4dn.xlarge"), server._host_memory_gb("g4dn.xlarge") // 2
        )

    def test_tensor_parallel_follows_gpu_count(self):
        self.assertEqual(server._tensor_parallel_size("g4dn.xlarge"), 1)
        self.assertEqual(server._tensor_parallel_size("g4dn.12xlarge"), 4)

    def test_no_g4dn_size_needs_a_swapfile(self):
        """Every G4dn size has >= 16 GiB of host RAM, twice its g5g namesake.

        So the G5g rig's g5g.xlarge rejection does NOT carry and the swap block
        never renders. Kept in the code because the threshold is a claim about
        the checkpoint (~10.2 GB to map), not about the host.
        """
        for size in server._G4DN_SIZES:
            self.assertFalse(server._needs_swap(size), size)
            self.assertNotIn("mkswap", server._user_data("google/gemma-4-E2B-it", size))

    def test_swap_gate_is_strictly_below_16_unlike_the_jax_g4dn_rig(self):
        """`gpu-jax-g4dn-2b` gates AT-OR-BELOW 16 GiB on the SAME instance type.

        That rig OOMs at exactly 16 GiB in `quantize_ple_table`, which upcasts a
        4.70 GB PLE table to float32 while the whole tree is resident -- a
        property of ITS loader. vLLM has no equivalent step, and the G5g rig
        measured a 16 GiB host needing no swapfile. Harmonising the two gates
        would provision swap nothing here needs, and would obscure that they
        encode different failures.
        """
        self.assertEqual(server._SWAP_BELOW_HOST_RAM_GB, 16)
        self.assertFalse(server._needs_swap("g4dn.xlarge"))

    def test_xlarge_is_accepted(self):
        server._validate_instance_type("g4dn.xlarge")  # must not raise

    def test_non_g4dn_rejected(self):
        for bad in ("g5g.2xlarge", "g6.xlarge", "inf2.xlarge", "g4dn.unknown"):
            with self.assertRaises(ValueError):
                server._validate_instance_type(bad)


class TuringConstraintTests(unittest.TestCase):
    """Turing (SM 7.5) has NEITHER bf16 NOR fp8. The G6 fork parent has both."""

    def test_dtype_is_float16_not_the_fork_parents_bfloat16(self):
        flags = server._serve_flags("google/gemma-4-E2B-it", "g4dn.xlarge")
        self.assertIn("--dtype float16", flags)
        # The single most likely copy-paste error from `gpu-vllm-g6-2b`, and a
        # silent one: bfloat16 does not error on Turing, PyTorch upconverts.
        self.assertNotIn("bfloat16", flags)
        self.assertEqual(server.DTYPE, "float16")

    def test_kv_cache_is_auto_and_fp8_is_not_reachable(self):
        flags = server._serve_flags("google/gemma-4-E2B-it", "g4dn.xlarge")
        self.assertIn("--kv-cache-dtype auto", flags)
        # Unlike on the G6 sibling, fp8 here is not "available but unused" --
        # there is no datapath at all.
        self.assertNotIn("fp8", flags)

    def test_attention_backend_is_unpinned(self):
        """MEASURED on the G5g sibling: vLLM v0.27 does not recognize
        VLLM_ATTENTION_BACKEND at all, and forces TRITON_ATTN for Gemma 4
        regardless, because the head dims are heterogeneous.

        So the backend is not the knob -- the TILE SIZE inside the forced kernel
        is, and that is what the patch changes.
        """
        self.assertEqual(server.ATTENTION_BACKEND, "")
        text = server._user_data("google/gemma-4-E2B-it", "g4dn.xlarge")
        # An EMPTY value must not be exported: vLLM seeing the variable set to
        # "" is not the same as not seeing it.
        self.assertNotIn("VLLM_ATTENTION_BACKEND", text)

    def test_the_from_source_build_path_is_gone(self):
        """The published amd64 manifest carries SM 7.5, so nothing compiles here.

        An arch list or a source ref would be an inert setting that looks
        meaningful -- and would suggest the G5g rig's 67-minute build applies.
        """
        for dead in ("TORCH_CUDA_ARCH_LIST", "VLLM_REF", "VLLM_STOCK_IMAGE",
                     "CUDA_TOOLKIT_PACKAGE"):
            self.assertFalse(hasattr(server, dead), f"{dead} survived the fork")

    def test_image_is_at_or_above_the_measured_vllm_floor(self):
        """v0.26.0 dies with AmbiguousGlobalPerLayerAttributeError because Gemma
        4's head_dim is per-layer.

        A MODEL constraint, not a chip one, so it holds on every part in this
        tree. The trap is that the G5g rig's VLLM_STOCK_IMAGE is v0.27.1 -- it
        only ever used that tag to reproduce the SM 7.5 absence and never served
        from it, so it is below the floor.
        """
        self.assertNotIn(":v0.27.1", server.VLLM_IMAGE)
        self.assertNotIn(":v0.26", server.VLLM_IMAGE)
        self.assertTrue(server.VLLM_IMAGE.startswith("vllm/vllm-openai:"))

    def test_image_tag_is_one_that_actually_exists(self):
        """THE FLOOR TEST ABOVE PASSED ON A TAG THAT DOES NOT EXIST.

        This rig inherited `vllm/vllm-openai:v0.27.2rc0` from `gpu-vllm-g6-2b`.
        CHECKED 2026-08-29 against both registries: 404 on Docker Hub, and no
        such git tag in vllm-project/vllm -- the sequence goes v0.27.1 then
        v0.28.0rc1. Cloud-init would have died at `docker pull`, on the FIRST
        stage, before any of this rig's interesting machinery ran.

        It survived because "not v0.27.1 and not v0.26" is trivially true of a
        tag that was never published. **A version floor that never checks the
        artifact exists is this tree's "an accepted flag is not evidence" rule,
        one level up -- in the test itself.**

        This test cannot reach the network, so it pins the specific phantom
        rather than proving existence. Re-check with:
          curl -o /dev/null -w '%{http_code}' \
            https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/<tag>
        """
        self.assertNotIn(":v0.27.2rc0", server.VLLM_IMAGE)
        self.assertNotIn("v0.27.2rc0", server.VLLM_PATCHED_IMAGE)
        # `nightly` is a MOVING tag. This rig renders deterministic user data on
        # purpose, and for triton_unified_attention.py `main` is byte-identical
        # to v0.28.0 anyway -- so nightly costs reproducibility and buys nothing.
        self.assertNotIn("nightly", server.VLLM_IMAGE)
        self.assertNotIn(":latest", server.VLLM_IMAGE)

    def test_the_two_image_names_are_distinguishable_both_ways(self):
        """verify_triton_patch decides "is the container on the stock tag?" by
        substring, so neither name may contain the other."""
        self.assertNotEqual(server.VLLM_IMAGE, server.VLLM_PATCHED_IMAGE)
        self.assertNotIn(server.VLLM_IMAGE, server.VLLM_PATCHED_IMAGE)
        self.assertNotIn(server.VLLM_PATCHED_IMAGE, server.VLLM_IMAGE)


class TuringPatchTests(unittest.TestCase):
    """The clamp is the only reason this rig can serve, and every failure is silent."""

    def test_the_marker_matches_the_patch_script(self):
        """server.py duplicates the marker so it stays importable alone.

        If these drift, the in-image verification looks for a string the patch
        never writes, and EVERY launch fails at `patch-verified-in-image` with a
        message blaming the COPY target -- a correct patch, diagnosed as a
        packaging bug.
        """
        self.assertIn(server._PATCH_MARKER, patch_triton_turing.MARKER)

    def test_payload_round_trips_and_is_deterministic(self):
        """Determinism is not cosmetic: `get_deployment_config` is supposed to be
        a reproducible artifact, and a payload that re-encodes differently makes
        every launch look like a change."""
        blob = server._patch_b64()
        self.assertEqual(
            gzip.decompress(base64.b64decode(blob)).decode(),
            (ROOT / "patch_triton_turing.py").read_text(),
        )
        self.assertEqual(blob, server._patch_b64())

    def test_user_data_stays_under_the_ec2_limit(self):
        """EC2 caps user data at 16 KB. The patch script is ~9 KB of deliberately
        comment-heavy Python, which is why it is gzipped rather than inlined --
        and why this needs a test rather than a comment."""
        for size in server._G4DN_SIZES:
            text = server._user_data("google/gemma-4-E2B-it", size)
            self.assertLess(len(text.encode()), 16384, f"{size}: user data too large")

    def test_digest_hashes_the_source_not_the_blob(self):
        """The digest rides inside the shipped payload, so hashing the payload
        would be circular."""
        import hashlib

        source = (ROOT / "patch_triton_turing.py").read_text()
        self.assertEqual(
            server._patch_digest(), hashlib.sha256(source.encode()).hexdigest()[:12]
        )
        self.assertNotIn(server._patch_digest(), server._patch_b64())

    def test_bootstrap_patches_verifies_and_serves_the_derived_tag(self):
        text = server._user_data("google/gemma-4-E2B-it", "g4dn.xlarge")
        self.assertIn(f"docker pull {server.VLLM_IMAGE}", text)
        self.assertIn(f"docker build -t {server.VLLM_PATCHED_IMAGE}", text)
        # What actually serves is the DERIVED tag. Serving the stock tag is the
        # failure verify_triton_patch exists to catch.
        self.assertIn(f"  {server.VLLM_PATCHED_IMAGE} \\\n", text)
        # Nothing is compiled: no source checkout, no toolkit, no Rust.
        self.assertNotIn("git clone", text)
        self.assertNotIn("cuda-toolkit", text)
        self.assertNotIn("rustup", text)

    def test_the_in_image_path_is_resolved_never_hardcoded(self):
        """site-packages carries the image's python version in its path, and that
        moves with the tag. A hardcoded path silently misses."""
        text = server._user_data("google/gemma-4-E2B-it", "g4dn.xlarge")
        self.assertIn("print(m.__file__)", text)
        # A hardcoded PATH, not the word: the bootstrap comment says "site-packages
        # carries the image's python version", which is the explanation, not a path.
        self.assertNotIn("/usr/local/lib/python3", text)
        self.assertNotIn("/site-packages", text)

    def test_a_failed_patch_kills_cloud_init(self):
        """Serving unpatched behind a patched tag reports success for ten minutes
        and then dies at engine start. The bootstrap must stop instead."""
        text = server._user_data("google/gemma-4-E2B-it", "g4dn.xlarge")
        self.assertIn("refusing to serve", text)
        self.assertIn("set -euxo pipefail", text)
        self.assertIn("exit 1", text)

    def test_the_built_image_is_verified_not_assumed(self):
        """A wrong COPY destination builds cleanly and leaves the module
        unpatched, so the check has to run INSIDE the built image."""
        text = server._user_data("google/gemma-4-E2B-it", "g4dn.xlarge")
        self.assertIn("patch-verified-in-image", text)
        self.assertIn(server._PATCH_MARKER, text)

    def test_patch_script_refuses_rather_than_no_ops(self):
        """The whole design rule, exercised: a file with none of the expected
        structure must exit non-zero, not quietly write nothing."""
        proc = subprocess.run(
            [sys.executable, str(ROOT / "patch_triton_turing.py"), "/dev/null"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("REFUSING TO WRITE", proc.stderr)

    def test_patch_script_ignores_num_stages_subscript_stores(self):
        """Upstream writes `launch_kwargs["num_stages"] = launch_num_stages`.

        A line-oriented pattern reads that as an assignment to `num_stages` and
        would clamp a local nothing reads -- applying HALF the fix while
        reporting success. Only real Name assignment targets count.
        """
        import ast as _ast

        tree = _ast.parse(UPSTREAM_SHAPE)
        name, note = patch_triton_turing.resolve_stages_var(tree, UPSTREAM_SHAPE)
        self.assertEqual(name, "launch_num_stages", note)

    def test_clamp_lands_BEFORE_the_tile_constants_are_consumed(self):
        """THE BUG THIS FIXTURE EXISTS FOR, caught against real upstream source.

        v0.28.0 copies the tile constants into a local well before the launch:

            if not use_3d:  tile_size = TILE_SIZE_PREFILL
            else:           tile_size = TILE_SIZE_DECODE
            if launch_num_stages is not None:
                launch_kwargs["num_stages"] = launch_num_stages
            kernel_unified_attention[grid](...)

        Anchoring on the launch site -- which is the obvious reading of the G5g
        rig's patch, and what this script did first -- would rewrite three
        variables that have ALREADY been consumed. The marker would be present,
        the in-image verification would pass, `verify_triton_patch` would report
        success, and the kernel would still request 98,304 bytes.

        So the clamp must land after the last tile ASSIGNMENT and before the
        first tile READ.
        """
        path = self._write(UPSTREAM_SHAPE)
        proc = self._run_patch(path)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = Path(path).read_text().splitlines()
        clamp = next(i for i, ln in enumerate(lines) if patch_triton_turing.MARKER in ln)
        first_read = next(
            i for i, ln in enumerate(lines) if "tile_size = TILE_SIZE_PREFILL" in ln
        )
        last_assign = max(
            i for i, ln in enumerate(lines) if "TILE_SIZE_DECODE = min(" in ln
        )
        self.assertLess(last_assign, clamp, "clamp must follow the last tile assignment")
        self.assertLess(clamp, first_read, "clamp must precede the first tile read")
        Path(path).unlink()

    def test_patch_refuses_when_the_clamp_would_have_no_effect(self):
        """A launcher whose tile constants are never read again is a file where
        the clamp is inert. Refuse rather than write a marker that means nothing.
        """
        inert = UPSTREAM_SHAPE.replace("tile_size = TILE_SIZE_PREFILL", "tile_size = 32")
        inert = inert.replace("tile_size = TILE_SIZE_DECODE", "tile_size = 16")
        path = self._write(inert)
        proc = self._run_patch(path)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("would have no effect", proc.stderr)
        Path(path).unlink()

    def test_patch_output_is_valid_python_and_idempotent(self):
        path = self._write(UPSTREAM_SHAPE)
        first = self._run_patch(path)
        self.assertEqual(first.returncode, 0, first.stderr)
        patched = Path(path).read_text()
        ast.parse(patched)  # must still be importable Python
        self.assertIn(patch_triton_turing.MARKER, patched)
        self.assertIn("TILE_SIZE_PREFILL //= 2", patched)
        second = self._run_patch(path)
        self.assertEqual(second.returncode, 0)
        self.assertEqual(Path(path).read_text(), patched, "second run was not a no-op")
        Path(path).unlink()

    def test_the_clamp_arithmetic_targets_gemma4s_global_heads_only(self):
        """Sanity-check what the clamp actually does at this model's shapes.

        fp16, BLOCK_M=16, budget 60000. Only the 512-wide global-attention
        prefill path exceeds it; the 256-wide sliding layers are untouched. If
        this ever clamps everything, the budget is wrong, not the tiles.
        """
        budget, block_m, esz = server.TURING_SMEM_BUDGET, 16, 2

        def clamp(tile, head):
            while tile > 16 and (block_m + 2 * tile) * head * esz > budget:
                tile //= 2
            return tile

        self.assertEqual(clamp(32, 256), 32, "sliding layers must not be clamped")
        self.assertEqual(clamp(32, 512), 16, "global prefill must be clamped 32 -> 16")
        self.assertLessEqual((block_m + 2 * 16) * 512 * esz, 65536)

    def _write(self, source):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(source)
            return handle.name

    def _run_patch(self, path):
        return subprocess.run(
            [sys.executable, str(ROOT / "patch_triton_turing.py"), path],
            capture_output=True, text=True,
        )

    def test_smem_budget_leaves_headroom_under_turings_hard_limit(self):
        """65,536 is the opt-in maximum and the tile arithmetic does not count
        the kernel's accumulators, so budgeting the whole limit still overflows."""
        self.assertLess(server.TURING_SMEM_BUDGET, 65536)
        self.assertGreater(server.TURING_SMEM_BUDGET, 32768)


class UserDataTests(unittest.TestCase):
    def test_shell_parses(self):
        for size in server._G4DN_SIZES:
            text = server._user_data("google/gemma-4-E2B-it", size)
            proc = subprocess.run(
                ["bash", "-n", "/dev/stdin"], input=text, text=True, capture_output=True
            )
            self.assertEqual(proc.returncode, 0, f"{size}: {proc.stderr}")

    def test_stage_markers_make_a_stall_attributable(self):
        text = server._user_data("google/gemma-4-E2B-it", "g4dn.xlarge")
        for marker in ("image-pull-start", "image-pull-done", "patch-resolve",
                       "patch-applied", "image-build-done", "patch-verified-in-image",
                       "serving-started"):
            self.assertIn(marker, text)
        self.assertIn("INSTALL_DONE", text)

    def test_token_comes_from_secrets_manager_not_user_data(self):
        # User data is readable from instance metadata by anything on the box.
        text = server._user_data("google/gemma-4-E2B-it", "g4dn.xlarge")
        self.assertIn("secretsmanager get-secret-value", text)

    def test_secret_fetch_disables_shell_tracing(self):
        # The script can run under `set -x`, and bash traces assignments WITH
        # THEIR VALUES -- so without this the token lands in the console log.
        text = server._user_data("google/gemma-4-E2B-it", "g4dn.xlarge")
        self.assertIn("set +x", text)
        self.assertLess(
            text.index("set +x"), text.index("secretsmanager"), "set +x must precede the fetch"
        )

    def test_multi_gpu_size_gets_the_right_tensor_parallel_size(self):
        self.assertIn(
            "--tensor-parallel-size 4",
            server._user_data("google/gemma-4-E2B-it", "g4dn.12xlarge"),
        )
        self.assertIn(
            "--tensor-parallel-size 8",
            server._user_data("google/gemma-4-E2B-it", "g4dn.metal"),
        )

    def test_no_mkswap_q_flag(self):
        """`mkswap -q` is a busybox flag util-linux rejects with
        `invalid option -- 'q'`. Under `set -e` it killed cloud-init on the G5g
        rig BEFORE anything logged, costing a launch."""
        for size in server._G4DN_SIZES:
            self.assertNotIn("mkswap -q", server._user_data("m", size))


class DeploymentConfigTests(unittest.TestCase):
    def test_config_is_offline_and_decodable(self):
        result = run(server.get_deployment_config(instance_type="g4dn.xlarge"))
        self.assertIn("aws ec2 run-instances", result)
        encoded = result.split("--user-data '", 1)[1].split("'", 1)[0]
        script = base64.b64decode(encoded).decode()
        self.assertIn("#!/usr/bin/env bash", script)

    def test_config_resolves_an_x86_64_ami_via_ssm(self):
        # The architecture axis is what separates this rig from gpu-vllm-g5g-2b:
        # an arm64 DLAMI cannot boot a G4dn.
        result = run(server.get_deployment_config())
        self.assertIn("aws ssm get-parameter", result)
        self.assertIn("/x86_64/", server.DLAMI_SSM_PARAMETER)
        self.assertNotIn("/arm64/", server.DLAMI_SSM_PARAMETER)
        self.assertIn("nvidia-driver-gpu", server.DLAMI_SSM_PARAMETER)

    def test_ami_name_fallback_matches_the_same_family_as_the_ssm_path(self):
        """Changing the SSM path without changing this filter in the same commit
        is a revert that reports success: the fallback quietly resolves a
        different image."""
        self.assertNotIn("ARM64", server.DLAMI_NAME)
        self.assertIn("Base", server.DLAMI_NAME)
        self.assertIn("Nvidia Driver GPU", server.DLAMI_NAME)

    def test_config_and_create_provision_the_same_root_volume(self):
        """The G5g rig PRINTED VolumeSize=200 while its create tool LAUNCHED 100.

        A copy-pasteable repro command that provisions a different volume from
        the tool it documents is how a manual reproduction fails to reproduce.
        """
        result = run(server.get_deployment_config())
        self.assertIn(f"VolumeSize={server.ROOT_VOLUME_GB}", result)
        self.assertIn(f"Throughput={server.ROOT_VOLUME_THROUGHPUT_MBPS}", result)
        self.assertIn(f"Iops={server.ROOT_VOLUME_IOPS}", result)

    def test_gp3_throughput_satisfies_the_iops_rule(self):
        """gp3 requires throughput <= IOPS * 0.25, enforced at run-instances
        time -- so violating it fails a LAUNCH, not just a disk."""
        self.assertLessEqual(
            server.ROOT_VOLUME_THROUGHPUT_MBPS, server.ROOT_VOLUME_IOPS * 0.25
        )

    def test_config_tags_with_rig_name(self):
        self.assertIn(
            f"Key=ManagedBy,Value={server.RIG_NAME}", run(server.get_deployment_config())
        )

    def test_config_spot_toggle(self):
        self.assertIn("MarketType=spot", run(server.get_deployment_config()))
        self.assertNotIn("MarketType=spot", run(server.get_deployment_config(spot=False)))

    def test_config_rejects_non_g4dn(self):
        for bad in ("g5g.2xlarge", "g6.xlarge"):
            self.assertTrue(
                run(server.get_deployment_config(instance_type=bad)).startswith("❌"), bad
            )
        self.assertFalse(
            run(server.get_deployment_config(instance_type="g4dn.xlarge")).startswith("❌")
        )


class RepoHygieneTests(unittest.TestCase):
    def test_shell_scripts_parse(self):
        for script in ("project-setup.sh", "init.sh", "set_env.sh", "save-aws-creds.sh"):
            proc = subprocess.run(
                ["bash", "-n", str(ROOT / script)], capture_output=True, text=True
            )
            self.assertEqual(proc.returncode, 0, f"{script}: {proc.stderr}")

    def test_tpu_env_agrees_with_server_defaults(self):
        # The directory name is a claim about tpu.env (NAMING.md). This asserts
        # the env file and the server actually agree, so the claim stays true.
        values = {}
        for line in (ROOT / "tpu.env").read_text().splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key] = value
        self.assertEqual(values["MODEL_NAME"], server.MODEL_NAME)
        self.assertEqual(values["INSTANCE_TYPE"], server.INSTANCE_TYPE)
        self.assertEqual(values["DTYPE"], server.DTYPE)
        self.assertEqual(values["KV_CACHE_DTYPE"], server.KV_CACHE_DTYPE)
        self.assertEqual(values["VLLM_IMAGE"], server.VLLM_IMAGE)
        self.assertEqual(values["VLLM_PATCHED_IMAGE"], server.VLLM_PATCHED_IMAGE)
        self.assertEqual(int(values["TURING_SMEM_BUDGET"]), server.TURING_SMEM_BUDGET)
        self.assertEqual(values["SERVICE_NAME"], server.SERVICE_NAME)
        # Both AMI keys, together: changing one without the other is the silent
        # revert described in DeploymentConfigTests.
        self.assertEqual(values["DLAMI_SSM_PARAMETER"], server.DLAMI_SSM_PARAMETER)
        self.assertEqual(values["DLAMI_NAME"], server.DLAMI_NAME)
        self.assertEqual(int(values["ROOT_VOLUME_GB"]), server.ROOT_VOLUME_GB)
        self.assertEqual(
            int(values["ROOT_VOLUME_THROUGHPUT_MBPS"]), server.ROOT_VOLUME_THROUGHPUT_MBPS
        )
        self.assertEqual(int(values["ROOT_VOLUME_IOPS"]), server.ROOT_VOLUME_IOPS)
        self.assertEqual(
            int(values["TENSOR_PARALLEL_SIZE"]), server._gpu_count(server.INSTANCE_TYPE)
        )

    def test_no_build_or_aarch64_key_survived_the_fork(self):
        text = (ROOT / "tpu.env").read_text()
        for key in ("TORCH_CUDA_ARCH_LIST=", "VLLM_REF=", "VLLM_STOCK_IMAGE=",
                    "CUDA_TOOLKIT_PACKAGE=", "VLLM_ATTENTION_BACKEND="):
            self.assertNotIn(f"\n{key}", text, f"{key} survived the fork as live config")

    def test_no_jax_module_survived_the_runtime_change(self):
        """This directory was a copy of `gpu-jax-g4dn-2b` before it was a vLLM
        rig. A leftover engine module would be imported by nothing and would
        describe a runtime this rig does not use."""
        for stale in ("jax_engine.py", "jax_openai_server.py", "tune_loop.py",
                      "profile_decode.py", "profile_prefill.py"):
            self.assertFalse((ROOT / stale).exists(), f"{stale} survived the runtime change")
        self.assertFalse((ROOT / "ports").exists(), "the vendored JAX model port survived")

    def test_rig_name_matches_directory(self):
        self.assertEqual(server.RIG_NAME, ROOT.name)

    def test_project_setup_derives_the_skill_stem(self):
        """A hardcoded stem is what silently survives a rename.

        On the JAX rig's fork this literal still named the OLD rig, so the script
        could not find the skill and died with `cannot locate the bundled skill`
        -- the rig was UNREGISTERABLE, not merely misregistered.
        """
        text = (ROOT / "project-setup.sh").read_text()
        self.assertNotIn('SKILL_STEM="gpu-', text)
        self.assertIn('SKILL_STEM="$(basename "$SCRIPT_DIR")-management"', text)

    def test_registration_files_agree_on_the_server_name(self):
        """Four places name this server, and a mismatch makes /mcp and the tool
        prefix disagree about what this rig is.

        Comments are stripped before checking for a sibling's name: these files
        deliberately EXPLAIN the fork, and prose about `g5g` or `g6` is the
        documentation working rather than a stale value.
        """
        for rel in (".mcp.json", ".claude-plugin/plugin.json", ".codex/config.toml",
                    ".claude/settings.local.json"):
            path = ROOT / rel
            if not path.is_file():  # .mcp.json and settings.local.json are gitignored
                continue
            text = path.read_text()
            self.assertIn(server.RIG_NAME, text, rel)
            live = "\n".join(
                line for line in text.splitlines() if not line.lstrip().startswith("#")
            )
            for sibling in ("g5g", "gpu-jax", "gpu-vllm-g6"):
                self.assertNotIn(sibling, live, f"{rel} still names {sibling} in live config")

    def test_benchmarks_carries_no_other_rigs_runs(self):
        """Benchmark JSON travelled with the forks in this tree, and several rigs
        carry numbers measured on hardware they are not. This rig has served
        nothing, so runs/ must be absent or empty."""
        runs = ROOT / "benchmarks" / "runs"
        self.assertTrue(not runs.exists() or not any(runs.iterdir()))

    def test_skill_is_complete_in_both_copies(self):
        # SKILL.md is a hand-written SOURCE file, but refresh_skill.py only
        # regenerates the mcp/ files beside it. So `rm -rf .claude/skills`
        # destroys it and `make skill` does not bring it back.
        stem = f"{ROOT.name}-management"
        for prefix in (f".claude/skills/{stem}", f"skills/{stem}"):
            skill = ROOT / prefix / "SKILL.md"
            self.assertTrue(skill.is_file(), f"{prefix}/SKILL.md is missing")
            self.assertIn(f"name: {stem}", skill.read_text(), f"{prefix}/SKILL.md stale name")
            for source in ("server.py", "patch_triton_turing.py", "project-setup.sh",
                           "requirements.txt"):
                self.assertTrue(
                    filecmp.cmp(ROOT / source, ROOT / prefix / "mcp" / source, shallow=False),
                    f"{prefix}/mcp/{source} is stale — run `make skill`",
                )


if __name__ == "__main__":
    unittest.main()
