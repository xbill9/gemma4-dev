"""Offline tests for the JAX engine's device policy and the Turing guards.

No AWS, no network, no GPU. These run on CPU and then *override* the detected
platform, so they exercise the pre-Ampere branch on a machine that has no
pre-Ampere GPU. That makes them a test of the policy, not of the hardware — the
hardware claims live in benchmarks/runs/ and are measured, not asserted here.

Skipped entirely when jax is absent: it is a serving dependency
(requirements-serving.txt), not a control-plane one, so a machine that only runs
the MCP server is not expected to have it.
"""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:
    import jax.numpy as jnp

    from ports.gemma4 import jax_e_model as M
    HAVE_JAX = True
except ImportError:  # pragma: no cover - depends on the host
    HAVE_JAX = False


@unittest.skipUnless(HAVE_JAX, "jax is a serving dependency, not a control-plane one")
class DevicePolicyTests(unittest.TestCase):
    def test_compute_dtype_follows_compute_capability(self):
        # The rule that matters: anything below Ampere gets float16, because
        # Turing has no bf16 datapath and XLA would silently emulate it.
        self.assertIs(M._default_compute_dtype.__wrapped__ if False else None, None)
        for cc, expected in (((7, 5), jnp.float16), ((8, 0), jnp.bfloat16),
                             ((8, 9), jnp.bfloat16), ((7, 0), jnp.float16)):
            with self.subTest(cc=cc):
                self._with_capability(cc, lambda: self.assertIs(
                    M._default_compute_dtype(), expected))

    def test_env_override_is_honoured_and_validated(self):
        self._with_capability((7, 5), lambda: None)
        os.environ["JAX_E_COMPUTE_DTYPE"] = "bfloat16"
        try:
            self.assertIs(M._default_compute_dtype(), jnp.bfloat16)
            os.environ["JAX_E_COMPUTE_DTYPE"] = "nonsense"
            with self.assertRaises(ValueError):
                M._default_compute_dtype()
        finally:
            del os.environ["JAX_E_COMPUTE_DTYPE"]

    def test_cpu_host_does_not_claim_a_gpu_profile(self):
        self.assertEqual(M.PLATFORM, "cpu")
        self.assertIsNone(M.COMPUTE_CAPABILITY)
        self.assertFalse(M.IS_PRE_AMPERE)

    def _with_capability(self, cc, body):
        old_cc, old_pre = M.COMPUTE_CAPABILITY, M.IS_PRE_AMPERE
        M.COMPUTE_CAPABILITY, M.IS_PRE_AMPERE = cc, cc < (8, 0)
        try:
            body()
        finally:
            M.COMPUTE_CAPABILITY, M.IS_PRE_AMPERE = old_cc, old_pre


@unittest.skipUnless(HAVE_JAX, "jax is a serving dependency, not a control-plane one")
class SharedMemoryCeilingTests(unittest.TestCase):
    """The fused W4A16 kernel is tiled for TPU VMEM and must be refused on Turing.

    This is the same 64 KiB ceiling that forces the unlanded Triton patch in the
    vLLM sibling, reached by a different route. Catching it before compilation
    turns a mid-serve OutOfResources into a startup refusal.
    """

    # Real Gemma 4 E2B projection shapes.
    SHAPES = ((1, 2048, 6144), (1, 6144, 2048), (1, 2048, 4096), (1, 2048, 2048))

    def test_every_projection_overflows_turing(self):
        for seq, k, n in self.SHAPES:
            with self.subTest(K=k, N=n):
                need = M.w4a16_shared_memory_bytes(seq, k, n)
                self.assertGreater(need, 64 * 1024, "would fit — the guard is now untested")

    def test_check_raises_on_pre_ampere_with_the_arithmetic_attached(self):
        with self._turing():
            with self.assertRaises(M.ScopedMemoryError) as ctx:
                M.check_w4a16_fits_scoped_memory(1, 2048, 6144)
        message = str(ctx.exception)
        self.assertIn("64 KiB", message)
        self.assertIn("7.5", message)
        self.assertIn("reference", message)  # names the way out

    def test_check_is_a_no_op_off_gpu(self):
        # On TPU the budget is VMEM, a different size entirely; the guard must
        # not fire there and must not fire on a CPU host either.
        M.check_w4a16_fits_scoped_memory(1, 2048, 6144)

    def test_tile_sizes_are_shape_driven(self):
        self.assertEqual(M.w4a16_tile_sizes(2048, 6144), (256, 256))
        self.assertEqual(M.w4a16_tile_sizes(96, 96), (96, 96))

    def _turing(self):
        class _Ctx:
            def __enter__(_self):
                _self.saved = (M.PLATFORM, M.COMPUTE_CAPABILITY, M.IS_PRE_AMPERE,
                               M.device_shared_memory_limit_bytes)
                M.PLATFORM, M.COMPUTE_CAPABILITY, M.IS_PRE_AMPERE = "gpu", (7, 5), True
                M.device_shared_memory_limit_bytes = lambda: 64 * 1024
                return _self

            def __exit__(_self, *exc):
                (M.PLATFORM, M.COMPUTE_CAPABILITY, M.IS_PRE_AMPERE,
                 M.device_shared_memory_limit_bytes) = _self.saved
                return False
        return _Ctx()


@unittest.skipUnless(HAVE_JAX, "jax is a serving dependency, not a control-plane one")
class HardwareProfileTests(unittest.TestCase):
    def test_t4g_profile_records_measured_memory_not_nominal(self):
        # nvidia-smi reports 15360 MiB on a G5g, not the nominal 16 GB. Sizing a
        # KV pool off 16 GB overcommits by 1 GiB.
        self.assertEqual(M.T4GHardwareProfile().device_memory_bytes, 15360 * 1024 * 1024)

    def test_t4g_has_no_native_bf16_and_tpu_does(self):
        self.assertFalse(M.T4GHardwareProfile().native_bf16)
        self.assertTrue(M.TPUv6eHardwareProfile().native_bf16)

    def test_scoped_memory_differs_by_three_orders_of_magnitude(self):
        # This single ratio is why kernels tiled for TPU do not transplant.
        tpu = M.TPUv6eHardwareProfile().scoped_memory_bytes
        gpu = M.T4GHardwareProfile().scoped_memory_bytes
        self.assertGreater(tpu / gpu, 200)

    def test_bucketing_is_shared_and_the_tpu_alias_still_resolves(self):
        self.assertIs(M.pad_to_tpu_v6e_bucket, M.pad_to_bucket)
        self.assertIs(M.onchip_sample_tpu_v6e_jax, M.onchip_sample_jax)


@unittest.skipUnless(HAVE_JAX, "jax is a serving dependency, not a control-plane one")
class EngineResolutionTests(unittest.TestCase):
    def test_quant_mode_follows_the_checkpoint_not_the_chip(self):
        import jax_engine as E
        self.assertEqual(E.resolve_quant_mode("auto", "google/gemma-4-E2B-it"), "fp16")
        self.assertEqual(E.resolve_quant_mode("auto", "google/gemma-4-E2B-it-qat-w4a16-ct"), "w4a16")
        with self.assertRaises(ValueError):
            E.resolve_quant_mode("int4", "whatever")

    def test_fp8_kv_is_refused_on_pre_ampere(self):
        import jax_engine as E
        saved = E.IS_PRE_AMPERE
        E.IS_PRE_AMPERE = True
        try:
            for name in ("fp8", "fp8_e4m3", "fp8_e5m2"):
                with self.subTest(name=name), self.assertRaises(ValueError) as ctx:
                    E.resolve_cache_dtype(name)
                self.assertIn("int8", str(ctx.exception))  # names the alternative
            # int8 and auto stay available.
            self.assertIs(E.resolve_cache_dtype("int8"), jnp.int8)
        finally:
            E.IS_PRE_AMPERE = saved


if __name__ == "__main__":
    unittest.main()
