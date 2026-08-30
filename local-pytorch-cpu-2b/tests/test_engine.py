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
                # expected is bound as a default: _with_capability calls body()
                # inside this iteration, so late binding never misfires here, but
                # ruff's B023 is a gate and the explicit bind costs nothing.
                self._with_capability(cc, lambda expected=expected: self.assertIs(
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


@unittest.skipUnless(HAVE_JAX, "jax is a serving dependency, not a control-plane one")
class PaddingWindowEvictionTests(unittest.TestCase):
    """Regression tests for docs/padding-window-eviction.md.

    Right-padding a prompt to a static bucket used to write pad K/V into the
    sliding layers' KV ring. Once ``pad_len >= sliding_window`` the ring held
    nothing but padding, every real token was evicted, and the model emitted a
    token loop that the server recorded as ``status="success"`` — measured
    2026-08-23 on a T4G.

    These run on CPU against the cache layout itself rather than against weights:
    what went wrong is *which positions occupy which ring slots*, and that is
    exactly computable. `W` and `S` stand in for sliding_window (512 on E2B) and
    a bucket; the arithmetic is scale-free.
    """

    W = 8       # stands in for config.sliding_window
    S = 32      # stands in for the padded bucket
    NEW = 4     # max_new_tokens

    def _ring_and_mask(self, real_len, fixed):
        """Return (ring occupants by position, attendable positions) at decode step 0."""
        # val[..., p, 0] = p, so a slot's contents name the position that wrote it.
        val = jnp.arange(self.S, dtype=jnp.float32).reshape(1, 1, self.S, 1)
        buf = jnp.zeros((1, 1, self.W, 1), dtype=jnp.float32)
        real = jnp.asarray([real_len], jnp.int32) if fixed else None
        ring = M._ring_store_one(buf, val, real)

        valid = jnp.concatenate(
            [jnp.arange(self.S)[None, :] < real_len,
             jnp.zeros((1, self.NEW), dtype=bool)], axis=1)
        # The whole bug in one expression: the old scheme decoded at bucket + t,
        # the fixed one at real_len + t.
        slot = (real_len if fixed else self.S)
        valid = valid.at[:, slot].set(True)
        mask = M.make_ring_decode_mask(valid, self.W, jnp.int32(slot))
        ok = mask[0, 0, 0] == 0.0
        occupants = [int(x) for x in ring[0, 0, :, 0]]
        attendable = sorted(occupants[j] for j in range(self.W) if bool(ok[j]))
        return occupants, attendable

    def test_padded_space_store_evicted_every_real_token(self):
        # Documents the defect the fix removes: with pad_len >= W the ring holds
        # only padding, so the only thing the model can attend to is the token it
        # just emitted — which is why the output was a loop rather than noise.
        for real_len in (self.S - self.W, self.S - self.W - 4):
            with self.subTest(real_len=real_len):
                occupants, attendable = self._ring_and_mask(real_len, fixed=False)
                self.assertTrue(all(p >= real_len for p in occupants),
                                f"ring should have held only padding, got {occupants}")
                self.assertEqual(len(attendable), 1)

    def test_real_position_store_keeps_the_window_full(self):
        # The fix: every ring slot holds a real position, for every padding.
        for real_len in range(self.W, self.S + 1):
            with self.subTest(real_len=real_len, pad=self.S - real_len):
                occupants, attendable = self._ring_and_mask(real_len, fixed=True)
                self.assertEqual(sorted(occupants), list(range(real_len - self.W, real_len)),
                                 "ring must hold exactly the last W real positions")
                self.assertEqual(attendable, sorted(occupants),
                                 "and every one of them must be attendable")

    def test_short_prompt_keeps_identity_layout(self):
        # A prompt shorter than the ring must still land at slot == position, and
        # the slots it never reached stay masked rather than reading zeros.
        occupants, _ = self._ring_and_mask(self.W - 3, fixed=True)
        self.assertEqual(occupants[: self.W - 3], list(range(self.W - 3)))
        self.assertEqual(occupants[self.W - 3 :], [0] * 3)   # untouched buffer

    def test_real_len_is_required_to_change_anything(self):
        # real_len=None must reproduce the old behaviour byte for byte: the chunked
        # prefill path passes no padding and relies on it.
        val = jnp.arange(self.S, dtype=jnp.float32).reshape(1, 1, self.S, 1)
        buf = jnp.zeros((1, 1, self.W, 1), dtype=jnp.float32)
        legacy = M._ring_store_one(buf, val, None)
        self.assertEqual([int(x) for x in legacy[0, 0, :, 0]],
                         list(range(self.S - self.W, self.S)))

    def test_batches_above_one_raise_rather_than_using_a_shared_slot(self):
        # The decode slot is now the row's real length, which is only common across
        # rows at B == 1. Silently falling back to the bucket would reinstate the bug.
        cfg = M.Gemma4EConfig()
        model = M.Gemma4EModelJAX(cfg)
        with self.assertRaises(NotImplementedError):
            M.generate_with_kv_cache(
                model, jnp.zeros((2, 8), jnp.int32), jnp.ones((2, 8), bool), {}, 1)


@unittest.skipUnless(HAVE_JAX, "jax is a serving dependency, not a control-plane one")
class BucketLadderTests(unittest.TestCase):
    """The ladder is defence in depth for the same bug — it bounds pad_len."""

    def test_padding_never_reaches_any_gemma4_sliding_window(self):
        # E2B declares sliding_window=512. A power-of-two ladder padded by up to
        # B/2 (2,047 on a 4,096 bucket), which is how the ring came to hold only
        # padding. Check every length, not a sample.
        worst = max(M.HardwareProfile.get_nearest_bucket(n) - n
                    for n in range(1, 16385))
        self.assertLess(worst, 128, f"worst-case padding is {worst} tokens")

    def test_ladder_is_sorted_and_covers_the_context_window(self):
        buckets = M.HardwareProfile.static_sequence_buckets
        self.assertEqual(list(buckets), sorted(buckets))
        self.assertEqual(len(set(buckets)), len(buckets))
        self.assertGreaterEqual(buckets[-1], 16384)

    def test_beyond_the_ladder_still_rounds_to_128(self):
        # The fallback past the last bucket already caps padding at 127; the ladder
        # now agrees with it instead of contradicting it.
        n = M.HardwareProfile.static_sequence_buckets[-1] + 300
        self.assertLess(M.HardwareProfile.get_nearest_bucket(n) - n, 128)


@unittest.skipUnless(HAVE_JAX, "jax is a serving dependency, not a control-plane one")
class PaddingInvarianceEndToEndTests(unittest.TestCase):
    """The property the bug violated: output must not depend on bucket padding.

    A four-layer random model on CPU, small enough to generate in about a second
    and large enough to reproduce the real failure — three sliding layers with a
    window of 8, one full-attention layer, exactly the structure that made E2B
    loop rather than produce noise.

    Run against the pre-fix port this test does not merely differ, it reproduces
    the reported signature: every padding at or above the window returns the SAME
    degenerate sequence as every other, with a token repeated four times in a row,
    while paddings below the window each diverge differently. That is the whole
    bug in one assertion, and it is why this is an end-to-end test rather than
    another check of the slot arithmetic.
    """

    WINDOW = 8
    REAL = 20

    @classmethod
    def setUpClass(cls):
        import jax
        cls.jax = jax
        cls.cfg = M.Gemma4EConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            num_global_key_value_heads=1, global_head_dim=8, num_kv_shared_layers=0,
            sliding_window=cls.WINDOW, hidden_size_per_layer_input=0,
            use_double_wide_mlp=False, logit_softcapping=0.0,
            layer_types=["sliding_attention"] * 3 + ["full_attention"],
        )
        cls.model = M.Gemma4EModelJAX(cls.cfg)
        cls.params = cls._build_params(jax.random.PRNGKey(0))
        cls.prompt = jax.random.randint(
            jax.random.PRNGKey(1), (1, cls.REAL), 0, cls.cfg.vocab_size)

    @classmethod
    def _build_params(cls, key):
        jax, cfg = cls.jax, cls.cfg
        keys = iter(jax.random.split(key, 64))
        rnd = lambda *shape: jax.random.normal(  # noqa: E731
            next(keys), shape, dtype=jnp.float32) * 0.05
        ones = jnp.ones((cfg.hidden_size,))
        params = {"embed_tokens": rnd(cfg.vocab_size, cfg.hidden_size), "final_norm": ones}
        for i in range(cfg.num_hidden_layers):
            sliding = cfg.layer_types[i] == "sliding_attention"
            hd = cfg.head_dim if sliding else cfg.global_head_dim
            nkv = cfg.num_key_value_heads if sliding else cfg.num_global_key_value_heads
            H, nh = cfg.hidden_size, cfg.num_attention_heads
            params[f"layer_{i}"] = {
                "attn": {"q_proj": rnd(H, nh * hd), "k_proj": rnd(H, nkv * hd),
                         "v_proj": rnd(H, nkv * hd), "o_proj": rnd(nh * hd, H)},
                "mlp": {"gate_proj": rnd(H, cfg.intermediate_size),
                        "up_proj": rnd(H, cfg.intermediate_size),
                        "down_proj": rnd(cfg.intermediate_size, H)},
                "input_layernorm": ones, "post_attention_layernorm": ones,
                "pre_feedforward_layernorm": ones, "post_feedforward_layernorm": ones,
            }
        return params

    def _generate(self, bucket, n=8):
        pad = bucket - self.REAL
        ids = jnp.pad(self.prompt, ((0, 0), (0, pad)))
        valid = jnp.concatenate(
            [jnp.ones((1, self.REAL), bool), jnp.zeros((1, pad), bool)], axis=1)
        out = M.generate_with_kv_cache(self.model, ids, valid, self.params, n,
                                       quant_mode="fp16", window_kv=True)
        return [int(x) for x in out[0]]

    def test_output_is_identical_however_much_the_prompt_is_padded(self):
        baseline = self._generate(self.REAL)                 # pad = 0
        # Straddle the window: below it (4), exactly at it (8), and far past it
        # (28) — the regime where the ring used to hold nothing but padding.
        for bucket in (self.REAL + 4, self.REAL + self.WINDOW, self.REAL + 28):
            with self.subTest(pad=bucket - self.REAL):
                self.assertEqual(self._generate(bucket), baseline)


if __name__ == "__main__":
    unittest.main()
