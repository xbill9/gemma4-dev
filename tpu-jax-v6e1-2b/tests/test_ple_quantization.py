"""The PLE table is 72% of E2B's resident weights, and QAT left it in BF16.

`embed_tokens_per_layer` is [vocab, layers*D_ple] — 4.70 GB of the 6.56 GB the
engine loads. The shipped W4A16 checkpoint compressed the 1.06 GB of transformer
weights and none of the 5.50 GB of lookup tables, so this is both the largest
remaining target and the lowest-risk one: the table is read by a gather, never a
matmul, so error cannot compound through a contraction.

Quantizing it buys HBM headroom, which on a v6e-1 converts directly into resident
KV tokens and therefore concurrent sequences.

Run: python3 -m unittest tests.test_ple_quantization
"""

import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ports.gemma4.jax_e_model import gather_ple, quantize_ple_table  # noqa: E402

VOCAB, LAYERS, D_PLE = 256, 35, 256
WIDTH = LAYERS * D_PLE


def table(seed: int = 0, per_slice_variation: bool = False):
    t = jax.random.normal(jax.random.PRNGKey(seed), (VOCAB, WIDTH),
                          dtype=jnp.bfloat16) * 2.0
    if per_slice_variation:
        # Real PLE slices do not share a magnitude across layers, and that is
        # exactly what a per-row scale cannot represent. Vary each layer slice
        # over two orders of magnitude to model it.
        mags = jnp.logspace(-1, 1, LAYERS, dtype=jnp.float32).reshape(1, LAYERS, 1)
        t = (t.reshape(VOCAB, LAYERS, D_PLE).astype(jnp.float32) * mags)
        t = t.reshape(VOCAB, WIDTH).astype(jnp.bfloat16)
    return t


def nbytes(params):
    return sum(a.size * a.dtype.itemsize for a in params.values()
               if hasattr(a, "size"))


class SizeTest(unittest.TestCase):
    """The point of the exercise is bytes; assert them."""

    def setUp(self):
        self.params = {"embed_tokens_per_layer": table()}
        self.base = nbytes(self.params)

    def test_int8_halves_the_table(self):
        q = quantize_ple_table(self.params, bits=8, group_size=D_PLE)
        self.assertLess(nbytes(q) / self.base, 0.52)
        self.assertGreater(nbytes(q) / self.base, 0.48)

    def test_int4_quarters_the_table(self):
        q = quantize_ple_table(self.params, bits=4, group_size=D_PLE)
        self.assertLess(nbytes(q) / self.base, 0.27)
        self.assertGreater(nbytes(q) / self.base, 0.24)

    def test_bf16_table_is_dropped(self):
        for bits in (4, 8):
            q = quantize_ple_table(self.params, bits=bits, group_size=D_PLE)
            self.assertNotIn("embed_tokens_per_layer", q,
                             "the BF16 table must not stay resident")

    def test_scale_overhead_is_small(self):
        """Per-layer-slice scales must not eat the saving they enable."""
        q = quantize_ple_table(self.params, bits=4, group_size=D_PLE)
        scale_bytes = q["embed_tokens_per_layer_scale"].size * 2
        packed_bytes = q["embed_tokens_per_layer_q4"].size
        self.assertLess(scale_bytes / packed_bytes, 0.05,
                        "scale overhead should be a couple of percent")


class AccuracyTest(unittest.TestCase):
    def setUp(self):
        self.params = {"embed_tokens_per_layer": table()}
        self.ids = jnp.array([[3, 17, 200, 255]], dtype=jnp.int32)
        self.ref = gather_ple(self.params, self.ids)

    def _rel_err(self, **kw):
        got = gather_ple(quantize_ple_table(self.params, **kw), self.ids)
        self.assertEqual(got.shape, self.ref.shape)
        denom = float(jnp.max(jnp.abs(self.ref.astype(jnp.float32))))
        return float(jnp.max(jnp.abs((got - self.ref).astype(jnp.float32)))) / denom

    def test_int8_is_near_lossless(self):
        self.assertLess(self._rel_err(bits=8, group_size=D_PLE), 0.02)

    def test_int4_is_bounded(self):
        # 4 bits is 16 levels per group; ~7% of the row max is expected and is
        # the price of the 3.5 GB. Judge acceptability on the real checkpoint,
        # not here.
        self.assertLess(self._rel_err(bits=4, group_size=D_PLE), 0.15)

    def test_int4_grouping_beats_a_single_row_scale(self):
        """Grouping is what makes 4 bits usable when slices differ in magnitude.

        Measured PER SLICE, deliberately. A global relative error is dominated by
        the largest-magnitude slice — the one slice a row-wide scale already fits
        — so it reports the two schemes as identical. What a row scale destroys is
        the SMALL slices, which it quantizes to nearly zero, and only a per-slice
        metric can see that.
        """
        params = {"embed_tokens_per_layer": table(per_slice_variation=True)}
        ref = gather_ple(params, self.ids).astype(jnp.float32)

        def per_slice_err(**kw):
            got = gather_ple(quantize_ple_table(params, **kw), self.ids).astype(jnp.float32)
            r = ref.reshape(*ref.shape[:-1], LAYERS, D_PLE)
            g = got.reshape(*got.shape[:-1], LAYERS, D_PLE)
            denom = jnp.maximum(jnp.max(jnp.abs(r), axis=-1), 1e-6)
            return float(jnp.mean(jnp.max(jnp.abs(g - r), axis=-1) / denom))

        grouped = per_slice_err(bits=4, group_size=D_PLE)
        row_wide = per_slice_err(bits=4, group_size=0)
        self.assertLess(grouped, row_wide * 0.5,
                        f"slice scales should clearly beat a row scale "
                        f"(grouped {grouped:.4f} vs row {row_wide:.4f})")

    def test_accuracy_ordering(self):
        self.assertLess(self._rel_err(bits=8, group_size=D_PLE),
                        self._rel_err(bits=4, group_size=D_PLE))


class GatherTest(unittest.TestCase):
    def setUp(self):
        self.params = {"embed_tokens_per_layer": table()}

    def test_unquantized_path_is_untouched(self):
        ids = jnp.array([[1, 2]], dtype=jnp.int32)
        out = gather_ple(self.params, ids)
        self.assertTrue(bool(jnp.array_equal(
            out, self.params["embed_tokens_per_layer"][ids])))

    def test_nibble_unpack_preserves_element_order(self):
        """A packed byte holds elements 2i and 2i+1; swapping them is silent."""
        q = quantize_ple_table(self.params, bits=4, group_size=D_PLE)
        ids = jnp.array([[7]], dtype=jnp.int32)
        got = gather_ple(q, ids)[0, 0]
        ref = self.params["embed_tokens_per_layer"][7]
        # Correlate against the reference: an order swap would decorrelate
        # adjacent pairs while leaving the magnitude distribution intact.
        g = got.astype(jnp.float32)
        r = ref.astype(jnp.float32)
        corr = float(jnp.corrcoef(g, r)[0, 1])
        self.assertGreater(corr, 0.98, f"element order looks scrambled (r={corr:.3f})")

    def test_batch_and_sequence_dims_survive(self):
        q = quantize_ple_table(self.params, bits=4, group_size=D_PLE)
        ids = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
        self.assertEqual(gather_ple(q, ids).shape, (2, 3, WIDTH))

    def test_jits(self):
        q = quantize_ple_table(self.params, bits=4, group_size=D_PLE)
        ids = jnp.array([[1, 2]], dtype=jnp.int32)
        # group size is a Python int in the params dict; it must not become a
        # traced value or the reshape will fail under jit.
        fn = jax.jit(lambda p, i: gather_ple(p, i), static_argnums=())
        out = jax.block_until_ready(fn(q, ids))
        self.assertEqual(out.shape, (1, 2, WIDTH))



class ResidencyTest(unittest.TestCase):
    """The quantized table must live where the original lived.

    Quantizing on the host is correct; LEAVING the result on the host is not.
    It fails silently and in the flattering direction — HBM capacity measurements
    improve, because the table is no longer in HBM — while every gather crosses
    the host interconnect. Observed at 18.5 s per decode step against 60 ms
    resident, with no error raised anywhere.
    """

    def test_output_shares_the_input_device(self):
        src = table()
        want = next(iter(src.devices()))
        params = {"embed_tokens_per_layer": src}
        for bits, key in ((8, "embed_tokens_per_layer_q8"),
                          (4, "embed_tokens_per_layer_q4")):
            q = quantize_ple_table(params, bits=bits, group_size=D_PLE)
            for name in (key, "embed_tokens_per_layer_scale"):
                got = next(iter(q[name].devices()))
                self.assertEqual(got, want,
                                 f"{name} landed on {got}, expected {want}")

if __name__ == "__main__":
    unittest.main()
