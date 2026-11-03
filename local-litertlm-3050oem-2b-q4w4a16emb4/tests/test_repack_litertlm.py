import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import repack_litertlm as r  # noqa: E402


class PackingTest(unittest.TestCase):
    def test_tflite_round_trip_covers_minus_eight(self):
        codes = np.arange(-8, 8, dtype=np.int8).repeat(4).reshape(2, 32)
        packed = r.pack_tfl(codes)
        self.assertEqual(packed.size, codes.size // 2)
        np.testing.assert_array_equal(r.unpack_tfl(packed, codes.shape), codes)

    def test_tflite_low_nibble_first(self):
        packed = r.pack_tfl(np.array([[1, -1]], np.int8))
        self.assertEqual(int(packed[0]), 0xF1)

    def test_dequant_repeats_scale_per_block(self):
        codes = np.ones((1, 64), np.int8)
        scales = np.array([[0.5, 2.0]], np.float16)
        d = r.dequant(codes, scales)
        self.assertTrue(np.all(d[0, :32] == 0.5) and np.all(d[0, 32:] == 2.0))


class RepackUnpackTest(unittest.TestCase):
    def test_compressed_tensors_nibbles_offset_by_eight(self):
        # One int32 packs 8 codes, nibble i at bits 4i, stored as code + 8.
        codes = np.array([-8, -1, 0, 1, 2, 3, 6, 7], np.int64)
        word = sum(int(c + 8) << (4 * i) for i, c in enumerate(codes))
        packed = np.array([[word]], np.uint32).view(np.int32)
        rp = r.Repack.__new__(r.Repack)
        rp.fused, rp._cache = {}, {}
        rp._get = lambda name: packed if name.endswith("_packed") else np.ones((1, 1), np.float32)
        got, _ = rp.codes_scales("w.weight")
        np.testing.assert_array_equal(got[0], codes.astype(np.int8))


if __name__ == "__main__":
    unittest.main()
