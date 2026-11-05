"""The patched int4 embedding (patches/int4-embeddings.diff) against the compressed-tensors packing.

Packing as local-vllm-3050oem-2b-q4w4a16emb4/repack/repack_q4_0.py writes it: nibble i of int32 word j
holds q + 8 for column 8j + i, one fp16 scale per 32 columns. Skipped where SGLang or the patch is absent.
"""

import unittest

import numpy as np

try:
    import torch
    from sglang.srt.models.gemma4_causal import Gemma4Int4ScaledEmbedding
except Exception:  # noqa: BLE001 - any import failure means the patched module is not available
    Gemma4Int4ScaledEmbedding = None


def pack(q):
    out, inn = q.shape
    u = (q.astype(np.int32) + 8).astype(np.uint32).reshape(out, inn // 8, 8)
    word = np.zeros((out, inn // 8), dtype=np.uint32)
    for i in range(8):
        word |= u[:, :, i] << np.uint32(4 * i)
    return word.view(np.int32)


@unittest.skipIf(Gemma4Int4ScaledEmbedding is None, "patched sglang not installed")
class Int4EmbeddingTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.vocab, self.dim, self.scale = 64, 256, 16.0
        self.q = rng.integers(-8, 8, size=(self.vocab, self.dim))
        self.s = rng.uniform(0.001, 0.05, size=(self.vocab, self.dim // 32)).astype(np.float16)
        m = Gemma4Int4ScaledEmbedding(self.vocab, self.dim, embed_scale=self.scale)
        m.weight_packed.data.copy_(torch.from_numpy(pack(self.q)))
        m.weight_scale.data.copy_(torch.from_numpy(self.s))
        self.m = m
        self.ref = self.q.reshape(self.vocab, -1, 32) * self.s.astype(np.float32)[:, :, None]
        self.ref = self.ref.reshape(self.vocab, self.dim)

    def test_rows(self):
        ids = torch.tensor([[0, 5, 63], [7, 7, 1]])
        got = self.m(ids).float().numpy()
        want = torch.from_numpy(self.ref[ids.numpy()]).to(torch.bfloat16).float().numpy() * self.scale
        self.assertEqual(got.shape, (2, 3, self.dim))
        np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-6)

    def test_dense_weight(self):
        got = self.m.weight.float().numpy()
        want = torch.from_numpy(self.ref).to(torch.bfloat16).float().numpy()
        np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
    unittest.main()
