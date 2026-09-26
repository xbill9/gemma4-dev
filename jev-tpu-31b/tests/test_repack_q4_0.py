import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import repack_q4_0 as rp  # noqa: E402


def on_grid(rng, rows, cols, peak=None):
    """bf16 bits of a weight whose every 32-wide group is step * q, q in [-8, 7].

    With `peak`, every group's largest level is exactly `peak`, the case the
    textbook `amax / 8` step gets wrong.
    """
    g = cols // rp.GROUP
    step = rng.uniform(1e-3, 3e-2, size=(rows, g, 1)).astype(np.float16).astype(np.float32)
    peak = peak or 8
    q = rng.integers(-peak, min(peak, 7) + 1, size=(rows, g, rp.GROUP))
    q[..., 0] = -peak  # Q4_0 puts a group's extreme at a negative level
    q[..., 1] = 1  # an odd level, so no coarser grid also fits
    return rp.f32_to_bf16((step * q).reshape(rows, cols)), q.reshape(rows, cols)


class Bits(unittest.TestCase):
    def test_bf16_round_trip(self):
        x = np.array([0.0, 1.0, -2.5, 3.140625, 1e-3], dtype=np.float32)
        u = rp.f32_to_bf16(x)
        np.testing.assert_array_equal(rp.f32_to_bf16(rp.bf16_to_f32(u)), u)
        self.assertEqual(rp.bf16_to_f32(u)[3], np.float32(3.140625))

    def test_bf16_rounds_to_nearest_even(self):
        # 1 + 2^-8 sits exactly between two bf16 values; even mantissa wins.
        self.assertEqual(rp.bf16_to_f32(rp.f32_to_bf16(np.float32([1 + 2 ** -8])))[0], 1.0)

    def test_pack_unpack_and_nibble_order(self):
        rng = np.random.default_rng(0)
        q = rng.integers(-8, 8, size=(5, 64)).astype(np.int8)
        np.testing.assert_array_equal(rp.unpack(rp.pack(q)), q)
        one = np.full((1, 8), -8, dtype=np.int8)
        one[0, 1] = -7  # column 1 -> nibble 1 holds 1
        self.assertEqual(rp.pack(one).view(np.uint32)[0, 0], 1 << 4)


class Grid(unittest.TestCase):
    def test_recovers_levels_when_peak_is_not_8(self):
        for peak in (3, 5, 7, 8):
            w, q_true = on_grid(np.random.default_rng(peak), 16, 128, peak=peak)
            q, scale, bad = rp.quantize(w)
            self.assertEqual(bad, 0, peak)
            np.testing.assert_array_equal(q, q_true, err_msg=f"peak {peak}")

    def test_textbook_step_would_regrid(self):
        # The trap the recovery exists for: amax / 8 on a peak-5 group moves values.
        w, _ = on_grid(np.random.default_rng(1), 4, 64, peak=5)
        g = rp.bf16_to_f32(w).reshape(4, 2, rp.GROUP)
        d = np.abs(g).max(axis=-1, keepdims=True) / 8
        self.assertGreater(np.abs(np.rint(g / d) * d - g).max(), 1e-4)

    def test_off_grid_tensor_is_not_quantized(self):
        rng = np.random.default_rng(2)
        w = rp.f32_to_bf16(rng.normal(size=(8, 64)).astype(np.float32))
        ents, bad = rp.quantized_entries("m", w)
        self.assertIsNone(ents)
        self.assertGreater(bad, 0)

    def test_entries_layout(self):
        w, _ = on_grid(np.random.default_rng(3), 12, 96)
        ents, bad = rp.quantized_entries("x.q_proj", w)
        self.assertEqual(bad, 0)
        got = {n: (t, a.shape, a.dtype) for n, t, a in ents}
        self.assertEqual(got["x.q_proj.weight_packed"], ("I32", (12, 12), np.int32))
        self.assertEqual(got["x.q_proj.weight_scale"], ("BF16", (12, 3), np.uint16))
        self.assertEqual(got["x.q_proj.weight_shape"][:2], ("I64", (2,)))


class EndToEnd(unittest.TestCase):
    def test_repack_then_verify(self):
        rng = np.random.default_rng(4)
        E, F, D = 3, 64, 96
        gate_up = np.stack([on_grid(rng, 2 * F, D)[0] for _ in range(E)])
        down = np.stack([on_grid(rng, D, F)[0] for _ in range(E)])
        L = "model.language_model.layers.0."
        src_tensors = [
            (L + "experts.gate_up_proj", "BF16", gate_up),
            (L + "experts.down_proj", "BF16", down),
            (L + "self_attn.q_proj.weight", "BF16", on_grid(rng, 64, D)[0]),
            (L + "mlp.up_proj.weight", "BF16", rp.f32_to_bf16(rng.normal(size=(64, D)).astype(np.float32))),
            (L + "router.proj.weight", "BF16", on_grid(rng, E, D)[0]),
            (L + "input_layernorm.weight", "BF16", rp.f32_to_bf16(np.ones(D, np.float32))),
            ("model.language_model.embed_tokens.weight", "BF16", on_grid(rng, 10, D)[0]),
        ]
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
            rp.write_safetensors(os.path.join(src, "model.safetensors"), src_tensors)
            json.dump({"architectures": ["Gemma4ForConditionalGeneration"]}, open(os.path.join(src, "config.json"), "w"))
            open(os.path.join(src, "tokenizer.json"), "w").write("{}")
            rp.repack(src, out, workers=1)
            self.assertEqual(rp.verify(src, out), 0)

            ck = rp.open_checkpoint(out)
            for i in range(E):
                for p in ("gate_proj", "up_proj", "down_proj"):
                    self.assertIn(f"{L}experts.{i}.{p}.weight_packed", ck)
            # gate is the first F rows of gate_up, up the rest.
            for p, rows in (("gate_proj", slice(0, F)), ("up_proj", slice(F, 2 * F))):
                name = f"{L}experts.1.{p}.weight_packed"
                q = rp.unpack(np.asarray(ck[name].raw(name)))
                np.testing.assert_array_equal(q, rp.quantize(gate_up[1, rows])[0])
            # The off-grid dense MLP stays bf16 and is ignored; router is ignored; embeddings copied.
            ign = json.load(open(os.path.join(out, "config.json")))["quantization_config"]["ignore"]
            self.assertIn(L + "mlp.up_proj", ign)
            self.assertIn(L + "router.proj", ign)
            self.assertIn(L + "mlp.up_proj.weight", ck)
            self.assertIn("model.language_model.embed_tokens.weight", ck)
            self.assertTrue(os.path.exists(os.path.join(out, "tokenizer.json")))
            report = json.load(open(os.path.join(out, "verify_report.json")))
            self.assertEqual(report["problems"], [])
            self.assertEqual(report["quantized"]["experts.gate_up_proj"]["groups_level_mismatch"], 0)


if __name__ == "__main__":
    unittest.main()
