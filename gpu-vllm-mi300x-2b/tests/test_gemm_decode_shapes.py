"""Offline tests for gemm_decode_shapes.py — the parts that need no torch and no GPU."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gemm_decode_shapes as g  # noqa: E402


def _row(shape, m, dtype, us, status="ok"):
    return {"shape": shape, "m": m, "dtype": dtype, "status": status, "us_per_call": us}


class ShapeTableTest(unittest.TestCase):
    def test_streamed_params_match_models_md(self):
        # ../MODELS.md: transformer matmuls 1.854 B + tied LM head 0.403 B streamed per token.
        # The fused table counts k/v on every layer, as the checkpoint does; allow 2%.
        streamed = g.summarize([])["streamed_params_per_token"]
        self.assertAlmostEqual(streamed / (1.854e9 + 0.403e9), 1.0, delta=0.02)

    def test_layer_counts_sum_to_35(self):
        counts = {name: c for name, _, _, c in g.E2B_SHAPES}
        self.assertEqual(counts["qkv_sliding"] + counts["qkv_full"], 35)
        self.assertEqual(counts["o_sliding"] + counts["o_full"], 35)
        self.assertEqual(counts["gate_up"] + counts["gate_up_wide"], 35)
        self.assertEqual(counts["down"] + counts["down_wide"], 35)
        self.assertEqual(counts["lm_head"], 1)


class ArithmeticTest(unittest.TestCase):
    def test_copies_exceed_infinity_cache(self):
        one = g.weight_bytes(2560, 1536, "bf16")
        n = g.copies_needed(one, 1 << 30)
        self.assertGreaterEqual(n * one, 1 << 30)
        self.assertGreater(n * one, 256 << 20)

    def test_hot_mode_is_one_copy(self):
        self.assertEqual(g.copies_needed(123, 0), 1)

    def test_big_weight_still_rotates(self):
        self.assertEqual(g.copies_needed(g.weight_bytes(262144, 1536, "bf16"), 1 << 30), 2)

    def test_metrics(self):
        m = g.cell_metrics(1, 1000, 1000, "bf16", 1e-6)
        self.assertAlmostEqual(m["weight_gbps"], 2000.0)
        self.assertAlmostEqual(m["tflops"], 2.0)
        self.assertAlmostEqual(m["us_per_call"], 1.0)

    def test_graph_is_the_default_mode(self):
        # Eager timing is dispatch-bound at decode shapes; see the module docstring.
        self.assertEqual(g.parse_args([]).mode, "graph")

    def test_graph_calls_bounded_by_output_memory(self):
        # lm_head at M=64 writes 32 MiB per call; 500 captured calls would hold 16 GiB.
        self.assertEqual(g.graph_calls(500, 64 * 262144 * 2), 64)
        self.assertEqual(g.graph_calls(2000, 2 * 2560), 2000)
        self.assertEqual(g.graph_calls(20, 8192 * 8192 * 2), 16)
        self.assertEqual(g.graph_calls(5, 1), 10)

    def test_plan_includes_control_at_8192(self):
        cells = g.plan([1, 8], include_control=True)
        self.assertIn(("control_8192", 8192, 8192, 8192), cells)
        self.assertIn(("control_8192", 8192, 8192, 1), cells)
        self.assertEqual(len(cells), 3 + len(g.E2B_SHAPES) * 2)
        self.assertNotIn("control_8192", {c[0] for c in g.plan([1], include_control=False)})


class SummarizeTest(unittest.TestCase):
    def _full(self, m, dtype, us):
        return [_row(name, m, dtype, us) for name, _, _, _ in g.E2B_SHAPES]

    def test_speedup_direction(self):
        rows = [_row("qkv_full", 1, "bf16", 10.0), _row("qkv_full", 1, "fp8", 5.0)]
        s = {(x["dtype"]): x["speedup_vs_bf16"] for x in g.summarize(rows)["speedups"]}
        self.assertAlmostEqual(s["fp8"], 2.0)
        self.assertAlmostEqual(s["bf16"], 1.0)

    def test_per_token_weights_by_count(self):
        rows = self._full(1, "bf16", 1.0) + self._full(1, "fp8", 0.5)
        pt = {p["dtype"]: p for p in g.summarize(rows)["per_token"] if p["m"] == 1}
        total_count = sum(c for _, _, _, c in g.E2B_SHAPES)
        self.assertAlmostEqual(pt["bf16"]["us_per_token"], total_count)
        self.assertAlmostEqual(pt["fp8"]["speedup_vs_bf16"], 2.0)

    def test_partial_dtype_gets_no_sum(self):
        rows = self._full(1, "bf16", 1.0) + self._full(1, "int8", 0.5)
        rows[-1]["status"] = "unsupported"
        pt = {p["dtype"]: p for p in g.summarize(rows)["per_token"] if p["m"] == 1}
        self.assertTrue(pt["int8"]["missing"])
        self.assertIsNone(pt["int8"]["us_per_token"])
        self.assertIsNone(pt["int8"]["speedup_vs_bf16"])

    def test_render_marks_unsupported(self):
        rows = self._full(1, "bf16", 1.0)
        rows.append(
            {"shape": "lm_head", "n": 262144, "k": 1536, "m": 1, "dtype": "int8", "status": "unsupported", "error": "E"}
        )
        for r in rows:
            r.update(n=r.get("n", 1), k=r.get("k", 1), tflops=0.0, weight_gbps=0.0, pct_measured_bw=0.0, cv_pct=0.0)
        md = g.render(
            {
                "device": "d",
                "arch": "a",
                "torch": "t",
                "rotate_bytes": 0,
                "repeats": 1,
                "rows": rows,
                "summary": g.summarize(rows),
            }
        )
        self.assertIn("Unsupported cells", md)
        self.assertIn("lm_head M=1 int8", md)


if __name__ == "__main__":
    unittest.main()
