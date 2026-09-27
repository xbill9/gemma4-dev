import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import score  # noqa: E402


def rec(probs_per_read, gold="a", entropy=0.0, names=("a", "b")):
    return {
        "names": list(names),
        "labels": ["A", "B"][: len(names)],
        "gold": gold,
        "reads": [
            {
                "probs": p,
                "entropy": entropy,
                "label_mass": 1.0,
                "argmax_is_label": True,
                "labels_returned": len(names),
            }
            for p in probs_per_read
        ],
        "latency": {"first_ms": 100.0, "extra_ms": 50.0},
    }


class Metrics(unittest.TestCase):
    def test_ece_perfect_and_overconfident(self):
        # 0.75 confidence, right 3 of 4 times: calibrated
        self.assertAlmostEqual(score.ece([0.75] * 4, [1, 1, 1, 0]), 0.0)
        # 1.0 confidence, right half the time: off by 0.5
        self.assertAlmostEqual(score.ece([1.0] * 4, [1, 1, 0, 0]), 0.5)

    def test_brier_and_nll(self):
        self.assertAlmostEqual(score.brier([[1.0, 0.0]], [0]), 0.0)
        self.assertAlmostEqual(score.brier([[0.5, 0.5]], [0]), 0.5)
        self.assertAlmostEqual(score.nll([[0.5, 0.5]], [1]), 0.6931471805599453)

    def test_auroc(self):
        self.assertEqual(score.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)
        self.assertEqual(score.auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0.0)
        self.assertEqual(score.auroc([0.5, 0.5], [1, 0]), 0.5)
        self.assertIsNone(score.auroc([0.5, 0.6], [1, 1]))

    def test_auto_readout_follows_threshold(self):
        confident = rec([[0.9, 0.1], [0.1, 0.9]], entropy=0.05)
        unsure = rec([[0.9, 0.1], [0.1, 0.9]], entropy=0.5)
        self.assertEqual(score.readout(confident, "dgauto")[0], [0.9, 0.1])
        self.assertEqual(score.readout(unsure, "dgauto")[0], [0.5, 0.5])

    def test_spread_is_proxy_stderr(self):
        r = rec([[0.8, 0.2], [0.6, 0.4]])
        # mean top prob 0.7, sample var 0.02, stderr sqrt(0.02 / 2) = 0.1
        self.assertAlmostEqual(score.spread(r), 0.1)
        self.assertIsNone(score.spread(rec([[0.8, 0.2]])))

    def test_score_counts_gate(self):
        recs = [
            rec([[0.9, 0.1]], gold="a", entropy=0.05),
            rec([[0.6, 0.4]], gold="b", entropy=0.5),
        ]
        s, *_ = score.score(recs, "ar")
        self.assertEqual(s["accuracy"], 0.5)
        self.assertEqual(s["gate_escalated"], 0.5)
        self.assertEqual(s["gate_acc_kept"], 1.0)
        self.assertEqual(s["gate_acc_escalated"], 0.0)

    def test_paired_diff_zero_for_identical(self):
        recs = [rec([[0.9, 0.1]]), rec([[0.2, 0.8]])]
        res = score.score(recs, "ar")
        pt, lo, hi = score.paired_diff(res, res, score.acc_stat)
        self.assertEqual((pt, lo, hi), (0.0, 0.0, 0.0))

    def test_temper_identity_and_sharpen(self):
        self.assertEqual([round(x, 6) for x in score.temper([0.7, 0.3], 1.0)], [0.7, 0.3])
        sharp = score.temper([0.7, 0.3], 0.5)
        self.assertGreater(sharp[0], 0.7)

    def test_fit_temperature_softens_overconfidence(self):
        # 0.99 confident but right only 60% of the time: the best fit is > 1
        probs = [[0.99, 0.01]] * 10
        gold = [0] * 6 + [1] * 4
        self.assertGreater(score.fit_temperature(probs, gold), 1.0)

    def test_flip_rate(self):
        base = {"a": rec([[0.9, 0.1]]), "b": rec([[0.2, 0.8]])}
        var = {"a": rec([[0.1, 0.9]], names=("b", "a")), "b": rec([[0.8, 0.2]], names=("b", "a"))}
        # a: base picks "a", variant picks "a" (index 1 of reversed names) -> same
        # b: base picks "b", variant picks "b" -> same
        self.assertEqual(score.flip_rate(base, var, "ar")["flips"], 0)
        # a variant that keeps the same letter while the names are reversed flips both
        flipped = {"a": rec([[0.9, 0.1]], names=("b", "a")), "b": rec([[0.2, 0.8]], names=("b", "a"))}
        self.assertEqual(score.flip_rate(base, flipped, "ar")["flips"], 2)


if __name__ == "__main__":
    unittest.main()
