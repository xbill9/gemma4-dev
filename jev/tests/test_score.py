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


if __name__ == "__main__":
    unittest.main()
