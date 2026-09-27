"""score_suite.py's ECE and Brier against Bespoke Labs' own implementation.

Skipped unless a Nimble checkout is available (NIMBLE_DIR, default /tmp/nimble).
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nimble_suite"))
import score_suite  # noqa: E402

NIMBLE = os.environ.get("NIMBLE_DIR", "/tmp/nimble")


@unittest.skipUnless(os.path.isdir(os.path.join(NIMBLE, "nimble")), "no Nimble checkout")
class AgainstNimble(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, NIMBLE)
        from nimble.evaluation import evaluate_pilot, evaluate_public

        cls.assess = staticmethod(evaluate_pilot.assess)
        cls.ece = staticmethod(evaluate_public.expected_calibration_error)

    def test_ece_matches(self):
        rng = random.Random(1)
        for _ in range(50):
            pairs = [(rng.random(), rng.random() < 0.6) for _ in range(rng.randint(5, 300))]
            pairs += [(1.0, True), (0.1, False)]
            self.assertAlmostEqual(score_suite.ece10(pairs), self.ece(pairs), places=12)

    def test_assess_matches(self):
        rng = random.Random(2)
        for kind, keys, target in (("noul", ["true", "false"], True), ("choice", ["a", "b", "c"], "b"), ("score", ["0", "1", "2", "3"], 2)):
            for _ in range(50):
                w = [rng.random() for _ in keys]
                p = {k: x / sum(w) for k, x in zip(keys, w)}
                theirs = self.assess(p, target, kind)
                rec = {"type": kind, "target": target}
                ours = score_suite.assess(p, score_suite.gold_key(rec))
                self.assertEqual(ours["correct"], theirs["correct"])
                self.assertAlmostEqual(ours["top"], theirs["top_probability"], places=12)
                self.assertAlmostEqual(ours["brier"], theirs["multiclass_brier"], places=12)
                self.assertAlmostEqual(ours["nll"], theirs["negative_log_likelihood"], places=12)


if __name__ == "__main__":
    unittest.main()
