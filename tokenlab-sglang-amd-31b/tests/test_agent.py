"""Offline tests for the ADK sample agent's tools and model wiring. No network."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from tokenlab_agent import agent
except ImportError as e:  # google-adk / litellm not installed
    agent = None
    SKIP = str(e)


@unittest.skipIf(agent is None, "google-adk not installed")
class CountTest(unittest.TestCase):
    IDS = [0, 2, 3, 20, 21, 22, 23, 10, 11, 12, 13]

    def test_ten_or_more_is_inclusive(self):
        r = agent.count_and_range(self.IDS, at_least=10)
        self.assertEqual((r["count"], r["min"], r["max"], r["filter"]), (8, 10, 23, "10 <= x"))

    def test_open_and_two_sided(self):
        self.assertEqual(agent.count_and_range(self.IDS)["count"], 11)
        self.assertEqual(agent.count_and_range(self.IDS, at_least=3, below=20)["count"], 5)
        self.assertIsNone(agent.count_and_range(self.IDS, at_least=100)["min"])

    def test_model_points_at_tokenlab(self):
        m = agent.root_agent.model
        self.assertEqual(m.model, "openai/Gemma-4-31B-IT")
        self.assertEqual(agent.root_agent.name, "tokenlab_agent")
        self.assertEqual({t.__name__ for t in agent.root_agent.tools}, {"list_tokenlab_models", "count_and_range"})


if __name__ == "__main__":
    unittest.main()
