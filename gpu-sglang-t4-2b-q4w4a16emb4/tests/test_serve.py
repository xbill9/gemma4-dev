import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import serve  # noqa: E402



class TestServe(unittest.TestCase):
    def env(self, **over):
        with mock.patch.dict(os.environ, over, clear=False):
            return serve.load_env()

    def test_int4_checkpoint_in_float16_on_turing(self):
        # the 3050 rig's W4A16 checkpoint; SGLang builds the model in float16 below sm_80
        env = serve.load_env()
        self.assertEqual(env["MODEL_NAME"], "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4")
        self.assertEqual(env["MODEL_REVISION"], "db715f83da4f13289761d7c45f4d19ceac84695d")
        cmd = serve.command(env)
        self.assertEqual(cmd[cmd.index("--dtype") + 1], "float16")

    def test_no_prefix_cache_no_drafter_by_default(self):
        cmd = serve.command(self.env(SPECULATIVE_MODEL=""))
        self.assertIn("--disable-radix-cache", cmd)
        self.assertNotIn("--speculative-algorithm", cmd)

    def test_drafter_from_env(self):
        cmd = serve.command(self.env(SPECULATIVE_MODEL="/d", SPECULATIVE_NUM_STEPS="3"))
        i = cmd.index("--speculative-num-steps")
        self.assertEqual(cmd[i + 1], "3")
        self.assertEqual(cmd[cmd.index("--speculative-num-draft-tokens") + 1], "4")
        self.assertEqual(cmd[cmd.index("--speculative-algorithm") + 1], "NEXTN")


if __name__ == "__main__":
    unittest.main()
