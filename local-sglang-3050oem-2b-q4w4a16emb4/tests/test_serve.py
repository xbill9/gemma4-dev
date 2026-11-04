import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import serve  # noqa: E402

VLLM_ENV = os.path.join(os.path.dirname(serve.HERE), "local-vllm-3050oem-2b-q4w4a16emb4", "tpu.env")


class TestServe(unittest.TestCase):
    def env(self, **over):
        with mock.patch.dict(os.environ, over, clear=False):
            return serve.load_env()

    def test_same_checkpoint_as_vllm_rig(self):
        mine, theirs = serve.load_env(), serve.load_env(VLLM_ENV)
        self.assertEqual(mine["MODEL_NAME"], theirs["MODEL_NAME"])
        self.assertEqual(mine["MODEL_REVISION"], theirs["MODEL_REVISION"])

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
