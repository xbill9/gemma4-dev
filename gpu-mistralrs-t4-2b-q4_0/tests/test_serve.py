"""Offline tests: the serve command built from tpu.env."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import serve  # noqa: E402


class TestServeCommand(unittest.TestCase):
    def test_serves_the_llamacpp_rigs_gguf(self):
        env = serve.load_env()
        cmd = serve.command(env)
        self.assertEqual(cmd[1], "serve")
        self.assertEqual(cmd[cmd.index("-f") + 1], env["MODEL_PATH"])
        self.assertIn("--mmproj", cmd)
        self.assertEqual(cmd[cmd.index("--prefix-cache-n") + 1], "0")
        llama = serve.load_env(os.path.join(serve.HERE, "..", "gpu-llamacpp-t4-2b-q4_0", "tpu.env"))
        self.assertEqual(env["MODEL_PATH"], llama["MODEL_PATH"])

    def test_no_drafter_by_default(self):
        self.assertNotIn("--mtp-model", serve.command(serve.load_env()))

    def test_drafter_from_env(self):
        with mock.patch.dict(os.environ, {"MTP_MODEL": "/d", "MTP_N_PREDICT": "3"}):
            cmd = serve.command(serve.load_env())
        self.assertEqual(cmd[cmd.index("--mtp-model") + 1], "/d")
        self.assertEqual(cmd[cmd.index("--mtp-n-predict") + 1], "3")


if __name__ == "__main__":
    unittest.main()
