"""Offline tests: the serve command built from tpu.env, and the attestation verdict."""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import attest  # noqa: E402
import serve  # noqa: E402


class TestServeCommand(unittest.TestCase):
    def test_pins_model_revision_and_parsers(self):
        cmd = serve.command(serve.load_env())
        env = serve.load_env()
        self.assertEqual(cmd[:3], ["vllm", "serve", env["MODEL_NAME"]])
        self.assertIn(env["MODEL_REVISION"], cmd)
        self.assertIn("--language-model-only", cmd)
        self.assertEqual(cmd[cmd.index("--tool-call-parser") + 1], "gemma4")
        self.assertIn("--enable-auto-tool-choice", cmd)

    def test_no_drafter_by_default(self):
        self.assertNotIn("--speculative-config", serve.command(serve.load_env()))

    def test_drafter_from_env(self):
        with mock.patch.dict(os.environ, {"SPECULATIVE_MODEL": "x/drafter", "NUM_SPECULATIVE_TOKENS": "3"}):
            cmd = serve.command(serve.load_env())
        spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
        self.assertEqual(spec, {"model": "x/drafter", "num_speculative_tokens": 3})

    def test_environment_wins_over_file(self):
        with mock.patch.dict(os.environ, {"MAX_NUM_SEQS": "8"}):
            cmd = serve.command(serve.load_env())
        self.assertEqual(cmd[cmd.index("--max-num-seqs") + 1], "8")


class TestAttestVerdict(unittest.TestCase):
    def _attest(self, libs_by_pid, environ):
        with (
            mock.patch.object(attest, "pid_owning_port", return_value=10),
            mock.patch.object(attest, "_children", return_value=[11]),
            mock.patch.object(attest, "_cmdline", return_value=["vllm", "serve"]),
            mock.patch.object(attest, "_environ", return_value=environ),
            mock.patch.object(attest, "_exe", return_value=""),
            mock.patch.object(attest, "_mapped_gpu_libs", side_effect=lambda p: libs_by_pid.get(p, [])),
        ):
            return attest.attest_port(8010)

    def test_gpu_when_child_maps_cuda(self):
        att = self._attest({11: ["libcuda"]}, {})
        self.assertEqual(att["device"], "gpu")
        self.assertIsNone(attest.mismatch(att, "gpu"))

    def test_cpu_when_nothing_mapped(self):
        self.assertEqual(self._attest({}, {})["device"], "cpu")

    def test_cpu_when_cards_hidden(self):
        self.assertEqual(self._attest({11: ["libcuda"]}, {"CUDA_VISIBLE_DEVICES": ""})["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
