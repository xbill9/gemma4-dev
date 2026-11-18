import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lmi  # noqa: E402


class TestEnv(unittest.TestCase):
    def test_quotes_stripped(self):
        self.assertEqual(lmi.E["ACCELERATOR_TYPE"], "NVIDIA L4")

    def test_server_args_match_g6_rig(self):
        a = lmi.server_args()
        for flag in (
            "--max-num-seqs 64",
            "--language-model-only",
            "--enable-auto-tool-choice",
            "--tool-call-parser gemma4",
            "--reasoning-parser gemma4",
        ):
            self.assertIn(flag, a)

    def test_function_env(self):
        v = lmi.function_env()["Variables"]
        self.assertEqual(v["MODEL_ID"], "/opt/model")
        self.assertEqual(v["SERVED_MODEL_NAME"], lmi.E["MODEL_NAME"])
        self.assertEqual(v["VLLM_GPU_MEM_UTIL"], "0.85")

    def test_accelerator_memory_valid(self):
        self.assertIn(int(lmi.E["ACCELERATOR_MEMORY_GB"]), (3, 6, 12, 16, 24, 48))

    def test_scaling_fits_quota(self):
        # g6.2xlarge has 8 vCPUs; max environments x 8 must fit MAX_VCPU_COUNT.
        self.assertLessEqual(int(lmi.E["MAX_ENVS"]) * 8, int(lmi.E["MAX_VCPU_COUNT"]))


if __name__ == "__main__":
    unittest.main()
