import json
import shlex
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lmi  # noqa: E402


class TestEnv(unittest.TestCase):
    def test_quotes_stripped(self):
        self.assertEqual(lmi.E["ACCELERATOR_TYPE"], "NVIDIA L4")

    def test_engine_known(self):
        self.assertIn(lmi.E["ENGINE"], ("vllm", "sglang", "cmd"))

    def test_function_env_fits_lambda(self):
        v = lmi.function_env()["Variables"]
        # Lambda caps the whole environment at 4 KB.
        self.assertLess(len(json.dumps(v)), 4096)
        self.assertEqual(v["SERVED_MODEL_NAME"], lmi.E["MODEL_NAME"])

    def test_cmd_engine_command_parses(self):
        if lmi.E["ENGINE"] != "cmd":
            self.skipTest("not a cmd engine")
        cmd = shlex.split(lmi.function_env()["Variables"]["ENGINE_CMD"])
        self.assertTrue(cmd[0].startswith("/"))
        self.assertIn(lmi.E["ENGINE_PORT"], cmd)
        self.assertTrue((ROOT / "image" / "engine_ext.py").exists())
        self.assertTrue((ROOT / "image" / "handler.py").exists())

    def test_build_args_defined(self):
        for k, v in lmi.build_args().items():
            self.assertTrue(v, k)
        df = (ROOT / "image" / "Dockerfile").read_text()
        for k in lmi.build_args():
            self.assertIn(f"ARG {k}", df)

    def test_accelerator_memory_valid(self):
        self.assertIn(int(lmi.E["ACCELERATOR_MEMORY_GB"]), (3, 6, 12, 16, 24, 48))

    def test_scaling_fits_quota(self):
        # g6.4xlarge has 16 vCPUs; max environments x 16 must fit MAX_VCPU_COUNT.
        self.assertLessEqual(int(lmi.E["MAX_ENVS"]) * 16, int(lmi.E["MAX_VCPU_COUNT"]))

    def test_memory_fits_instance(self):
        # g6.2xlarge: 8 vCPUs, 32,768 MiB.
        vcpus = int(lmi.E["MEMORY_MB"]) / 1024 / float(lmi.E["MEMORY_GIB_PER_VCPU"])
        self.assertLessEqual(vcpus, 8)


if __name__ == "__main__":
    unittest.main()
