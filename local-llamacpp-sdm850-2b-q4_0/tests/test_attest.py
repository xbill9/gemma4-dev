"""Offline unit tests for arm attestation.

attest.py answers one question — WHICH BINARY IS ANSWERING ON THE PORT — and it
exists because this rig and `local-llamacpp-1650ti-2b-q4_0` are two arms of a
control that deliberately share one endpoint. Nothing here touches a real
process: /proc is faked.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

RIG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RIG_DIR))

import attest  # noqa: E402

CPU_MAPS = (
    "55e0-55e1 r-xp 00000000 00:01 100 /home/xbill/llama.cpp/build-cpu/bin/llama-server\n"
    "7f00-7f01 r-xp 00000000 00:01 101 /usr/lib/x86_64-linux-gnu/libm.so.6\n"
    "7f02-7f03 r-xp 00000000 00:01 102 /home/xbill/llama.cpp/build-cpu/bin/libggml-cpu.so\n"
    "7f04-7f05 rw-p 00000000 00:00 0 \n"
)
GPU_MAPS = CPU_MAPS + (
    "7f06-7f07 r-xp 00000000 00:01 103 /home/xbill/llama.cpp/build/bin/libggml-cuda.so\n"
    "7f08-7f09 r-xp 00000000 00:01 104 /usr/lib/x86_64-linux-gnu/libcuda.so.1\n"
)


class TestExpectedDevice(unittest.TestCase):
    def test_this_rig_is_the_cpu_arm(self):
        """Not configurable, for the same reason -ngl 0 is not: an arm that can be
        flipped by an env var measures whichever device it happened to find."""
        self.assertEqual(attest.EXPECTED_DEVICE, "cpu")

    def test_expected_device_is_not_read_from_the_environment(self):
        src = (RIG_DIR / "attest.py").read_text()
        self.assertNotIn("EXPECTED_DEVICE = os.environ", src)
        self.assertNotIn("getenv(\"EXPECTED_DEVICE\"", src)


class TestMappedGpuLibs(unittest.TestCase):
    """/proc/<pid>/maps, not ldd: llama.cpp dlopen's its backends, so a CUDA
    backend can be absent from ldd output and present in the live process."""

    def test_cpu_process_maps_no_gpu_libraries(self):
        with patch.object(attest, "_proc_text", return_value=CPU_MAPS):
            self.assertEqual(attest._mapped_gpu_libs(1), [])

    def test_gpu_process_is_caught_by_its_mappings(self):
        with patch.object(attest, "_proc_text", return_value=GPU_MAPS):
            self.assertEqual(attest._mapped_gpu_libs(1), ["ggml-cuda", "libcuda"])

    def test_anonymous_mappings_are_skipped(self):
        with patch.object(attest, "_proc_text", return_value="7f04-7f05 rw-p 0 00:00 0 \n"):
            self.assertEqual(attest._mapped_gpu_libs(1), [])


class TestFlagValue(unittest.TestCase):
    def test_reads_the_real_argv_not_the_env_file(self):
        argv = ["llama-server", "-m", "/models/x.gguf", "-ngl", "99", "-t", "4"]
        self.assertEqual(attest._flag_value(argv, "-ngl", "--n-gpu-layers"), "99")
        self.assertEqual(attest._flag_value(argv, "-m", "--model"), "/models/x.gguf")
        self.assertIsNone(attest._flag_value(argv, "-tb", "--threads-batch"))

    def test_equals_form(self):
        self.assertEqual(attest._flag_value(["s", "--ctx-size=8192"], "-c", "--ctx-size"), "8192")

    def test_trailing_flag_with_no_value(self):
        self.assertIsNone(attest._flag_value(["llama-server", "-ngl"], "-ngl"))


class TestDeviceVerdict(unittest.TestCase):
    """The verdict needs TWO signals to agree: a GPU backend mapped in, and
    layers actually assigned to it."""

    def _attest(self, maps, argv, environ=""):
        def proc_text(pid, name):
            return {"maps": maps, "cmdline": "\0".join(argv), "environ": environ}[name]
        with patch.object(attest, "pid_owning_port", return_value=99), \
             patch.object(attest, "_proc_text", side_effect=proc_text), \
             patch.object(attest, "_exe", return_value="/bin/llama-server"), \
             patch.object(attest, "sha256_of", return_value="f" * 64):
            return attest.attest_port(8080)

    def test_cpu_arm(self):
        att = self._attest(CPU_MAPS, ["llama-server", "-ngl", "0", "-t", "4"],
                           "CUDA_VISIBLE_DEVICES=\0")
        self.assertEqual(att["device"], "cpu")
        self.assertTrue(att["serving"])
        self.assertEqual(att["cuda_visible_devices"], "")
        self.assertIsNone(attest.mismatch(att, "cpu"))

    def test_gpu_arm(self):
        att = self._attest(GPU_MAPS, ["llama-server", "-ngl", "99", "-t", "4"])
        self.assertEqual(att["device"], "gpu")
        self.assertEqual(att["n_gpu_layers"], 99)
        self.assertIsNone(attest.mismatch(att, "gpu"))

    def test_gpu_arm_is_a_mismatch_for_this_rig(self):
        att = self._attest(GPU_MAPS, ["llama-server", "-ngl", "99"])
        why = attest.mismatch(att, attest.EXPECTED_DEVICE)
        self.assertIsNotNone(why)
        self.assertIn("gpu", why)

    def test_cuda_build_with_ngl_zero_is_mixed_not_cpu(self):
        """The case the rig's -ngl 0 alone would not catch: the device is
        initialised and llama.cpp can still move large prefill batches onto it.
        Reported as `mixed`, never rounded down to `cpu`."""
        att = self._attest(GPU_MAPS, ["llama-server", "-ngl", "0"])
        self.assertEqual(att["device"], "mixed")
        self.assertIsNotNone(attest.mismatch(att, "cpu"))
        self.assertIsNotNone(attest.mismatch(att, "gpu"))

    def test_missing_ngl_flag_defaults_to_zero(self):
        att = self._attest(CPU_MAPS, ["llama-server", "-t", "4"])
        self.assertEqual(att["device"], "cpu")

    def test_unparseable_ngl_does_not_raise(self):
        att = self._attest(CPU_MAPS, ["llama-server", "-ngl", "all"])
        self.assertEqual(att["n_gpu_layers"], 0)


class TestNothingServing(unittest.TestCase):
    def test_no_owner_is_not_serving(self):
        with patch.object(attest, "pid_owning_port", return_value=None):
            att = attest.attest_port(8080)
        self.assertFalse(att["serving"])
        self.assertEqual(att["device"], "none")
        self.assertIn("nothing is serving", attest.mismatch(att, "cpu"))

    def test_expected_any_never_mismatches(self):
        with patch.object(attest, "pid_owning_port", return_value=None):
            att = attest.attest_port(8080)
        self.assertIsNone(attest.mismatch(att, "any"))


if __name__ == "__main__":
    unittest.main()
