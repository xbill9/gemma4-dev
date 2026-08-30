"""Offline tests for the serving payload's device policy.

No AWS, no network, no GPU. The compute capability is stubbed, so these
exercise the *policy* on a machine with no CUDA device at all — the hardware
claims live in benchmarks/runs/ and are measured, not asserted here.

This file replaced a JAX one at the PyTorch fork. That version imported
`ports.gemma4.jax_e_model`, which does not exist in a PyTorch rig, so it caught
its own ImportError and skipped every test — silently, forever. A test file that
can only skip is worse than no test file: it reports green. If you ever see
these skip, check that torch and pydantic are installed rather than assuming the
skip is expected.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch

    import torch_openai_server as S
    HAVE_TORCH = True
except ImportError:  # pragma: no cover - depends on the host
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch/pydantic are serving deps, not control-plane ones")
class DevicePolicyTests(unittest.TestCase):
    """The dtype is read off the device, never off a config."""

    def _at_capability(self, cc):
        """Run resolve_compute_dtype as if the device reported capability `cc`."""
        with mock.patch.object(torch.cuda, "get_device_capability", return_value=cc), \
             mock.patch.object(torch.cuda, "get_device_name", return_value=f"stub-sm{cc[0]}{cc[1]}"):
            return S.resolve_compute_dtype(torch.device("cuda"))

    def test_compute_dtype_follows_compute_capability(self):
        """Ampere and later get bfloat16; anything below it gets float16.

        SM 8.9 is THIS rig's L4 and is the case that matters here. SM 7.5 is the
        T4G sibling, kept because the same file serves both and a regression
        would be invisible on Ada.
        """
        for cc, expected in (
            ((8, 9), torch.bfloat16),   # L4, Ada -- this rig
            ((8, 0), torch.bfloat16),   # A100, the boundary itself
            ((7, 5), torch.float16),    # T4/T4G, Turing -- the sibling
            ((7, 0), torch.float16),    # V100, Volta
        ):
            with self.subTest(compute_capability=cc):
                self.assertIs(self._at_capability(cc), expected)

    def test_the_boundary_is_ampere_not_the_major_version(self):
        # (7, 5) < (8, 0) is the whole rule. A `major < 8` test would agree here
        # but a `major <= 8` or a float comparison on "8.9" would not.
        self.assertIs(self._at_capability((8, 6)), torch.bfloat16)
        self.assertIs(self._at_capability((7, 5)), torch.float16)

    def test_bfloat16_is_never_chosen_below_ampere(self):
        """The one failure this guard exists for, and it is SILENT.

        bfloat16 on a pre-Ampere GPU does not raise -- CUDA emulates it through
        fp32, output stays correct, and decode simply loses most of its
        throughput. So there is no error to assert on; the only defence is that
        the branch is never taken.
        """
        for cc in ((7, 5), (7, 0), (6, 1)):
            with self.subTest(compute_capability=cc):
                self.assertIsNot(self._at_capability(cc), torch.bfloat16)

    def test_non_cuda_device_falls_back_to_float32(self):
        # A CPU host must not be handed a 16-bit dtype: this path is what makes
        # the module importable off-accelerator for the control plane.
        self.assertIs(S.resolve_compute_dtype(torch.device("cpu")), torch.float32)


if __name__ == "__main__":
    unittest.main()
