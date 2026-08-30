"""Offline tests for the serving payload's device policy.

No AWS, no network, no GPU. The compute capability is patched, so these exercise
the pre-Ampere branch on a machine that has no pre-Ampere GPU — a test of the
POLICY, not of the hardware. The hardware claims belong in benchmarks/runs/ and
are measured, not asserted here.

WHAT THIS FILE REPLACED, because the shape of the mistake is worth recording:
the fork carried the JAX rig's tests/test_engine.py, whose 22 tests imported
`ports.gemma4.jax_e_model` and `jax_engine`. This rig vendors neither, and jax is
not a control-plane dependency, so every one of those tests was skipped on every
run — forever, and silently. A suite that always skips reads as coverage in the
summary line and asserts nothing. Note the JAX rig's own reason for the skip was
legitimate there; it stopped being true the moment the engine changed.

torch, pydantic and fastapi ARE control-plane dependencies in practice (they are
installed here), but the guards below are still real: the payload is written to
be importable without CUDA, and this file is where that stays true.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch
    HAVE_TORCH = True
except ImportError:  # pragma: no cover - depends on the host
    HAVE_TORCH = False


def _capability(major, minor):
    """Patch the device capability without patching what reads it.

    get_device_name is patched alongside because resolve_compute_dtype logs it,
    and on a CPU-only host it raises rather than returning a placeholder.
    """
    return mock.patch.multiple(
        "torch.cuda",
        get_device_capability=mock.Mock(return_value=(major, minor)),
        get_device_name=mock.Mock(return_value=f"fake sm_{major}{minor}"),
    )


@unittest.skipUnless(HAVE_TORCH, "torch is not importable on this host")
class DevicePolicyTests(unittest.TestCase):
    """The single most expensive lesson carried over from the JAX sibling.

    bfloat16 on a pre-Ampere GPU DOES NOT RAISE. CUDA accepts it and emulates
    through fp32, so the only symptom is that most of decode disappears into
    conversion. That is worse than an error, and it is why the dtype is read off
    the live device instead of taken from config.
    """

    def _resolve(self, major, minor):
        import torch_openai_server as S
        with _capability(major, minor):
            return S.resolve_compute_dtype(torch.device("cuda"))

    def test_turing_gets_float16(self):
        # SM 7.5 is the T4 in this box and the T4G in the G5g sibling.
        self.assertIs(self._resolve(7, 5), torch.float16)

    def test_the_boundary_is_ampere_not_the_chip_name(self):
        for cc, expected in (((7, 0), torch.float16),    # Volta
                             ((7, 5), torch.float16),    # Turing
                             ((8, 0), torch.bfloat16),   # A100
                             ((8, 6), torch.bfloat16),   # A10G
                             ((8, 9), torch.bfloat16),   # L4, the g6 siblings
                             ((9, 0), torch.bfloat16)):
            with self.subTest(compute_capability=cc):
                self.assertIs(self._resolve(*cc), expected)

    def test_a_cpu_device_never_claims_a_16_bit_path(self):
        # The control plane and any CPU-only box must not silently pick a dtype
        # the device has no unit for.
        import torch_openai_server as S
        self.assertIs(S.resolve_compute_dtype(torch.device("cpu")), torch.float32)

    def test_the_policy_is_logged_not_just_applied(self):
        """A resolved dtype nobody can see is how the JAX rig lost a week.

        Its device-policy banner went to a root logger with no handler and was
        discarded on every run, on the one rig whose entire premise is which
        dtype the device picked.
        """
        import torch_openai_server as S
        with _capability(7, 5), self.assertLogs(level="INFO") as captured:
            S.resolve_compute_dtype(torch.device("cuda"))
        line = "\n".join(captured.output)
        self.assertIn("compute_capability=7.5", line)
        self.assertIn("pre_ampere=True", line)
        self.assertIn("compute_dtype=float16", line)


@unittest.skipUnless(HAVE_TORCH, "torch is not importable on this host")
class DuplicatedPolicyTests(unittest.TestCase):
    """torch_generate.py carries its own copy, and it must not drift.

    It is a deliberate duplicate rather than an import: torch_generate is the
    out-of-band smoke test, and it has to be runnable when the server module
    cannot be imported at all. That is exactly the condition under which a
    drifted copy would go unnoticed, so it is asserted here instead.
    """

    def test_both_copies_agree_across_the_boundary(self):
        import torch_generate as G
        import torch_openai_server as S
        for cc in ((7, 0), (7, 5), (8, 0), (8, 9)):
            with self.subTest(compute_capability=cc), _capability(*cc):
                device = torch.device("cuda")
                self.assertIs(G.resolve_compute_dtype(device),
                              S.resolve_compute_dtype(device))

    def test_both_copies_agree_off_gpu(self):
        import torch_generate as G
        import torch_openai_server as S
        device = torch.device("cpu")
        self.assertIs(G.resolve_compute_dtype(device), S.resolve_compute_dtype(device))


class PayloadShapeTests(unittest.TestCase):
    """Structural claims that hold with or without torch installed."""

    def test_no_jax_import_survives_in_the_payload(self):
        # The fork's verify_gpu step imported jax on a rig that installs none,
        # which killed install.sh under `set -e` and left INSTALL_DONE unwritten.
        # Nothing in the payload may reach for jax.
        for name in ("torch_openai_server.py", "torch_generate.py"):
            with self.subTest(payload=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                self.assertNotIn("import jax", text)
                self.assertNotIn("from jax", text)

    def test_no_vendored_model_port_is_referenced(self):
        # ports/gemma4/ belongs to the JAX rigs. This one uses transformers, so
        # a reference to it here would be a path that does not ship.
        for name in ("torch_openai_server.py", "torch_generate.py", "server.py"):
            with self.subTest(payload=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                self.assertNotIn("from ports", text)
                self.assertNotIn("import ports", text)
        self.assertFalse((ROOT / "ports").exists(), "ports/ is a JAX-rig directory")

    def test_bfloat16_is_never_hardcoded_as_the_compute_dtype(self):
        """The one substitution that would undo this rig's central guard.

        A literal `dtype=torch.bfloat16` would sail past every test above,
        because those test the resolver rather than its callers.
        """
        for name in ("torch_openai_server.py", "torch_generate.py"):
            with self.subTest(payload=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                for line in text.splitlines():
                    code = line.split("#", 1)[0]
                    self.assertNotIn("dtype=torch.bfloat16", code)


if __name__ == "__main__":
    unittest.main()
