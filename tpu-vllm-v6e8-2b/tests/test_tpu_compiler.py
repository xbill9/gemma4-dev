"""Unit tests for Option A TorchDynamo staging compiler module."""

import unittest
import torch
import torch.nn as nn
from tpu_compiler import compile_for_tpu, validate_tpu_inputs, get_tpu_compiler_backend


class SimpleMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TestTPUCompilerOptionA(unittest.TestCase):

    def test_validate_tpu_inputs(self):
        # 128 aligned tensor
        aligned_tensor = torch.randn(2, 128)
        validate_tpu_inputs(aligned_tensor)

        # Unaligned tensor (should trigger warning without error)
        unaligned_tensor = torch.randn(2, 50)
        validate_tpu_inputs(unaligned_tensor)

    def test_compile_for_tpu_staging(self):
        model = SimpleMLP().eval()
        compiled_forward = compile_for_tpu(model.forward, force_staging=True)

        x = torch.randn(4, 128, dtype=torch.bfloat16)
        with torch.no_grad():
            output_orig = model(x)
            output_compiled = compiled_forward(x)

        self.assertEqual(output_orig.shape, output_compiled.shape)
        torch.testing.assert_close(output_orig, output_compiled)

    def test_backend_selection(self):
        backend = get_tpu_compiler_backend(force_staging=True)
        self.assertTrue(callable(backend))


if __name__ == "__main__":
    unittest.main()
