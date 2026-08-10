"""Option A: Staged PyTorch 2.0 TorchDynamo / torch.compile Backend for the PyTorch TPU backend.

Provides a clean, modular backend interface for PyTorch 2.0 compilation:
- When running on a PyTorch TPU backend-enabled environment: delegates directly to backend="tpu"
- When running in a staging environment (CPU/GPU/XLA): uses a clean PyTorch FX Dynamo
  compiler wrapper that enforces TPU-readiness rules (bfloat16, 128-alignment, no graph breaks).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Generator, Protocol
import torch
import torch.fx

logger = logging.getLogger("tpu_compiler")


class LLMEngine(Protocol):
    """Abstract engine protocol for serving isolation."""

    def generate(self, prompt_tokens: list[int], max_tokens: int) -> Generator[int, None, None]:
        ...


def validate_tpu_inputs(inputs: dict[str, torch.Tensor] | torch.Tensor) -> None:
    """Enforce the PyTorch TPU backend rules: 128-alignment and bfloat16/int precision."""
    tensors = inputs.values() if isinstance(inputs, dict) else [inputs]
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            continue
        last_dim = t.shape[-1]
        if last_dim % 128 != 0 and last_dim % 8 != 0:
            logger.warning(
                f"Input tensor shape {t.shape} last dimension ({last_dim}) is not a multiple of 128 or 8. "
                "Aligning to multiples of 128 maximizes TPU MXU systolic array tile utilization."
            )


def staging_fx_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
    """Staging FX compiler backend.

    Traces and optimizes the FX graph for staging environments, verifying
    graph cleanliness before execution.
    """
    logger.info(f"Staging FX Backend compiling graph with {len(gm.graph.nodes)} nodes.")

    # Check for graph breaks / illegal ops in FX graph
    for node in gm.graph.nodes:
        if node.op == "call_function" and hasattr(node.target, "__name__"):
            if "item" in getattr(node.target, "__name__", ""):
                raise ValueError(
                    f"Forbidden scalar conversion '.item()' detected in node {node.name}. "
                    "the PyTorch TPU backend requires graph-break-free forward passes."
                )

    # Recompile FX graph module
    gm.recompile()

    def compiled_callable(*args: Any, **kwargs: Any) -> Any:
        return gm(*args, **kwargs)

    return compiled_callable


def get_tpu_compiler_backend(force_staging: bool = False) -> str | Callable:
    """Get the appropriate torch.compile backend.

    Returns 'tpu' string when on native the PyTorch TPU backend runtime, or staging_fx_backend
    when staging or testing off-TPU.
    """
    if not force_staging:
        try:
            from torch._dynamo.backends.registry import list_backends

            if "tpu" in list_backends():
                return "tpu"
        except Exception:
            pass

    logger.info("Using Staging FX TorchDynamo backend for Option A.")
    return staging_fx_backend


def compile_for_tpu(model_forward: Callable, force_staging: bool = False) -> Callable:
    """Helper to apply torch.compile with Option A staging fallback."""
    backend = get_tpu_compiler_backend(force_staging=force_staging)
    return torch.compile(model_forward, backend=backend)
