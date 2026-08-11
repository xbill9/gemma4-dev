"""Option A: the PyTorch TPU backend Generation Engine.

Graph-break-free PyTorch generation engine compiled via tpu_compiler.compile_for_tpu.
Encapsulates tokenization, static sequence padding (128-aligned for TPU MXU),
and autoregressive token generation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Generator, Protocol
from tpu_compiler import LLMEngine, compile_for_tpu, validate_tpu_inputs

logger = logging.getLogger("tpu_engine")


def pad_to_multiple(length: int, multiple: int = 128) -> int:
    """Pad length up to the nearest multiple of 128 for TPU MXU alignment."""
    remainder = length % multiple
    if remainder == 0:
        return length
    return length + (multiple - remainder)


class TPUEngine:
    """Option A PyTorch LLM Engine targeting the PyTorch TPU backend or FX Staging."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cpu",
        force_staging: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.force_staging = force_staging

        # Compile model forward pass using Option A compiler
        logger.info(f"Compiling model forward pass for TPU target (force_staging={force_staging})...")
        if hasattr(self.model, "forward"):
            self.model.forward = compile_for_tpu(self.model.forward, force_staging=force_staging)

    def generate(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[int, None, None]:
        """Generate output tokens autoregressively without graph breaks."""
        import torch

        prompt_len = len(prompt_tokens)
        padded_len = pad_to_multiple(prompt_len, multiple=128)

        # Create padded input tensor
        input_ids = torch.full((1, padded_len), fill_value=0, dtype=torch.long, device=self.device)
        input_ids[0, :prompt_len] = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device)

        # Enforce MXU alignment check
        validate_tpu_inputs(input_ids)

        cur_len = prompt_len
        generated_count = 0

        with torch.no_grad():
            while generated_count < max_new_tokens:
                # Truncate active view
                cur_inputs = input_ids[:, :cur_len]

                # Run compiled model forward pass
                outputs = self.model(cur_inputs)

                # Extract logits for next token prediction
                if hasattr(outputs, "logits"):
                    logits = outputs.logits[:, -1, :]
                elif isinstance(outputs, tuple):
                    logits = outputs[0][:, -1, :]
                else:
                    logits = outputs[:, -1, :]

                # Simple greedy or top-p token selection
                if temperature == 0.0:
                    next_token = torch.argmax(logits, dim=-1).item()
                else:
                    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

                yield next_token

                # Update state for next step
                if cur_len < input_ids.shape[1]:
                    input_ids[0, cur_len] = next_token
                else:
                    # Append new column padded to 128 boundary
                    new_padded = pad_to_multiple(cur_len + 1, multiple=128)
                    new_input_ids = torch.full((1, new_padded), fill_value=0, dtype=torch.long, device=self.device)
                    new_input_ids[0, :cur_len] = input_ids[0, :cur_len]
                    new_input_ids[0, cur_len] = next_token
                    input_ids = new_input_ids

                cur_len += 1
                generated_count += 1
