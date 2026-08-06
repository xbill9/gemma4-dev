"""Static-KV-cache decode parity test for ports/gemma4/model.py.

Validates (on CPU, float32, random weights, MINI config imported from
parity_test.py) that greedy generation via the cached prefill + decode_step
path is numerically identical to the parity-proven no-cache forward():

  (a) reference: no-cache full forward each step, argmax at the last position;
  (b) cached prefill (SDPA is_causal on full layers) + decode_step;
  (c) cached path with use_sdpa_causal=False (eager causal mask everywhere).

All three must emit IDENTICAL token sequences and per-step logits within 1e-4.
Also checks batch=2 (two same-length prompts), that a larger-than-needed
MAX_SEQ leaves results unchanged (cache padding correctly masked), and that
torch.compile(model.decode_step, backend="aot_eager", fullgraph=True) traces
the decode step without graph breaks.

Run:  cd /home/xbill/tpu-pytorch-v5e1-12b/ports/gemma4 && python3 decode_test.py
"""

import sys

import torch

import model as port_model
from parity_test import MINI_TEXT_CONFIG, fuzz_weights

TOLERANCE = 1e-4
PROMPT_LEN = 12
NEW_TOKENS = 24
MAX_SEQ = 64  # > PROMPT_LEN + NEW_TOKENS - 1 = 35 used positions
ALT_MAX_SEQ = 48  # different padding amount must not change anything


def build_model() -> port_model.Gemma4DenseForCausalLM:
  cfg = port_model.Gemma4DenseConfig.from_text_config(dict(MINI_TEXT_CONFIG))
  model = port_model.Gemma4DenseForCausalLM(cfg).eval()
  fuzz_weights(model)  # same fuzzer as the HF parity test (seeded)
  return model


@torch.no_grad()
def reference_generate(model, input_ids, max_new_tokens):
  """No-cache greedy loop: full forward each step (the parity reference)."""
  tokens = input_ids.clone()
  step_logits = []
  for _ in range(max_new_tokens):
    logits = model(tokens)  # existing no-cache forward, untouched
    last = logits[:, -1, :]
    step_logits.append(last)
    tokens = torch.cat([tokens, last.argmax(dim=-1, keepdim=True)], dim=1)
  return tokens, step_logits


def compare(name, ref, other):
  ref_tokens, ref_logits = ref
  other_tokens, other_logits = other
  tokens_ok = torch.equal(ref_tokens, other_tokens)
  max_diff = max(
      (a - b).abs().max().item() for a, b in zip(ref_logits, other_logits)
  )
  ok = tokens_ok and max_diff < TOLERANCE
  print(
      f"[{name}] tokens {'identical' if tokens_ok else 'DIFFER'}, "
      f"per-step logits max abs diff {max_diff:.3e} "
      f"({'PASS' if ok else 'FAIL'}, tolerance {TOLERANCE:g})"
  )
  if not tokens_ok:
    print(f"  ref:   {ref_tokens.tolist()}")
    print(f"  other: {other_tokens.tolist()}")
  return ok, max_diff


def compile_decode_check(model, input_ids):
  """decode_step must trace fullgraph under aot_eager (static-shape proof)."""
  batch, prompt_len = input_ids.shape
  try:
    compiled = torch.compile(
        model.decode_step, backend="aot_eager", fullgraph=True
    )
    with torch.no_grad():
      cache = model.new_kv_cache(batch, MAX_SEQ)
      logits = model.prefill(input_ids, cache)
      step_logits = [logits]
      next_token = logits.argmax(dim=-1, keepdim=True)
      tokens = torch.cat([input_ids, next_token], dim=1)
      for step in range(NEW_TOKENS - 1):
        pos = torch.tensor([prompt_len + step], dtype=torch.long)
        logits = compiled(next_token, pos, cache)
        step_logits.append(logits)
        next_token = logits.argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
  except Exception as exc:  # graph break / trace failure
    print(f"[compile fullgraph] FAIL: {type(exc).__name__}: {exc}")
    return False, None
  print("[compile fullgraph] decode_step traced with fullgraph=True "
        "(backend=aot_eager): PASS")
  return True, (tokens, step_logits)


def main():
  torch.manual_seed(0)
  model = build_model()
  gen = torch.Generator().manual_seed(42)

  results = []
  ref_by_batch = {}
  for batch in (1, 2):
    prompt = torch.randint(
        1,
        MINI_TEXT_CONFIG["vocab_size"],
        (batch, PROMPT_LEN),
        generator=gen,
    )
    ref = reference_generate(model, prompt, NEW_TOKENS)
    ref_by_batch[batch] = (prompt, ref)

    cached_sdpa = model.generate_cached(
        prompt, NEW_TOKENS, MAX_SEQ, use_sdpa_causal=True
    )
    cached_eager = model.generate_cached(
        prompt, NEW_TOKENS, MAX_SEQ, use_sdpa_causal=False
    )
    cached_alt = model.generate_cached(
        prompt, NEW_TOKENS, ALT_MAX_SEQ, use_sdpa_causal=True
    )
    results.append(
        compare(f"batch={batch} cached sdpa MAX_SEQ={MAX_SEQ}", ref,
                cached_sdpa)
    )
    results.append(
        compare(f"batch={batch} cached eager MAX_SEQ={MAX_SEQ}", ref,
                cached_eager)
    )
    results.append(
        compare(f"batch={batch} cached sdpa MAX_SEQ={ALT_MAX_SEQ}", ref,
                cached_alt)
    )

  prompt, ref = ref_by_batch[2]
  compile_ok, compiled_run = compile_decode_check(model, prompt)
  if compile_ok:
    results.append(compare("batch=2 compiled decode_step", ref, compiled_run))
  else:
    results.append((False, float("inf")))

  worst = max(diff for _, diff in results)
  if all(ok for ok, _ in results):
    print(f"DECODE PASS (worst-case per-step logits max abs diff {worst:.3e})")
    return 0
  print(f"DECODE FAIL (worst-case per-step logits max abs diff {worst:.3e})")
  return 1


if __name__ == "__main__":
  sys.exit(main())
