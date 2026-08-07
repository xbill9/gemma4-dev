"""Deterministic output probe for the bf16-vs-fp8 KV cache comparison.

Runs a fixed prompt set at temperature 0 against /v1/chat/completions and writes every
response verbatim. Run it once per arm; diff the two JSON files to get the divergence rate.

Why chat and not completions: raw /v1/completions returns an empty completion on -it models,
which is exactly what benchmarking_suite.py hits. Useless for measuring output quality.

Why the long-context probes matter most: tpu_inference quantizes K/V with
static_per_tensor_quantize_tensor at a hardcoded scale of 1.0 (models/jax/gemma4.py:407-408,
calculate_kv_scales=False). Quantization error lands in cached KV, so it accumulates with the
number of cached tokens. A 14k-token recall is far more sensitive than a 20-token question.

stdlib only on purpose — this runs on the TPU VM host, outside the container.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

MODEL = "google/gemma-4-E2B-it"

# Fixed filler. Byte-identical across arms or the comparison is meaningless.
FILLER = "The maintenance log records routine inspection of the coolant loop and no anomalies were observed. "
NEEDLE = "The calibration constant for bay seven is 48213."
NEEDLE_QUESTION = "What is the calibration constant for bay seven? Answer with just the number."


def build_haystack(approx_tokens: int, depth: float) -> str:
    """A deterministic haystack with the needle buried at `depth` through it.

    approx_tokens is approximate by design (~4 chars/token); the exact length does not
    matter, only that it is identical across arms.
    """
    reps = max(1, (approx_tokens * 4) // len(FILLER))
    before = int(reps * depth)
    return (FILLER * before) + NEEDLE + " " + (FILLER * (reps - before))


def prompt_set() -> List[Dict[str, Any]]:
    """The probe set. Short prompts catch gross breakage; long ones catch KV error accumulation."""
    probes: List[Dict[str, Any]] = [
        {
            "id": "short-factual",
            "max_tokens": 128,
            "prompt": "What is the capital of Australia? Answer in one sentence.",
        },
        {"id": "short-arithmetic", "max_tokens": 256, "prompt": "What is 847 * 63? Show your working step by step."},
        {
            "id": "short-code",
            "max_tokens": 256,
            "prompt": "Write a Python function that returns the nth Fibonacci number iteratively. Code only.",
        },
        {
            "id": "short-instruction",
            "max_tokens": 128,
            "prompt": "List exactly three primary colors, one per line, no other text.",
        },
        {
            "id": "short-multilingual",
            "max_tokens": 128,
            "prompt": "Translate to French, output only the translation: 'The server is running low on memory.'",
        },
        {
            "id": "short-regional",
            "max_tokens": 256,
            "prompt": "In New Jersey, is it pork roll or Taylor ham?",
        },
    ]
    for approx, depth in ((2000, 0.5), (8000, 0.5), (14000, 0.75)):
        probes.append(
            {
                "id": f"needle-{approx}",
                "max_tokens": 32,
                "prompt": build_haystack(approx, depth) + "\n\n" + NEEDLE_QUESTION,
            }
        )
    return probes


def query(url: str, prompt: str, max_tokens: int, timeout: float) -> Dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "stream": False,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - started
    choice = data["choices"][0]
    return {
        "text": choice["message"]["content"],
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}),
        "elapsed_s": round(elapsed, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--arm", required=True, help="Label for this arm, e.g. bf16 or fp8")
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    results = []
    for probe in prompt_set():
        print(f"[{args.arm}] {probe['id']} ...", flush=True)
        error: Optional[str] = None
        got: Dict[str, Any] = {}
        try:
            got = query(args.url, probe["prompt"], probe["max_tokens"], args.timeout)
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            error = f"{type(e).__name__}: {e}"
            print(f"  FAILED: {error}", flush=True)
        results.append(
            {
                "id": probe["id"],
                "prompt_chars": len(probe["prompt"]),
                "max_tokens": probe["max_tokens"],
                "error": error,
                **got,
            }
        )

    with open(args.output, "w") as f:
        json.dump({"arm": args.arm, "model": MODEL, "results": results}, f, indent=2)
    ok = sum(1 for r in results if not r["error"])
    print(f"\nWrote {args.output} — {ok}/{len(results)} probes succeeded")


if __name__ == "__main__":
    main()
