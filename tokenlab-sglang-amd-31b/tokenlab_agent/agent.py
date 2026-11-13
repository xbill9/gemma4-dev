"""A small ADK agent whose model is Gemma 4 31B on AMD Token Lab.

ADK reaches Token Lab through LiteLLM's OpenAI provider: model "openai/<MODEL_NAME>",
api_base = Token Lab's base URL, api_key = TOKENLAB_API_KEY. Both come from this rig's
tpu.env and .env via tokenlab.load_config(), so nothing is duplicated here.

Two tools:
  list_tokenlab_models  - what Token Lab serves right now (GET /v1/models)
  count_and_range       - counting, filtering and min/max done in code, so the model
                          quotes an exact result instead of doing arithmetic itself

Run from the rig directory:
  adk run tokenlab_agent            # terminal chat
  adk web                           # browser dev UI, pick tokenlab_agent
  python3 run_agent.py "question"   # one turn, prints tool calls and the answer
"""

import os
import sys
from typing import Optional

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tokenlab  # noqa: E402

CFG = tokenlab.load_config()


def list_tokenlab_models() -> dict:
    """List the models AMD Token Lab serves right now.

    Returns:
        dict: {"models": [{"id", "service", "context_length", "inputs"}...]} or {"error": ...}.
    """
    resp = tokenlab.open_api(CFG, "/models", timeout=20)
    if resp.status != 200:
        return {"error": f"HTTP {resp.status}: {tokenlab.error_message(resp)}"}
    import json

    with resp:
        data = json.loads(resp.read()).get("data", [])
    return {
        "models": [
            {
                "id": m.get("id"),
                "service": m.get("description"),
                "context_length": m.get("context_length"),
                "inputs": (m.get("architecture") or {}).get("input_modalities"),
            }
            for m in data
        ]
    }


def count_and_range(numbers: list[float], at_least: Optional[float] = None, below: Optional[float] = None) -> dict:
    """Count the numbers that satisfy at_least <= x < below, and give their min, max and sum.

    Use this for any counting, filtering, summing or comparing instead of doing it yourself.
    Leave a bound out (null) to leave that side open.

    Args:
        numbers: the values to examine.
        at_least: inclusive lower bound, or null for none.
        below: exclusive upper bound, or null for none.

    Returns:
        dict: the filter applied, the count of matching values, and their min, max and sum.
    """
    hits = [x for x in numbers if (at_least is None or x >= at_least) and (below is None or x < below)]
    return {
        "filter": f"{'' if at_least is None else f'{at_least} <= '}x{'' if below is None else f' < {below}'}",
        "total_values": len(numbers),
        "count": len(hits),
        "min": min(hits) if hits else None,
        "max": max(hits) if hits else None,
        "sum": sum(hits),
    }


model = LiteLlm(
    model=f"openai/{CFG['MODEL_NAME']}",
    api_base=CFG["TOKENLAB_BASE_URL"],
    api_key=CFG.get("TOKENLAB_API_KEY"),
    # Token Lab answers 429 at its shared per-model limit and asks clients to retry.
    num_retries=int(CFG.get("RETRY_MAX", 30)),
    timeout=600,
    # Thinking off: answers start sooner on a slow shared endpoint.
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

root_agent = Agent(
    name="tokenlab_agent",
    model=model,
    description="Gemma 4 31B on AMD Token Lab, with a model-listing tool and an exact counting tool.",
    instruction=(
        "You are a concise assistant running as Gemma 4 31B on AMD Token Lab. "
        "When asked what Token Lab serves, call list_tokenlab_models and answer from its result. "
        "Never count, filter, sum or compare numbers yourself: call count_and_range with the exact "
        "bounds the question states ('10 or more' is at_least=10; 'under 5' is below=5), then quote "
        "its count and say which filter you used."
    ),
    tools=[list_tokenlab_models, count_and_range],
)
