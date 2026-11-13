#!/usr/bin/env python3
"""Run one turn of tokenlab_agent and print each tool call, tool result and the answer.

python3 run_agent.py "How many of 0, 2, 3, 20, 21, 22, 23, 10, 11, 12, 13 are 10 or more?"
"""

import asyncio
import json
import sys
import time

from google.adk.runners import InMemoryRunner
from google.genai import types

from tokenlab_agent.agent import root_agent


async def main(prompt: str) -> int:
    runner = InMemoryRunner(agent=root_agent, app_name="tokenlab_agent")
    session = await runner.session_service.create_session(app_name="tokenlab_agent", user_id="local")
    t0 = time.time()
    answer = ""
    async for ev in runner.run_async(
        user_id="local",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
        for part in ev.content.parts if ev.content and ev.content.parts else []:
            dt = f"{time.time() - t0:6.1f}s"
            if part.function_call:
                print(f"{dt} call   {part.function_call.name}({json.dumps(part.function_call.args)})", flush=True)
            elif part.function_response:
                print(f"{dt} result {json.dumps(part.function_response.response)[:400]}", flush=True)
            elif part.text and ev.author == root_agent.name:
                answer += part.text
    print(f"{time.time() - t0:6.1f}s answer {answer.strip()}")
    return 0 if answer.strip() else 1


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How many of 0, 2, 3, 20, 21, 22, 23, 10, 11, 12, 13 are 10 or more?"
    sys.exit(asyncio.run(main(q)))
