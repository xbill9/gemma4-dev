"""Offline tests for the serving payload.

No AWS, no network, no GPU. e2b_engine.py, w4gemv.py, w4attn.py and openai_server.py need CUDA and Triton to do
anything, so here they are only compiled; their hardware behaviour is measured, in benchmarks/runs/ (and, for the
engine itself, in local-pytorch-1650ti-2b-q4w4a16/benchmarks/runs/). The tool-call parser is pure Python and is
exercised directly.
"""

import py_compile
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gemma4_tools import parse_tool_calls  # noqa: E402

PAYLOAD = ("openai_server.py", "e2b_engine.py", "w4gemv.py", "w4attn.py", "gemma4_tools.py")


class PayloadCompilesTests(unittest.TestCase):
    def test_every_payload_module_compiles(self):
        for name in PAYLOAD:
            with self.subTest(module=name):
                py_compile.compile(str(ROOT / name), doraise=True, cfile=None)


class ToolCallParsingTests(unittest.TestCase):
    def test_plain_text_has_no_calls(self):
        content, calls = parse_tool_calls("The answer is 4.")
        self.assertEqual(calls, [])
        self.assertIn("The answer is 4.", content)

    def test_one_call_becomes_an_openai_tool_call(self):
        raw = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:3}<tool_call|>'
        _, calls = parse_tool_calls(raw)
        self.assertEqual(len(calls), 1)
        fn = calls[0]["function"]
        self.assertEqual(fn["name"], "get_weather")
        self.assertIn('"city": "Paris"', fn["arguments"])
        self.assertIn('"days": 3', fn["arguments"])


if __name__ == "__main__":
    unittest.main()
