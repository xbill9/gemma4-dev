"""Offline checks of gen_eval.py's scoring: python3 -m unittest discover -s tests -v"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gen_eval as g  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def bfcl():
    return [json.loads(line) for line in open(os.path.join(DATA, "bfcl_simple.jsonl"))]


def realize(v, pick):
    """An accepted value as a model would send it: nested dicts hold lists of alternatives."""
    if isinstance(v, dict):
        out = {}
        for k, alts in v.items():
            given = [a for a in alts if a != ""]
            if given:
                out[k] = realize(given[min(pick, len(given) - 1)], pick)
        return out
    if isinstance(v, list):
        return [realize(x, pick) for x in v]
    return v


def reference_call(rec, pick=0):
    name, acc = next(iter(rec["possible"][0].items()))
    return [{"name": g.tool_name(name), "arguments": realize(acc, pick)}]


class Gsm8k(unittest.TestCase):
    def test_answer_line_wins(self):
        self.assertTrue(g.gsm8k_right("3 + 4 = 7 so 7 * 2 = 14\nAnswer: 14", "14"))

    def test_falls_back_to_last_number(self):
        self.assertTrue(g.gsm8k_right("The total is $1,250.", "1250"))

    def test_wrong(self):
        self.assertFalse(g.gsm8k_right("Answer: 13", "14"))
        self.assertFalse(g.gsm8k_right("no idea", "14"))
        self.assertFalse(g.gsm8k_right(None, "14"))

    def test_bold_and_decimal(self):
        self.assertTrue(g.gsm8k_right("Answer: **18.00**", "18"))


class Bfcl(unittest.TestCase):
    def test_every_reference_answer_passes(self):
        fails = [
            (r["id"], g.bfcl_check(reference_call(r), r)[1])
            for r in bfcl()
            if not g.bfcl_check(reference_call(r), r)[0]
        ]
        self.assertEqual(fails, [])

    def test_every_alternative_answer_passes(self):
        fails = [r["id"] for r in bfcl() if not g.bfcl_check(reference_call(r, pick=1), r)[0]]
        self.assertEqual(fails, [])

    def test_wrong_value_fails(self):
        r = bfcl()[0]
        call = reference_call(r)
        call[0]["arguments"]["base"] = 11
        self.assertFalse(g.bfcl_check(call, r)[0])

    def test_missing_required_fails(self):
        r = bfcl()[0]
        call = reference_call(r)
        del call[0]["arguments"]["height"]
        self.assertFalse(g.bfcl_check(call, r)[0])

    def test_extra_parameter_fails(self):
        r = bfcl()[0]
        call = reference_call(r)
        call[0]["arguments"]["colour"] = "red"
        self.assertFalse(g.bfcl_check(call, r)[0])

    def test_no_call_or_two_calls_fail(self):
        r = bfcl()[0]
        self.assertFalse(g.bfcl_check([], r)[0])
        self.assertFalse(g.bfcl_check(reference_call(r) * 2, r)[0])

    def test_string_standardized(self):
        self.assertTrue(g.match("New York, NY", "new york ny", "string"))
        self.assertFalse(g.match("Boston", "new york ny", "string"))

    def test_int_accepted_for_float(self):
        self.assertTrue(g.match(5, 5.0, "float"))
        self.assertFalse(g.match(5.5, 5, "integer"))

    def test_schema_converted(self):
        t = g.tools_for(bfcl()[0]["functions"])[0]["function"]
        self.assertEqual(t["parameters"]["type"], "object")
        dotted = [r for r in bfcl() if "." in r["functions"][0]["name"]][0]
        self.assertNotIn(".", g.tools_for(dotted["functions"])[0]["function"]["name"])


if __name__ == "__main__":
    unittest.main()
