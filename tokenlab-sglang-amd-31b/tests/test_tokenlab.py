"""Offline tests: config precedence, retry behaviour, usage parsing. No network."""

import io
import os
import sys
import tempfile
import unittest
import urllib.error
from email.message import Message
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import chat_ui  # noqa: E402
import tokenlab  # noqa: E402


def http_error(code, retry_after=None):
    h = Message()
    if retry_after is not None:
        h["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError(
        "https://x/v1/chat/completions", code, "err", h, io.BytesIO(b'{"error":{"message":"busy"}}')
    )


class Ok:
    status = 200


class ConfigTest(unittest.TestCase):
    def test_precedence_env_file_then_environment(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "tpu.env"), "w") as f:
                f.write("# c\nMODEL_NAME=A\nTOKENLAB_BASE_URL=https://h/v1/\n")
            with open(os.path.join(d, ".env"), "w") as f:
                f.write("TOKENLAB_API_KEY='k1'\n")
            with mock.patch.dict(os.environ, {"MODEL_NAME": "B"}, clear=False):
                os.environ.pop("TOKENLAB_API_KEY", None)
                cfg = tokenlab.load_config(d)
        self.assertEqual(cfg["MODEL_NAME"], "B")
        self.assertEqual(cfg["TOKENLAB_API_KEY"], "k1")
        self.assertEqual(cfg["TOKENLAB_BASE_URL"], "https://h/v1")

    def test_rig_tpu_env_names_the_model(self):
        env = tokenlab.read_env(os.path.join(tokenlab.HERE, "tpu.env"))
        self.assertEqual(env["MODEL_NAME"], "Gemma-4-31B-IT")
        self.assertTrue(env["TOKENLAB_BASE_URL"].startswith("https://api.tokenlab.amd.com"))


class RetryTest(unittest.TestCase):
    cfg = {"TOKENLAB_API_KEY": "k", "TOKENLAB_BASE_URL": "https://h/v1", "RETRY_MAX": "4", "RETRY_WAIT_S": "0.5"}

    @mock.patch("tokenlab.time.sleep")
    def test_429_then_ok_honours_retry_after(self, sleep):
        seen = []
        with mock.patch("urllib.request.urlopen", side_effect=[http_error(429, 1), http_error(503), Ok()]) as u:
            r = tokenlab.open_api(self.cfg, "/chat/completions", b"{}", on_retry=lambda *a: seen.append(a))
        self.assertEqual(r.status, 200)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1.0, 0.5])
        self.assertEqual([s[1] for s in seen], [429, 503])
        req = u.call_args.args[0]
        self.assertEqual(req.get_header("Authorization"), "Bearer k")
        self.assertEqual(req.full_url, "https://h/v1/chat/completions")

    @mock.patch("tokenlab.time.sleep")
    def test_non_retryable_returned_at_once(self, sleep):
        with mock.patch("urllib.request.urlopen", side_effect=[http_error(403)]):
            r = tokenlab.open_api(self.cfg, "/models")
        self.assertEqual(r.code, 403)
        sleep.assert_not_called()

    @mock.patch("tokenlab.time.sleep")
    def test_gives_up_after_retry_max(self, sleep):
        with mock.patch("urllib.request.urlopen", side_effect=[http_error(429) for _ in range(4)]):
            r = tokenlab.open_api(self.cfg, "/chat/completions", b"{}")
        self.assertEqual(r.code, 429)
        self.assertEqual(sleep.call_count, 3)
        self.assertEqual(tokenlab.error_message(r), "busy")

    def test_missing_key_raises(self):
        with self.assertRaises(RuntimeError):
            tokenlab.open_api({"TOKENLAB_BASE_URL": "https://h/v1"}, "/models")


class UsageTest(unittest.TestCase):
    def test_last_usage_in_sse_tail(self):
        tail = (
            b': ping\n\ndata: {"choices":[{"delta":{"content":"hi"}}],"usage":null}\nid: 3\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":8,"cost":1.1e-06}}\n\n'
            b"event: done\ndata: [DONE]\n\n"
        )
        u = chat_ui.usage_from_sse(tail)
        self.assertEqual(u["completion_tokens"], 8)
        self.assertAlmostEqual(u["cost"], 1.1e-06)

    def test_no_usage(self):
        self.assertIsNone(chat_ui.usage_from_sse(b"data: [DONE]\n"))


if __name__ == "__main__":
    unittest.main()
