"""HTTP-level tests for jax_openai_server backed by the pure-JAX engine.

Boots the FastAPI app against a synthetic tiny checkpoint with a stub tokenizer
(no Hub download) and drives the real endpoints, including SSE streaming.

Run: python3 -m unittest tests.test_openai_server
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fastapi.testclient import TestClient  # noqa: E402

import jax_openai_server as srv  # noqa: E402
from jax_engine import JaxGemmaEngine  # noqa: E402
from test_jax_engine import write_tiny_checkpoint  # noqa: E402


class StubTokenizer:
    """Minimal byte-level stand-in; avoids a Hub download in tests."""

    eos_token_id = 200
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        text = " ".join(m["content"] for m in messages)
        return self(text)["input_ids"]

    def __call__(self, text):
        # No truncation: lets a long prompt actually exceed max_model_len.
        ids = [(b % 250) + 1 for b in text.encode()]
        return {"input_ids": ids or [1]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(f"<{int(i)}>" for i in ids)

    def convert_tokens_to_ids(self, tok):
        return -1


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        ckpt = Path(cls._tmp.name) / "tiny"
        write_tiny_checkpoint(ckpt)

        engine = JaxGemmaEngine(
            model_id="synthetic/tiny", kv_cache_dtype="bf16",
            quant_mode="fp16", max_model_len=64,
        )
        engine.load(local_dir=str(ckpt))

        srv.ENGINE = engine
        srv.TOKENIZER = StubTokenizer()
        srv.MODEL_ID = "synthetic/tiny"
        cls.client = TestClient(srv.app)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_health_reports_ready_jax_backend(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["backend"], "jax")
        self.assertIsNotNone(body["device"])

    def test_models_endpoint(self):
        body = self.client.get("/v1/models").json()
        self.assertEqual(body["data"][0]["id"], "synthetic/tiny")

    def test_chat_template_accepts_transformers_5_mapping(self):
        original = srv.TOKENIZER

        class MappingTokenizer(StubTokenizer):
            def apply_chat_template(
                self, messages, tokenize=True, add_generation_prompt=True
            ):
                return {"input_ids": [11, 12, 13]}

        try:
            srv.TOKENIZER = MappingTokenizer()
            ids = srv._chat_prompt_ids(
                [srv.ChatMessage(role="user", content="hello")])
        finally:
            srv.TOKENIZER = original
        self.assertEqual(ids, [11, 12, 13])

    def test_chat_completion_non_streaming(self):
        r = self.client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello tpu"}],
            "max_tokens": 5, "temperature": 0.0,
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertTrue(body["choices"][0]["message"]["content"])
        usage = body["usage"]
        self.assertEqual(usage["completion_tokens"], 5)
        self.assertGreater(usage["prompt_tokens"], 0)
        self.assertGreater(usage["decode_tokens_per_second"], 0)
        self.assertIn("prefill_ms", usage)

    def test_chat_completion_streaming_sse(self):
        with self.client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "stream me"}],
            "max_tokens": 4, "temperature": 0.0, "stream": True,
        }) as r:
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
            payloads = [
                line[len("data: "):]
                for line in r.iter_lines()
                if line.startswith("data: ")
            ]

        self.assertEqual(payloads[-1], "[DONE]")
        chunks = [json.loads(p) for p in payloads[:-1]]
        content = [c["choices"][0]["delta"].get("content") for c in chunks]
        self.assertEqual(len([c for c in content if c]), 4, "expected one chunk per token")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
        self.assertTrue(all(c["object"] == "chat.completion.chunk" for c in chunks))

    def test_streaming_and_non_streaming_agree_greedily(self):
        payload = {
            "messages": [{"role": "user", "content": "determinism"}],
            "max_tokens": 6, "temperature": 0.0,
        }
        plain = self.client.post("/v1/chat/completions", json=payload).json()
        with self.client.stream("POST", "/v1/chat/completions",
                                json={**payload, "stream": True}) as r:
            chunks = [
                json.loads(line[len("data: "):])
                for line in r.iter_lines()
                if line.startswith("data: ") and not line.endswith("[DONE]")
            ]
        streamed = "".join(
            c["choices"][0]["delta"].get("content") or "" for c in chunks
        )
        self.assertEqual(streamed.strip(), plain["choices"][0]["message"]["content"])

    def test_text_completion(self):
        r = self.client.post("/v1/completions", json={
            "prompt": "raw completion", "max_tokens": 3, "temperature": 0.0,
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["object"], "text_completion")
        self.assertEqual(body["usage"]["completion_tokens"], 3)

    def test_metrics_exposes_jax_gauges(self):
        self.client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "metrics"}], "max_tokens": 3,
        })
        text = self.client.get("/metrics").text
        for metric in (
            "tpu_jax_requests_total",
            "tpu_jax_decode_tokens_per_second",
            "tpu_jax_prefill_milliseconds",
            "tpu_jax_weight_bytes",
        ):
            self.assertIn(metric, text)
        self.assertRegex(text, r"tpu_jax_weight_bytes\{model=\"synthetic/tiny\"\} [1-9]")

    def test_oversized_prompt_returns_500_not_crash(self):
        r = self.client.post("/v1/completions", json={
            "prompt": "x" * 400, "max_tokens": 4,
        })
        self.assertEqual(r.status_code, 500)
        self.assertIn("max_model_len", r.json()["detail"])

    def test_503_before_engine_ready(self):
        saved = srv.ENGINE
        srv.ENGINE = None
        try:
            r = self.client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hi"}],
            })
            self.assertEqual(r.status_code, 503)
        finally:
            srv.ENGINE = saved


if __name__ == "__main__":
    unittest.main()
