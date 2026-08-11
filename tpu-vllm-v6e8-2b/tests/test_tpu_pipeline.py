"""Full Integration Test Suite for Option A the PyTorch TPU backend Pipeline."""

import unittest
from fastapi.testclient import TestClient
import tpu_compiler
from tpu_engine import TPUEngine, pad_to_multiple
from tpu_server import create_app


class DummyModel:

    def forward(self, input_ids):

        class DummyOutput:

            def __init__(self, batch, seq_len):
                import torch

                self.logits = torch.randn(batch, seq_len, 100)

        return DummyOutput(input_ids.shape[0], input_ids.shape[1])

    def __call__(self, input_ids):
        return self.forward(input_ids)


class DummyTokenizer:

    def encode(self, text):
        return [ord(c) % 100 for c in text]

    def decode(self, tokens):
        return "".join([chr(t + 65) for t in tokens])


class TestTPUPipeline(unittest.TestCase):

    def test_pad_to_multiple(self):
        self.assertEqual(pad_to_multiple(100, 128), 128)
        self.assertEqual(pad_to_multiple(128, 128), 128)
        self.assertEqual(pad_to_multiple(129, 128), 256)

    def test_engine_and_server_pipeline(self):
        engine = TPUEngine(
            model=DummyModel(),
            tokenizer=DummyTokenizer(),
            device="cpu",
            force_staging=True,
        )

        # Test engine generation
        prompt_tokens = [1, 2, 3, 4]
        gen_tokens = list(engine.generate(prompt_tokens, max_new_tokens=5, temperature=0.0))
        self.assertEqual(len(gen_tokens), 5)

        # Test server HTTP endpoints
        app = create_app(engine=engine)
        client = TestClient(app)

        # Health endpoint
        res = client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "healthy")

        # Chat completions non-streaming
        chat_req = {
            "model": "google/gemma-4-E2B-it",
            "messages": [{"role": "user", "content": "Hello TPU"}],
            "max_tokens": 5,
            "stream": False,
        }
        res = client.post("/v1/chat/completions", json=chat_req)
        self.assertEqual(res.status_code, 200)
        json_resp = res.json()
        self.assertIn("choices", json_resp)
        self.assertEqual(len(json_resp["choices"]), 1)


if __name__ == "__main__":
    unittest.main()
