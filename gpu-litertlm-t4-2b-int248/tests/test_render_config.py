import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import render_config  # noqa: E402

RIG = Path(__file__).resolve().parent.parent


class ReadEnvTest(unittest.TestCase):
    def test_quotes_comments_and_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.env"
            p.write_text('# c\nA="NVIDIA GeForce"\nB=gpu\nC=\n\nnot a line\n')
            env = render_config.read_env(p)
        self.assertEqual(env, {"A": "NVIDIA GeForce", "B": "gpu", "C": ""})


class BuildConfigTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in ("MODEL_ID", "BACKEND", "CACHE", "SPECULATIVE_DECODING", "MAX_NUM_TOKENS"):
            os.environ.pop(key, None)

    def test_rig_env_renders(self):
        cfg = render_config.build_config(render_config.read_env(RIG / "tpu.env"))
        env = render_config.read_env(RIG / "tpu.env")
        model = cfg["models"][env["MODEL_ID"]]
        self.assertEqual(model["backend"], env["BACKEND"])
        self.assertEqual(model["speculative_decoding"], env["SPECULATIVE_DECODING"])
        if not env.get("MAX_NUM_TOKENS"):
            self.assertNotIn("max_num_tokens", model)

    def test_environment_wins(self):
        os.environ["SPECULATIVE_DECODING"] = "off"
        cfg = render_config.build_config({"MODEL_ID": "m", "SPECULATIVE_DECODING": "on"})
        self.assertEqual(cfg["models"]["m"]["speculative_decoding"], "off")

    def test_rejects_bad_value(self):
        with self.assertRaises(ValueError):
            render_config.build_config({"MODEL_ID": "m", "BACKEND": "cuda"})

    def test_requires_model_id(self):
        with self.assertRaises(ValueError):
            render_config.build_config({"BACKEND": "gpu"})

    def test_max_tokens_is_int(self):
        cfg = render_config.build_config({"MODEL_ID": "m", "MAX_NUM_TOKENS": "8192"})
        self.assertEqual(cfg["models"]["m"]["max_num_tokens"], 8192)


if __name__ == "__main__":
    unittest.main()
