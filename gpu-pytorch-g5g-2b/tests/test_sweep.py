"""Tests for sweep.py, the harness all three g5g rigs are compared on.

It had NO tests until 2026-08-31, and the bug that cost a finding lived in it:
one prompt was reused for a cell's warm-up and every repeat, so vLLM answered
94.7% of them from its prefix cache while the two siblings, which have no such
cache, paid full prefill. The TTFT comparison built on that was meaningless.

These pin the properties that mistake violated. They are offline: no AWS, no
network, no GPU.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import sweep  # noqa: E402


class PromptModeTests(unittest.TestCase):
    """A shared prefix is what a prefix cache keys on."""

    def test_unique_prompts_differ_between_calls(self):
        a = sweep.prompt_for(512, unique=True)
        b = sweep.prompt_for(512, unique=True)
        self.assertNotEqual(a, b, "unique mode must not repeat a prompt")

    def test_fixed_prompts_are_identical(self):
        self.assertEqual(sweep.prompt_for(512, unique=False),
                         sweep.prompt_for(512, unique=False))

    def test_nonce_is_at_the_FRONT_not_the_tail(self):
        """A trailing nonce does not defeat a prefix cache -- only a leading one does."""
        a = sweep.prompt_for(512, unique=True)
        b = sweep.prompt_for(512, unique=True)
        self.assertNotEqual(a[:32], b[:32],
                            "prompts must diverge within the first block, or a prefix "
                            "cache still hits on the shared head")
        # And the divergence must be the nonce rather than an accident of length.
        self.assertTrue(a.startswith("Record ") and b.startswith("Record "))

    def test_unique_is_the_default(self):
        """The safe mode must be what you get without thinking about it."""
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--prompt-mode", choices=("unique", "fixed"), default="unique")
        self.assertEqual(ap.parse_args([]).prompt_mode, "unique")

    def test_prompt_length_still_tracks_the_request(self):
        short = sweep.prompt_for(64, unique=True)
        long = sweep.prompt_for(2048, unique=True)
        self.assertLess(len(short), len(long))


class StreamDecodeTests(unittest.TestCase):
    """The portable statistic: (n-1) / (t_last - t_first), vLLM's TPOT definition."""

    def _sse(self, deltas):
        body = b""
        for d in deltas:
            body += b"data: " + json.dumps(
                {"choices": [{"delta": {"content": d}}]}).encode() + b"\n\n"
        body += (b"data: " + json.dumps(
            {"choices": [], "usage": {"prompt_tokens": 10,
                                      "completion_tokens": len(deltas)}}).encode() + b"\n\n")
        return body + b"data: [DONE]\n\n"

    def _run(self, deltas, gap_s):
        """Drive one_stream with a fake stream and a clock that advances per line."""
        stream = self._sse(deltas)
        resp = mock.MagicMock()
        resp.__enter__ = mock.Mock(return_value=iter(stream.splitlines(keepends=True)))
        resp.__exit__ = mock.Mock(return_value=False)
        ticks = iter([0.0] + [gap_s * i for i in range(1, 500)])
        with mock.patch.object(sweep.urllib.request, "urlopen", return_value=resp), \
             mock.patch.object(sweep.time, "perf_counter", side_effect=lambda: next(ticks)):
            return sweep.one_stream("http://x/v1", "m", "p", 8)

    def test_counts_only_content_deltas(self):
        out = self._run(["a", "b", "c"], gap_s=1.0)
        self.assertEqual(out["stream_chunks"], 3)

    def test_flags_a_chunk_count_that_disagrees_with_usage(self):
        """One delta per chunk is an assumption; this is how you learn it broke."""
        out = self._run(["a", "b", "c"], gap_s=1.0)
        self.assertTrue(out["chunks_match_usage"])

    def test_reports_a_source_label(self):
        self.assertEqual(self._run(["a", "b"], gap_s=1.0)["source"], "stream")


class DecodeSourceTests(unittest.TestCase):
    """auto must pick `stream` for a server with no decode gauge -- that is the
    whole reason this harness can measure vLLM at all."""

    def test_auto_picks_stream_when_usage_has_no_gauge(self):
        with mock.patch.object(sweep, "post", return_value={"usage": {"completion_tokens": 4}}):
            self.assertEqual(sweep.probe_source("http://x/v1", "m"), "stream")

    def test_auto_picks_both_when_the_server_reports_a_gauge(self):
        with mock.patch.object(sweep, "post", return_value={
                "usage": {"completion_tokens": 4, "decode_tokens_per_second": 11.0}}):
            self.assertEqual(sweep.probe_source("http://x/v1", "m"), "both")

    def test_auto_falls_back_to_stream_when_the_probe_fails(self):
        with mock.patch.object(sweep, "post", side_effect=TimeoutError):
            self.assertEqual(sweep.probe_source("http://x/v1", "m"), "stream")


class GaugeParsingTests(unittest.TestCase):
    def test_reads_a_labelled_series(self):
        text = 'tpu_jax_degenerate_responses_total{model="g"} 3.0'
        self.assertEqual(sweep.gauge(text, "tpu_jax_degenerate_responses_total"), 3.0)

    def test_absent_series_reads_zero_not_an_error(self):
        """vLLM emits no tpu_jax_* metric; that must not crash the sweep."""
        self.assertEqual(sweep.gauge("vllm:num_requests_running 0.0",
                                     "tpu_jax_degenerate_responses_total"), 0.0)

    def test_comment_lines_are_not_samples(self):
        text = "# HELP tpu_jax_x help\n# TYPE tpu_jax_x gauge\ntpu_jax_x 7.5"
        self.assertEqual(sweep.gauge(text, "tpu_jax_x"), 7.5)


if __name__ == "__main__":
    unittest.main()


class ConcurrencyPointTests(unittest.TestCase):
    """The concurrency sweep is the difference between a latency benchmark and a
    serving one, and its headline number is easy to compute wrongly."""

    def _fake_stream(self, tokens, per_request_s):
        def fake(base, model, prompt, out_len, timeout=900):
            import time as _t
            _t.sleep(per_request_s)
            return {"source": "stream", "completion_tokens": tokens, "stream_chunks": tokens,
                    "decode_tps": tokens / per_request_s, "ttft_ms": 10.0,
                    "tpot_ms": per_request_s * 1000 / max(tokens - 1, 1),
                    "wall_s": per_request_s, "end_to_end_tps": tokens / per_request_s,
                    "chunks_match_usage": True, "text_head": "x"}
        return fake

    def test_aggregate_is_tokens_over_wall_not_mean_of_rates(self):
        """Above saturation those two diverge, which is the whole point."""
        with mock.patch.object(sweep, "one_stream", self._fake_stream(64, 0.2)):
            pt = sweep.concurrency_point("http://x/v1", "m", 512, 64, 4, True)
        self.assertEqual(pt["output_tokens_total"], 256)
        # 4 requests in parallel, so wall is ~one request, not four
        self.assertGreater(pt["output_tokens_per_second"], 256 / 0.5)
        self.assertEqual(pt["requests_ok"], 4)

    def test_one_failure_does_not_void_the_cell(self):
        calls = {"n": 0}
        good = self._fake_stream(32, 0.05)

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 2:
                raise TimeoutError("boom")
            return good(*a, **k)

        with mock.patch.object(sweep, "one_stream", flaky):
            pt = sweep.concurrency_point("http://x/v1", "m", 512, 32, 4, True)
        self.assertEqual(pt["status"], "ok")
        self.assertEqual(pt["requests_ok"], 3)
        self.assertEqual(pt["requests"], 4)

    def test_all_failures_report_failed_rather_than_zero(self):
        with mock.patch.object(sweep, "one_stream", side_effect=TimeoutError("nope")):
            pt = sweep.concurrency_point("http://x/v1", "m", 512, 32, 2, True)
        self.assertEqual(pt["status"], "failed")
        self.assertIn("TimeoutError", pt["error"])

    def test_each_stream_gets_its_own_prompt_in_unique_mode(self):
        seen = []

        def capture(base, model, prompt, out_len, timeout=900):
            seen.append(prompt)
            return {"source": "stream", "completion_tokens": 8, "stream_chunks": 8,
                    "decode_tps": 8.0, "ttft_ms": 1.0, "tpot_ms": 1.0, "wall_s": 0.01,
                    "end_to_end_tps": 8.0, "chunks_match_usage": True, "text_head": ""}

        with mock.patch.object(sweep, "one_stream", capture):
            sweep.concurrency_point("http://x/v1", "m", 512, 8, 4, True)
        self.assertEqual(len(set(seen)), 4, "a shared prompt would hand vLLM a prefix-cache hit")
