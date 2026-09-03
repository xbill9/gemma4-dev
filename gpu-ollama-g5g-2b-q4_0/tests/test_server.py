"""Offline regression tests for the G5g Ollama MCP server.

No AWS, no network, no GPU, no compiler.

**THE POINT OF THIS FILE IS THE RENDERED BOOTSTRAP.** When `gpu-pytorch-g5g-2b`
was forked from the JAX rig it shipped FIVE fatal bugs that all survived to the
first real launch — `import jax` in a rig with no jax, an ExecStart naming a file
not in the payload, three serve flags the server did not define, a quoted pip
spec holding two packages, and `torch.compile(backend="tpu")` on CUDA. 89 offline
tests passed throughout, because **not one of them asserted on the rendered
bootstrap**. A fork rewrites the parts you read and leaves the parts you execute.

So the bootstrap tests below are not decoration. They are the tests.
"""

import re
import sys
import unittest
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import server  # noqa: E402

# Spelled out rather than read from `server`, so a change to the served artifact
# shows up as a test edit rather than passing vacuously against whatever the
# module happens to hold.
TAG = "gemma4:e2b-it-qat"
REPO = TAG          # the "model" this rig serves is an Ollama tag, not a repo
DEFAULT_SIZE = "g5g.2xlarge"
SWAP_SIZE = "g5g.xlarge"

# Every OLLAMA_* variable the daemon defines, from envconfig/config.go
# (2026-09-02). _serve_env must set nothing outside this set.
KNOWN_OLLAMA_VARS = {
    "OLLAMA_AUTH", "OLLAMA_CONTEXT_LENGTH", "OLLAMA_DEBUG",
    "OLLAMA_DEBUG_LOG_REQUESTS", "OLLAMA_EDITOR", "OLLAMA_FLASH_ATTENTION",
    "OLLAMA_GO_TEMPLATE", "OLLAMA_GPU_OVERHEAD", "OLLAMA_HOST",
    "OLLAMA_IGPU_ENABLE", "OLLAMA_KEEP_ALIVE", "OLLAMA_KV_CACHE_TYPE",
    "OLLAMA_LLM_LIBRARY", "OLLAMA_LOAD_TIMEOUT", "OLLAMA_MAX_LOADED_MODELS",
    "OLLAMA_MAX_QUEUE", "OLLAMA_MAX_TRANSFER_STREAMS", "OLLAMA_MODELS",
    "OLLAMA_NOHISTORY", "OLLAMA_NOPRUNE", "OLLAMA_NO_CLOUD",
    "OLLAMA_NUM_PARALLEL", "OLLAMA_ORIGINS", "OLLAMA_REMOTES",
    "OLLAMA_SCHED_SPREAD", "OLLAMA_VULKAN",
}


class ServeEnv(unittest.TestCase):
    """`ollama serve` takes no arguments; all configuration is environment.

    That inverts the sibling's hazard. An unknown FLAG makes argparse exit 2 and
    the unit crash-loop -- loud, and the PyTorch fork's first-launch failure.
    An unknown ENVIRONMENT VARIABLE is silently ignored by the daemon, which
    serves happily at its own defaults. So these tests matter more here, not
    less.
    """

    def setUp(self):
        self.env = server._serve_env(TAG, DEFAULT_SIZE)

    def test_sets_nothing_the_daemon_does_not_define(self):
        unknown = set(self.env) - KNOWN_OLLAMA_VARS
        self.assertEqual(
            unknown, set(),
            f"variables Ollama does not define (silently ignored): {unknown}",
        )

    def test_the_module_var_list_matches_the_daemon(self):
        self.assertEqual(set(server._KNOWN_OLLAMA_VARS), KNOWN_OLLAMA_VARS)

    def test_cuda_variant_is_pinned(self):
        # Unpinned, a CUDA 13 driver selects cuda_v13 -- PTX for every arch and
        # native SASS for none -- and JITs at load. The llama.cpp sibling builds
        # native sm_75, so pinning is what makes the pair comparable.
        self.assertEqual(self.env["OLLAMA_LLM_LIBRARY"], "cuda_v12")

    def test_context_is_pinned_not_derived_from_vram(self):
        # Ollama's default is 0 = "4k/32k/256k based on VRAM", so two instance
        # sizes would silently get two contexts.
        self.assertEqual(self.env["OLLAMA_CONTEXT_LENGTH"], str(server.CONTEXT_SIZE))
        self.assertNotEqual(self.env["OLLAMA_CONTEXT_LENGTH"], "0")

    def test_keep_alive_is_forever(self):
        # Default 5m. A sweep pausing longer reloads the model and records the
        # reload as latency.
        self.assertTrue(self.env["OLLAMA_KEEP_ALIVE"].startswith("-"))

    def test_binds_the_family_port_not_ollamas_own(self):
        # Ollama's default is 127.0.0.1:11434 -- both the port and the interface
        # would be wrong: bound to loopback, nothing off-box could reach it.
        self.assertEqual(self.env["OLLAMA_HOST"], f"0.0.0.0:{server.LLAMA_PORT}")

    def test_blob_store_is_on_the_sized_volume(self):
        self.assertTrue(self.env["OLLAMA_MODELS"].startswith(server.APP_DIR))


class RenderedBootstrap(unittest.TestCase):
    """The tests the PyTorch fork did not have."""

    @classmethod
    def setUpClass(cls):
        cls.ud = server._user_data(TAG, DEFAULT_SIZE)
        cls.code = [
            line for line in cls.ud.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    def test_it_renders_at_all(self):
        # str.format() raises on a stray literal brace, which is how a bootstrap
        # edit breaks the deploy without touching anything that looks like code.
        self.assertGreater(len(self.ud), 2000)

    def test_no_executable_line_mentions_a_foreign_runtime(self):
        bad = [
            line for line in self.code
            if re.search(r"\btorch\b|\bjax\b|cmake|nvcc|llama-server|--hf-repo",
                         line, re.I)
        ]
        self.assertEqual(bad, [], f"stale runtime in the bootstrap: {bad}")

    def test_there_is_no_build(self):
        # The whole asymmetry against the llama.cpp sibling. If a compiler ever
        # appears here, the two rigs stop differing in the way they exist to.
        for tool in ("build-essential", "cmake", "ccache", "GGML_CUDA",
                     "CMAKE_CUDA_ARCHITECTURES"):
            self.assertNotIn(tool, self.ud)

    def test_execstart_is_bare(self):
        # `ollama serve` takes no model arguments; everything is environment.
        self.assertIn(f"ExecStart={server.APP_DIR}/bin/ollama serve", self.ud)
        self.assertNotIn("ollama serve --", self.ud)

    def test_every_serve_var_reaches_the_environment_file(self):
        # The unit's ExecStart carries nothing, so a variable that fails to reach
        # the EnvironmentFile is simply not applied -- silently.
        for key, value in server._serve_env(TAG, DEFAULT_SIZE).items():
            self.assertIn(f"{key}={value}", self.ud)

    def test_pinned_cuda_variant_is_asserted_present_in_the_bundle(self):
        # OLLAMA_LLM_LIBRARY bypasses autodetection; it does NOT create a library
        # that is not there. A bad pin makes the daemon fall back, possibly to
        # CPU, with no error.
        self.assertIn(f"lib/ollama/{server.OLLAMA_LLM_LIBRARY}", self.ud)
        self.assertRegex(self.ud, r"FATAL: OLLAMA_LLM_LIBRARY[\s\S]{0,500}?exit 1")

    def test_the_model_is_asserted_to_be_in_vram(self):
        # Ollama chooses its own offload and cannot be told to fail, so the only
        # honest check is where the bytes ended up after a load.
        self.assertIn("size_vram", self.ud)
        self.assertIn("/api/ps", self.ud)
        self.assertIn("the model is on the CPU", self.ud)

    def test_verify_runs_after_the_pull_and_before_install_done(self):
        self.assertLess(self.ud.index("ollama pull"), self.ud.index("verify_gpu\n"))
        self.assertLess(
            self.ud.index("verify_gpu\n"),
            self.ud.index(f"touch {server.APP_DIR}/INSTALL_DONE"),
        )

    def test_it_waits_for_the_daemon_before_pulling(self):
        # `enable --now` returns when the process starts, not when it listens.
        self.assertLess(
            self.ud.index("curl -fsS http://127.0.0.1"), self.ud.index("ollama pull")
        )

    def test_unattended_upgrades_is_masked(self):
        self.assertIn("mask apt-daily-upgrade.service", self.ud)

    def test_hf_token_fetch_is_not_traced(self):
        # Inert on this rig -- Ollama pulls from its own registry -- but the
        # xtrace discipline is the part worth never losing.
        i = self.ud.index("secretsmanager get-secret-value")
        self.assertIn("set +x", self.ud[:i])
        self.assertIn("set -x", self.ud[i:])

    def test_token_is_never_in_user_data_itself(self):
        self.assertNotIn("HF_TOKEN=hf_", self.ud)

    def test_swapfile_renders_for_every_size_at_or_below_16gib(self):
        self.assertIn("mkswap", server._user_data(TAG, SWAP_SIZE))
        self.assertIn("mkswap", self.ud)
        self.assertNotIn("mkswap", server._user_data(TAG, "g5g.4xlarge"))

    def test_mkswap_takes_no_busybox_flag(self):
        code = [
            line for line in server._user_data(TAG, SWAP_SIZE).splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertEqual([line for line in code if "mkswap -q" in line], [])

    def test_home_is_set_in_the_environment_file(self):
        """MEASURED 2026-09-02 on i-0db1233beb07f0da7 -- the rig's first launch.

        systemd gives a unit a minimal environment with no $HOME, and
        `ollama serve` exits 1 with `Error: $HOME is not defined` on every start.
        `Restart=on-failure` then crash-looped it 13 times in ~2 minutes while the
        bootstrap's daemon-ready poll sat on a dead port.

        **Setting OLLAMA_MODELS does not substitute for it**: Ollama reads $HOME
        for its own config and signing key regardless of where blobs live. This is
        the only variable here that is NOT an OLLAMA_* name, which is exactly why
        it was missed.
        """
        self.assertIn("HOME=/root", self.ud)

    def test_the_daemon_ready_wait_is_fatal(self):
        """The same launch showed why a timeout must not fall through.

        The loop expired after 120s and carried on to `ollama pull` against a dead
        port, so a crash-looping daemon surfaced as a confusing pull failure two
        stages later instead of naming itself.
        """
        self.assertIn("FATAL: ollama did not bind", self.ud)
        self.assertRegex(self.ud, r"FATAL: ollama did not bind[\s\S]{0,400}?exit 1")

    def test_stage_markers_exist_for_every_long_phase(self):
        for stage in ("runtime-deps", "download-extract", "verify-bundle",
                      "service-start", "daemon-ready", "gpu-verify"):
            self.assertIn(f"stage {stage}", self.ud)


class TimingProbe(unittest.TestCase):
    """Ollama has no /metrics. The gauge comes from /api/generate's own timings.

    **Every duration Ollama returns is a Go `time.Duration`, i.e. NANOSECONDS.**
    Dividing by 1e6 instead of 1e9 yields a rate 1000x too low, which reads as a
    catastrophically broken rig rather than as a unit bug. These tests pin the
    unit, because nothing else would catch it.
    """

    # 300 tokens generated in 10s of decode = 30 tok/s.
    # 512 prompt tokens prefilled in 2s     = 256 tok/s (never added in).
    BODY: ClassVar[dict] = {
        "eval_count": 300, "eval_duration": 10_000_000_000,
        "prompt_eval_count": 512, "prompt_eval_duration": 2_000_000_000,
        "load_duration": 0, "total_duration": 12_500_000_000,
    }

    def test_decode_rate_uses_nanoseconds(self):
        self.assertAlmostEqual(server._decode_rate(self.BODY), 30.0)

    def test_decode_rate_excludes_prefill(self):
        # Folding in the prompt tokens and prefill seconds would give 67.7 -- a
        # different and incomparable number.
        self.assertNotAlmostEqual(server._decode_rate(self.BODY), 812 / 12.0, places=1)

    def test_prefill_is_reported_separately(self):
        self.assertAlmostEqual(server._prefill_rate(self.BODY), 256.0)

    def test_rates_are_none_when_nothing_was_generated(self):
        self.assertIsNone(server._decode_rate({"eval_count": 0, "eval_duration": 0}))
        self.assertIsNone(server._decode_rate({}))

    def test_there_is_no_prom_parser_left(self):
        # Ollama registers no Prometheus route (verified against server/routes.go).
        # A scraper here would silently return nothing.
        self.assertFalse(hasattr(server, "_parse_prom"))


class Identity(unittest.TestCase):
    def test_rig_name_matches_the_directory(self):
        self.assertEqual(server.RIG_NAME, ROOT.name)

    def test_rig_name_parses_as_five_slots(self):
        slots = server.RIG_NAME.split("-")
        self.assertEqual(len(slots), 5, "platform-runtime-hardware-model-encoding")
        self.assertEqual(slots[0], "gpu")
        self.assertEqual(slots[1], "ollama")
        self.assertEqual(slots[4], "q4_0")   # the encoding, not `gguf`

    def test_managed_by_tag_scopes_discovery_to_this_rig(self):
        self.assertEqual(server.MANAGED_BY, server.RIG_NAME)

    def test_build_id_names_version_and_cuda_variant(self):
        self.assertIn(server.OLLAMA_VERSION, server._build_id())
        self.assertIn(server.OLLAMA_LLM_LIBRARY, server._build_id())

    def test_ollama_version_is_pinned(self):
        self.assertRegex(server.OLLAMA_VERSION, r"^v\d+\.\d+\.\d+$")
        self.assertIn(server.OLLAMA_VERSION, server.OLLAMA_TARBALL_URL)

    def test_tarball_is_the_generic_arm64_bundle_not_jetpack(self):
        # The jetpack bundles are for Jetson and carry a different arch set;
        # "arm64 + NVIDIA" is not a reason to reach for one.
        self.assertIn("ollama-linux-arm64.tar.zst", server.OLLAMA_TARBALL_URL)
        self.assertNotIn("jetpack", server.OLLAMA_TARBALL_URL)

    def test_model_is_the_qat_tag_not_the_plain_one(self):
        # `gemma4:e2b` is a DIFFERENT artifact: one 7.162 GB layer, no projector.
        self.assertEqual(server.MODEL_NAME, TAG)
        self.assertNotIn(server.MODEL_NAME, server.MODEL_TAG_ALTERNATIVES)

    def test_no_llamacpp_machinery_survived_the_fork(self):
        for gone in ("_serve_argv", "LLAMA_CPP_REF", "CUDA_ARCH", "BUILD_DIR",
                     "MODEL_FILE", "N_GPU_LAYERS", "_parse_prom"):
            self.assertFalse(
                hasattr(server, gone),
                f"{gone} is a llama.cpp-rig name and should not exist here.",
            )


class EnvIsHonest(unittest.TestCase):
    """tpu.env must not carry keys nothing reads.

    The PyTorch fork inherited QUANT_MODE/PLE_BITS/INT8_LM_HEAD/DTYPE and carried
    them for a month, until its own CLAUDE.md had to warn that they were inert.
    A key present in an env file reads as a supported knob.
    """

    def test_every_key_is_read_by_server(self):
        env = [
            line.split("=", 1)[0]
            for line in (ROOT / "tpu.env").read_text().splitlines()
            if re.match(r"^[A-Z_]+=", line)
        ]
        read = set(re.findall(r'os\.getenv\(\s*"([A-Z_]+)"', (ROOT / "server.py").read_text()))
        inert = [k for k in env if k not in read]
        self.assertEqual(inert, [], f"tpu.env keys nothing reads: {inert}")

    def test_env_pins_the_cuda_variant(self):
        text = (ROOT / "tpu.env").read_text()
        self.assertIn("OLLAMA_LLM_LIBRARY=cuda_v12", text)

    def test_env_pins_context_and_keep_alive(self):
        text = (ROOT / "tpu.env").read_text()
        self.assertIn("CONTEXT_SIZE=4096", text)
        self.assertIn("OLLAMA_KEEP_ALIVE=-1", text)


class SkillSnapshot(unittest.TestCase):
    def test_snapshot_list_carries_no_serving_payload(self):
        # Read the `names` tuple the script actually copies. refresh_skill.py's
        # comment names the sibling payload files it deliberately dropped, so
        # searching the whole file matches its own explanation.
        import ast as _ast
        tree = _ast.parse((ROOT / "refresh_skill.py").read_text())
        names = [
            _ast.literal_eval(node.value)
            for node in _ast.walk(tree)
            if isinstance(node, _ast.Assign)
            and getattr(node.targets[0], "id", None) == "names"
        ]
        self.assertEqual(len(names), 1, "expected exactly one `names` tuple")
        self.assertEqual(
            set(names[0]), {"server.py", "project-setup.sh", "requirements.txt"}
        )


if __name__ == "__main__":
    unittest.main()
